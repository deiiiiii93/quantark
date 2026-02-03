import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Union

from asset.equity.product.option.observation_schedule import ResolvedObservationRecord

class PDEEvent(ABC):
    """
    Represents a discrete event that modifies the PDE state at a specific time.
    """
    @abstractmethod
    def apply(
        self,
        state,
        t_idx: int,
        current_time: float,
        s_vec: np.ndarray,
        pricing_env
    ) -> None:
        """
        Apply the event to the state at the given time index.
        
        Args:
            state: PDESystemState object
            t_idx: Current time index
            current_time: Current time in years
            s_vec: Spatial grid (price)
            pricing_env: PricingEnvironment
        """
        pass

    @staticmethod
    def _get_df(pricing_env, start_time: float, end_time: Optional[float]) -> float:
        if end_time is None or end_time <= start_time:
            return 1.0
        df_end = pricing_env.get_discount_factor(end_time)
        df_start = pricing_env.get_discount_factor(start_time)
        if df_start == 0.0:
            return 0.0
        return float(df_end / df_start)

class KnockOutEvent(PDEEvent):
    """
    Applies Knock-Out logic: if barrier breached, value = payoff.
    """
    def __init__(self, record: ResolvedObservationRecord, is_reverse: bool, state_indices: Optional[List[int]] = None):
        self.record = record
        self.is_reverse = is_reverse
        self.state_indices = state_indices # If None, apply to all states
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_ref_id: Optional[int] = None
        self._mask_len: Optional[int] = None
        
    def apply(self, state, t_idx, current_time, s_vec, pricing_env):
        barrier = float(self.record.barrier)
        payoff = float(self.record.payoff or 0.0)
        
        # UP barrier for standard KO
        s_vec_id = id(s_vec)
        if (
            self._mask_cache is None
            or self._mask_ref_id != s_vec_id
            or self._mask_len != len(s_vec)
        ):
            if self.is_reverse:
                mask = s_vec <= barrier
            else:
                mask = s_vec >= barrier
            self._mask_cache = mask
            self._mask_ref_id = s_vec_id
            self._mask_len = len(s_vec)
        else:
            mask = self._mask_cache
            
        df = self._get_df(pricing_env, current_time, self.record.settlement_time)
        value = payoff * df
        
        # Apply to selected states (KO terminates the product if all states, or just resets if one)
        if self.state_indices is None:
            state.grids[mask, t_idx, :] = value
        else:
            for idx in self.state_indices:
                state.grids[mask, t_idx, idx] = value

class KnockInEvent(PDEEvent):
    """
    Applies Knock-In logic: copies values from Source(KI) to Target(No-KI).
    """
    def __init__(self, barrier: float, is_reverse: bool, source_idx: int, target_idx: int):
        self.barrier = barrier
        self.is_reverse = is_reverse
        self.source = source_idx
        self.target = target_idx
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_ref_id: Optional[int] = None
        self._mask_len: Optional[int] = None
        
    def apply(self, state, t_idx, current_time, s_vec, pricing_env):
        # DOWN barrier for standard KI
        s_vec_id = id(s_vec)
        if (
            self._mask_cache is None
            or self._mask_ref_id != s_vec_id
            or self._mask_len != len(s_vec)
        ):
            if self.is_reverse:
                mask = s_vec >= self.barrier
            else:
                mask = s_vec <= self.barrier
            self._mask_cache = mask
            self._mask_ref_id = s_vec_id
            self._mask_len = len(s_vec)
        else:
            mask = self._mask_cache
            
        state.grids[mask, t_idx, self.target] = state.grids[mask, t_idx, self.source]

class PhoenixCouponEvent(PDEEvent):
    """
    Applies Phoenix coupon logic with memory.
    """
    def __init__(
        self, 
        barrier: float, 
        base_coupon: float, 
        accumulated_vector: np.ndarray,
        settlement_time: Optional[float],
        is_reverse: bool,
        is_memory: bool,
        v0_indices: List[int],
        v1_indices: List[int]
    ):
        self.barrier = barrier
        self.base_coupon = base_coupon
        self.accumulated_vector = accumulated_vector
        self.settlement_time = settlement_time
        self.is_reverse = is_reverse
        self.is_memory = is_memory
        self.v0_indices = v0_indices # Indices for Not-KI states [k=0, 1, ...]
        self.v1_indices = v1_indices # Indices for KI states [k=0, 1, ...]
        
    def apply(self, state, t_idx, current_time, s_vec, pricing_env):
        df = self._get_df(pricing_env, current_time, self.settlement_time)
        
        if self.is_reverse:
            pay_mask = s_vec <= self.barrier
        else:
            pay_mask = s_vec >= self.barrier
            
        # We assume the state at t_idx currently holds V(t+, k) (values after observation)
        # We update it to V(t-, k) (values before observation)
        
        # Extract current grids (N, num_states)
        # IMPORTANT: Take a copy to avoid reading updated values during the loop
        grid_slice = state.grids[:, t_idx, :].copy()
        
        # Values if Paid: V(t+, 0) + Coupon
        # We need state 0 (memory 0) for both V0 and V1 groups
        v0_reset_val = grid_slice[:, self.v0_indices[0]] # V0(t+, 0)
        v1_reset_val = grid_slice[:, self.v1_indices[0]] # V1(t+, 0)
        
        max_k = len(self.v0_indices) - 1
        
        # Process V0 (Not KI) states
        for k, idx in enumerate(self.v0_indices):
            # Calculate total coupon for this memory state
            acc_coupon = self.accumulated_vector[k] if self.is_memory else 0.0
            total_pay = (self.base_coupon + acc_coupon) * df
            
            # Value if paid
            val_paid = v0_reset_val + total_pay
            
            # Value if missed
            # V(t+, k+1) if memory, else V(t+, 0) usually? 
            # Non-memory Phoenix: if miss, nothing accumulated, just continue. So V(t+, 0).
            # Memory Phoenix: if miss, k increases.
            next_k = min(k + 1, max_k) if self.is_memory else 0
            val_missed = grid_slice[:, self.v0_indices[next_k]]
            
            # Update grid
            # If Pay: val_paid
            # If Miss: val_missed
            state.grids[pay_mask, t_idx, idx] = val_paid[pay_mask]
            state.grids[~pay_mask, t_idx, idx] = val_missed[~pay_mask]
            
        # Process V1 (KI) states
        # Logic is similar, but typically memory coupons still pay even if KI happened?
        # Yes, usually KI affects redemption, not coupons.
        for k, idx in enumerate(self.v1_indices):
            acc_coupon = self.accumulated_vector[k] if self.is_memory else 0.0
            total_pay = (self.base_coupon + acc_coupon) * df
            
            val_paid = v1_reset_val + total_pay
            
            next_k = min(k + 1, max_k) if self.is_memory else 0
            val_missed = grid_slice[:, self.v1_indices[next_k]]
            
            state.grids[pay_mask, t_idx, idx] = val_paid[pay_mask]
            state.grids[~pay_mask, t_idx, idx] = val_missed[~pay_mask]

class MaturityEvent(PDEEvent):
    """
    Forces the grid values to a specified maturity payoff function.
    Useful for multi-maturity products like KO-reset.
    """
    def __init__(self, payoff_func, state_indices: List[int]):
        self.payoff_func = payoff_func
        self.state_indices = state_indices
        
    def apply(self, state, t_idx, current_time, s_vec, pricing_env):
        # Calculate payoff for current spots
        # We assume payoff_func accepts (spot, pricing_env)
        # Vectorized apply? Or loop.
        # s_vec is numpy array. If payoff_func is vectorized, great.
        # BaseEquityProduct payoff methods are usually scalar.
        
        # Use simple list comp if not vectorizable (safe)
        # But we pass s_vec to solver, so we might want to vectorize.
        # Let's assume scalar func and map it.
        
        # We need to handle 'accumulated_coupons' if needed, but for simple V0 maturity it's 0
        
        # Precompute values
        values = np.array([self.payoff_func(s, pricing_env) for s in s_vec])
        
        for idx in self.state_indices:
            state.grids[:, t_idx, idx] = values
