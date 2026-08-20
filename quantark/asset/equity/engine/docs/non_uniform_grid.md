# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 2023

@author: yaofuxin

@description: The problem is to find a function Y(ε) that transforms a uniform mesh ε ∈ [0, 1] into a nonuniform mesh Y ∈ [Y_min, Y_max] that is denser near some critical points B_k. The density factors β and the critical points B_k are given. The function Y(ε) satisfies the ODE:

dY(ε) / dε = A * (sum(J_k(ε)^-2))^-0.5

where J_k(ε) = sqrt(β^2 + (Y(ε) - B_k)^2) and A is a constant that can be determined by the boundary conditions Y(0) = Y_min and Y(1) = Y_max.

The Runge-Kutta method is a numerical method that approximates the solution of an ODE by using a series of steps with different slopes. The most common version is the fourth-order Runge-Kutta method (RK4), which uses four slopes k_1, k_2, k_3, and k_4 at each step. The formula for RK4 is:

Y_n+1 = Y_n + h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)

where h is the step size and

k_1 = f(ε_n, Y_n)
k_2 = f(ε_n + h/2, Y_n + h/2 * k_1)
k_3 = f(ε_n + h/2, Y_n + h/2 * k_2)
k_4 = f(ε_n + h, Y_n + h * k_3)

and f is the right-hand side of the ODE.

To apply RK4 to your problem, you need to define the function f as:

f(ε, Y) = A * (sum(J_k(ε)^-2))^-0.5

and use the initial condition Y(0) = Y_min. You also need to choose a suitable step size h and a number of steps N such that ε_N = 1. Then you can iterate the RK4 formula from n = 0 to n = N to obtain the approximate values of Y(ε) at each step.

To find the value of A, you can use the final condition Y(1) = Y_max and solve for A using a root-finding method such as bisection or Newton's method.
"""


import numpy as np
from typing import List, Sequence, Union


class TravellRandallMultiStrikes:
    """
    Travella–Randall method for non-uniform meshes with multiple critical points.

    This class solves the ODE described above using a shooting method to find the
    constant A such that Y(0) = L and Y(1) = U. It returns a mesh of N+1 nodes on
    [L, U] concentrated around the critical points in K.
    """

    def __init__(
        self,
        L: Union[float, Sequence[float]],
        U: Union[float, Sequence[float]],
        K: Sequence[float],
        N: int,
        beta: float,
    ) -> None:
        self.L = float(L) if np.isscalar(L) else float(np.min(L))  # lower boundary
        self.U = float(U) if np.isscalar(U) else float(np.max(U))  # upper boundary
        self.K: List[float] = list(K) if not np.isscalar(K) else [float(K)]
        self.N = int(N)
        self.beta = float(beta)
        self.A = 0.0  # constant scaling factor determined by shooting
        self.h = 1.0 / self.N
        self._K_arr = np.asarray(self.K, dtype=float)

        # Validate inputs
        if not np.isfinite(self.L) or not np.isfinite(self.U):
            raise ValueError("L and U must be finite.")
        if self.U <= self.L:
            raise ValueError("Require U > L.")
        if self.N < 2:
            raise ValueError("N must be an integer >= 2.")
        if not np.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be a positive finite number.")
        if len(self._K_arr) == 0:
            raise ValueError("K must be a non-empty sequence of critical points.")
        # It's typical to require K within [L, U] for clustering; enforce softly
        if np.min(self._K_arr) < self.L or np.max(self._K_arr) > self.U:
            raise ValueError("All critical points K must lie within [L, U].")

    # Define the function f
    def __f__(self, e: float, y: float, A: float) -> float:
        # Vectorized evaluation over critical points
        j = np.sqrt(self.beta * self.beta + (y - self._K_arr) ** 2)
        s = np.sum(j**-2)
        return A * (s**-0.5)

    # Define the RK4 formula
    def __rk4__(self, e: float, y: float, A: float) -> float:
        h = self.h
        k1 = self.__f__(e, y, A)
        k2 = self.__f__(e + h * 0.5, y + h * 0.5 * k1, A)
        k3 = self.__f__(e + h * 0.5, y + h * 0.5 * k2, A)
        k4 = self.__f__(e + h, y + h * k3, A)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Define a function to check the final condition
    def __check__(self, A: float) -> float:
        e = 0.0
        y = self.L
        for _ in range(self.N):
            y = self.__rk4__(e, y, A)
            e += self.h
        return y - self.U

    # Use bisection method to find A
    def __find_A__(self) -> None:
        # Initial bracket for A. Start with [0, scale] and expand if needed.
        a_lo = 0.0
        a_hi = max(4.0 * self.U, self.U - self.L)
        f_lo = self.__check__(a_lo)
        f_hi = self.__check__(a_hi)

        # Expand upper bound until we bracket the root or hit a cap
        expand_iters = 0
        while f_lo * f_hi > 0.0 and expand_iters < 20:
            a_hi *= 2.0
            f_hi = self.__check__(a_hi)
            expand_iters += 1

        # Bisection
        tol = 1e-6
        max_iter = 100
        for _ in range(max_iter):
            if abs(a_hi - a_lo) <= tol:
                break
            a_mid = 0.5 * (a_lo + a_hi)
            f_mid = self.__check__(a_mid)
            if f_mid == 0.0:
                a_lo = a_hi = a_mid
                break
            if f_lo * f_mid < 0.0:
                a_hi, f_hi = a_mid, f_mid
            else:
                a_lo, f_lo = a_mid, f_mid
        self.A = 0.5 * (a_lo + a_hi)

    # generate the values of mesh at each step
    def generate_mesh(self) -> np.ndarray:
        """
        Generate and return the non-uniform mesh as a numpy array of length N+1.
        """
        self.__find_A__()
        e = 0.0
        y = self.L
        mesh = np.empty(self.N + 1, dtype=float)
        mesh[0] = y
        for n in range(1, self.N + 1):
            y = self.__rk4__(e, y, self.A)
            mesh[n] = y
            e += self.h
        return mesh
