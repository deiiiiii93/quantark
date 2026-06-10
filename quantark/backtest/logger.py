"""
Comprehensive logging system for backtest execution.

Provides multi-level logging for trades, state, events, and performance.
"""
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import sys


class BacktestLogger:
    """
    Comprehensive logging system for backtest execution.
    
    Provides separate loggers for:
    - Trade executions
    - Portfolio state
    - Strategy events
    - Performance metrics
    - General backtest info
    
    All logs can be written to files and/or console with configurable verbosity.
    
    Attributes:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_console: Whether to output to console
        enable_file: Whether to output to files
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: str = 'INFO',
        enable_console: bool = True,
        enable_file: bool = True,
        backtest_name: str = "backtest"
    ):
        """
        Initialize backtest logger.
        
        Args:
            log_dir: Directory for log files (created if doesn't exist)
            log_level: Logging level
            enable_console: Enable console output
            enable_file: Enable file output
            backtest_name: Name for this backtest (used in log filenames)
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.backtest_name = backtest_name
        
        # Create log directory if needed
        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        # In-memory storage for structured logs
        self.trade_logs: List[Dict[str, Any]] = []
        self.state_logs: List[Dict[str, Any]] = []
        self.event_logs: List[Dict[str, Any]] = []
        self.performance_logs: List[Dict[str, Any]] = []
    
    def _setup_loggers(self):
        """Set up separate loggers for different types of information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main logger
        self.logger = self._create_logger(
            'backtest_main',
            f"{self.backtest_name}_main_{timestamp}.log"
        )
        
        # Trade logger
        self.trade_logger = self._create_logger(
            'backtest_trades',
            f"{self.backtest_name}_trades_{timestamp}.log"
        )
        
        # State logger
        self.state_logger = self._create_logger(
            'backtest_state',
            f"{self.backtest_name}_state_{timestamp}.log"
        )
        
        # Event logger
        self.event_logger = self._create_logger(
            'backtest_events',
            f"{self.backtest_name}_events_{timestamp}.log"
        )
        
        # Performance logger
        self.performance_logger = self._create_logger(
            'backtest_performance',
            f"{self.backtest_name}_performance_{timestamp}.log"
        )
    
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a logger with file and/or console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        logger.handlers.clear()  # Clear existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler
        if self.enable_file:
            file_handler = logging.FileHandler(
                self.log_dir / filename,
                mode='w'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_backtest_start(self, config: Dict[str, Any]):
        """
        Log backtest start with configuration.
        
        Args:
            config: Backtest configuration dictionary
        """
        self.logger.info("=" * 80)
        self.logger.info("BACKTEST STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
    
    def log_backtest_end(self, summary: Dict[str, Any]):
        """
        Log backtest completion with summary.
        
        Args:
            summary: Summary statistics dictionary
        """
        self.logger.info("=" * 80)
        self.logger.info("BACKTEST COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Summary: {json.dumps(summary, indent=2, default=str)}")
    
    def log_trade(
        self,
        timestamp: datetime,
        trade_type: str,
        underlying: str,
        quantity: float,
        price: float,
        notional: float,
        transaction_cost: float,
        reason: str,
        **kwargs
    ):
        """
        Log a trade execution.
        
        Args:
            timestamp: Execution timestamp
            trade_type: Type of trade
            underlying: Underlying asset
            quantity: Quantity traded
            price: Execution price
            notional: Notional value
            transaction_cost: Cost incurred
            reason: Reason for trade
            **kwargs: Additional metadata
        """
        trade_info = {
            'timestamp': timestamp,
            'trade_type': trade_type,
            'underlying': underlying,
            'quantity': quantity,
            'price': price,
            'notional': notional,
            'transaction_cost': transaction_cost,
            'reason': reason,
            **kwargs
        }
        
        self.trade_logs.append(trade_info)
        
        direction = "BUY" if quantity > 0 else "SELL"
        self.trade_logger.info(
            f"[{timestamp}] {trade_type.upper()} | {direction} {abs(quantity):.2f} "
            f"{underlying} @ ${price:.2f} | Notional: ${notional:.2f} | "
            f"Cost: ${transaction_cost:.2f} | Reason: {reason}"
        )
    
    def log_state(
        self,
        timestamp: datetime,
        portfolio_value: float,
        pnl: float,
        delta: float,
        cash: float = 0.0,
        **kwargs
    ):
        """
        Log portfolio state snapshot.
        
        Args:
            timestamp: State timestamp
            portfolio_value: Total portfolio value
            pnl: Cumulative P&L
            delta: Portfolio delta
            cash: Cash balance
            **kwargs: Additional state information
        """
        state_info = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'delta': delta,
            'cash': cash,
            **kwargs
        }
        
        self.state_logs.append(state_info)
        
        self.state_logger.info(
            f"[{timestamp}] Value: ${portfolio_value:,.2f} | P&L: ${pnl:,.2f} | "
            f"Delta: {delta:.2f} | Cash: ${cash:,.2f}"
        )
    
    def log_event(
        self,
        timestamp: datetime,
        event_type: str,
        message: str,
        level: str = 'INFO',
        **kwargs
    ):
        """
        Log a strategy or backtest event.
        
        Args:
            timestamp: Event timestamp
            event_type: Type of event
            message: Event message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            **kwargs: Additional metadata
        """
        event_info = {
            'timestamp': timestamp,
            'event_type': event_type,
            'message': message,
            'level': level,
            **kwargs
        }
        
        self.event_logs.append(event_info)
        
        log_func = getattr(self.event_logger, level.lower())
        log_func(f"[{timestamp}] {event_type.upper()}: {message}")
    
    def log_performance(
        self,
        timestamp: datetime,
        metric_name: str,
        metric_value: float,
        **kwargs
    ):
        """
        Log a performance metric.
        
        Args:
            timestamp: Measurement timestamp
            metric_name: Name of metric
            metric_value: Value of metric
            **kwargs: Additional metadata
        """
        perf_info = {
            'timestamp': timestamp,
            'metric_name': metric_name,
            'metric_value': metric_value,
            **kwargs
        }
        
        self.performance_logs.append(perf_info)
        
        self.performance_logger.info(
            f"[{timestamp}] {metric_name}: {metric_value}"
        )
    
    def log_hedge_decision(
        self,
        timestamp: datetime,
        should_hedge: bool,
        current_delta: float,
        threshold: float,
        reason: str
    ):
        """
        Log a hedging decision.
        
        Args:
            timestamp: Decision timestamp
            should_hedge: Whether hedge will be executed
            current_delta: Current portfolio delta
            threshold: Delta threshold
            reason: Reason for decision
        """
        decision = "HEDGE" if should_hedge else "NO HEDGE"
        self.event_logger.info(
            f"[{timestamp}] {decision} | Delta: {current_delta:.2f} | "
            f"Threshold: {threshold:.2f} | Reason: {reason}"
        )
        
        self.log_event(
            timestamp=timestamp,
            event_type='hedge_decision',
            message=f"{decision}: {reason}",
            should_hedge=should_hedge,
            current_delta=current_delta,
            threshold=threshold
        )
    
    def save_structured_logs(self):
        """Save structured logs to JSON files."""
        if not self.enable_file:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades
        if self.trade_logs:
            trade_file = self.log_dir / f"{self.backtest_name}_trades_{timestamp}.json"
            with open(trade_file, 'w') as f:
                json.dump(self.trade_logs, f, indent=2, default=str)
            self.logger.info(f"Saved trade logs to {trade_file}")
        
        # Save states
        if self.state_logs:
            state_file = self.log_dir / f"{self.backtest_name}_states_{timestamp}.json"
            with open(state_file, 'w') as f:
                json.dump(self.state_logs, f, indent=2, default=str)
            self.logger.info(f"Saved state logs to {state_file}")
        
        # Save events
        if self.event_logs:
            event_file = self.log_dir / f"{self.backtest_name}_events_{timestamp}.json"
            with open(event_file, 'w') as f:
                json.dump(self.event_logs, f, indent=2, default=str)
            self.logger.info(f"Saved event logs to {event_file}")
        
        # Save performance
        if self.performance_logs:
            perf_file = self.log_dir / f"{self.backtest_name}_performance_{timestamp}.json"
            with open(perf_file, 'w') as f:
                json.dump(self.performance_logs, f, indent=2, default=str)
            self.logger.info(f"Saved performance logs to {perf_file}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged information.
        
        Returns:
            Dictionary with log statistics
        """
        return {
            'num_trades': len(self.trade_logs),
            'num_states': len(self.state_logs),
            'num_events': len(self.event_logs),
            'num_performance_metrics': len(self.performance_logs),
            'log_dir': str(self.log_dir) if self.enable_file else None
        }
    
    def __repr__(self) -> str:
        return (
            f"BacktestLogger("
            f"trades={len(self.trade_logs)}, "
            f"states={len(self.state_logs)}, "
            f"events={len(self.event_logs)})"
        )

