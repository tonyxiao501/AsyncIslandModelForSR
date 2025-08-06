"""
Advanced Logging System for Symbolic Regression

This module provides a centralized logging system with different verbosity levels
to reduce terminal clutter while maintaining important information.
"""

import logging
import sys
from typing import Optional, Dict, Any
from enum import Enum
import time
from datetime import datetime


class LogLevel(Enum):
    """Enumeration of logging levels for symbolic regression"""
    SILENT = 0      # No output except critical errors
    MINIMAL = 1     # Only final results and critical info
    MODERATE = 2    # Progress updates and key milestones
    DETAILED = 3    # Detailed evolution progress
    VERBOSE = 4     # All information including debug details


class SymbolicRegressionLogger:
    """
    Centralized logger for symbolic regression with context-aware formatting
    """
    
    def __init__(self, log_level: LogLevel = LogLevel.MODERATE, 
                 log_to_file: bool = False, log_file_path: Optional[str] = None):
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.start_time = time.time()
        self.last_progress_time = time.time()
        
        # Create logger
        self.logger = logging.getLogger('symbolic_regression')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        if self.log_level != LogLevel.SILENT:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_to_file:
            if log_file_path is None:
                log_file_path = f"symbolic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _should_log(self, required_level: LogLevel) -> bool:
        """Check if message should be logged based on current log level"""
        return self.log_level.value >= required_level.value
    
    def critical(self, message: str, **kwargs):
        """Always logged - critical errors and failures"""
        if self.log_level != LogLevel.SILENT:
            self.logger.error(f"CRITICAL: {message}")
    
    def info(self, message: str, required_level: LogLevel = LogLevel.MINIMAL):
        """General information with configurable level"""
        if self._should_log(required_level):
            self.logger.info(message)
    
    def progress(self, message: str, force: bool = False):
        """Progress updates - throttled to avoid spam"""
        if not self._should_log(LogLevel.MODERATE):
            return
            
        current_time = time.time()
        # Throttle progress messages to every 2 seconds unless forced
        if force or (current_time - self.last_progress_time) >= 2.0:
            self.logger.info(f"PROGRESS: {message}")
            self.last_progress_time = current_time
    
    def evolution_step(self, generation: int, best_fitness: float, 
                      avg_fitness: float, diversity: float, additional_info: str = ""):
        """Log evolution step with detailed metrics"""
        if not self._should_log(LogLevel.DETAILED):
            return
            
        elapsed = time.time() - self.start_time
        message = (f"Gen {generation:3d}: Best={best_fitness:.6f} "
                  f"Avg={avg_fitness:.6f} Div={diversity:.3f} "
                  f"({elapsed:.1f}s)")
        if additional_info:
            message += f" {additional_info}"
        
        self.logger.info(message)
    
    def milestone(self, message: str):
        """Important milestones - always shown except in silent mode"""
        if self.log_level != LogLevel.SILENT:
            self.logger.info(f"MILESTONE: {message}")
    
    def warning(self, message: str):
        """Warnings - shown from minimal level onwards"""
        if self._should_log(LogLevel.MINIMAL):
            self.logger.warning(message)
    
    def debug(self, message: str):
        """Debug information - only in verbose mode"""
        if self._should_log(LogLevel.VERBOSE):
            self.logger.debug(f"DEBUG: {message}")
    
    def result_summary(self, results: Dict[str, Any]):
        """Log final results summary"""
        if self.log_level == LogLevel.SILENT:
            return
            
        self.logger.info("=" * 60)
        self.logger.info("SYMBOLIC REGRESSION RESULTS:")
        self.logger.info("=" * 60)
        
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"{key:.<30} {value:.6f}")
            else:
                self.logger.info(f"{key:.<30} {value}")
    
    def ensemble_summary(self, n_fits: int, n_successful: int, 
                        top_expressions: list, execution_time: float):
        """Log ensemble fitting summary"""
        if self.log_level == LogLevel.SILENT:
            return
            
        self.logger.info(f"Ensemble fitting completed in {execution_time:.1f}s")
        self.logger.info(f"Successfully fitted {n_successful}/{n_fits} regressors")
        
        if self._should_log(LogLevel.MODERATE):
            self.logger.info(f"Top {len(top_expressions)} expressions selected:")
            for i, expr_info in enumerate(top_expressions[:5]):  # Show top 5
                fitness = expr_info.get('fitness', 'N/A')
                complexity = expr_info.get('complexity', 'N/A')
                self.logger.info(f"  {i+1}. Fitness: {fitness:.6f}, Complexity: {complexity:.3f}")


# Global logger instance
_global_logger: Optional[SymbolicRegressionLogger] = None


def get_logger() -> SymbolicRegressionLogger:
    """Get or create the global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SymbolicRegressionLogger()
    return _global_logger


def set_log_level(level: LogLevel):
    """Set the global logging level"""
    global _global_logger
    if _global_logger is None:
        _global_logger = SymbolicRegressionLogger(log_level=level)
    else:
        _global_logger.log_level = level


def configure_logging(log_level: LogLevel = LogLevel.MODERATE,
                     log_to_file: bool = False,
                     log_file_path: Optional[str] = None) -> SymbolicRegressionLogger:
    """Configure the global logging system"""
    global _global_logger
    _global_logger = SymbolicRegressionLogger(
        log_level=log_level,
        log_to_file=log_to_file,
        log_file_path=log_file_path
    )
    return _global_logger


# Convenience functions for common operations
def log_info(message: str, level: LogLevel = LogLevel.MINIMAL):
    """Log info message at specified level"""
    get_logger().info(message, level)


def log_progress(message: str, force: bool = False):
    """Log progress message"""
    get_logger().progress(message, force)


def log_milestone(message: str):
    """Log milestone message"""
    get_logger().milestone(message)


def log_warning(message: str):
    """Log warning message"""
    get_logger().warning(message)


def log_evolution_step(generation: int, best_fitness: float, 
                      avg_fitness: float, diversity: float, additional_info: str = ""):
    """Log evolution step"""
    get_logger().evolution_step(generation, best_fitness, avg_fitness, diversity, additional_info)


def log_debug(message: str):
    """Log debug message"""
    get_logger().debug(message)
