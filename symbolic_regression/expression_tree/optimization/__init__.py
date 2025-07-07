"""Optimization utilities for expression trees."""

from .memory_pool import NodePool, get_global_pool, clear_global_pool, reset_global_pool

__all__ = ['NodePool', 'get_global_pool', 'clear_global_pool', 'reset_global_pool']
