"""
Pareto front utilities for multi-objective tracking (error vs. complexity).

This module maintains a compact non-dominated set of candidates and supports
lightweight CSV logging. It is optional and can be enabled from evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Iterable
import csv
import os


@dataclass
class ParetoItem:
    error: float           # e.g., 1 - R^2 (clipped to [0, 2])
    complexity: float      # expression.complexity()
    expression_str: str    # string form for persistence
    generation: int        # when it was recorded


class ParetoFront:
    """Maintain a non-dominated set of (error, complexity) items.

    Notes:
        - Minimizes both error and complexity.
        - Keeps a bounded capacity by removing weakly dominated points
          biased toward diversity along the front.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._items: List[ParetoItem] = []

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[ParetoItem]:
        return list(self._items)

    @staticmethod
    def _dominates(a: ParetoItem, b: ParetoItem) -> bool:
        return (a.error <= b.error and a.complexity <= b.complexity) and (
            a.error < b.error or a.complexity < b.complexity
        )

    def add(self, item: ParetoItem) -> bool:
        """Add an item; return True if it was inserted (i.e., not dominated)."""
        # Discard if dominated by any existing point
        for p in self._items:
            if self._dominates(p, item):
                return False

        # Remove points dominated by the new item
        self._items = [p for p in self._items if not self._dominates(item, p)]

        # Insert
        self._items.append(item)

        # Enforce capacity with a simple thinning strategy along error
        if len(self._items) > self.capacity:
            # Sort by error then complexity; keep evenly spaced samples
            self._items.sort(key=lambda x: (x.error, x.complexity))
            step = max(1, len(self._items) // self.capacity)
            self._items = self._items[::step][: self.capacity]
        return True

    def extend(self, items: Iterable[ParetoItem]) -> int:
        inserted = 0
        for it in items:
            if self.add(it):
                inserted += 1
        return inserted

    def to_csv(self, path: str):
        """Append current front to CSV (idempotent header)."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["generation", "error", "complexity", "expression"])
            for it in self._items:
                w.writerow([it.generation, f"{it.error:.8f}", f"{it.complexity:.6f}", it.expression_str])
