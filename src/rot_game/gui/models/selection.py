"""Selection state shared across canvas, inspector, and lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Set

from PySide6.QtCore import QObject, Signal


@dataclass(frozen=True)
class SelectionSnapshot:
    """Immutable view of the current selection."""

    primary: Optional[int]
    selected: FrozenSet[int]
    hover: Optional[int]


class SelectionState(QObject):
    """Centralizes canvas selection so every component reads consistent state."""

    changed = Signal(object)  # Emits SelectionSnapshot
    hover_changed = Signal(object)  # Emits Optional[int]

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._primary: Optional[int] = None
        self._selected: Set[int] = set()
        self._hover: Optional[int] = None

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def primary(self) -> Optional[int]:
        return self._primary

    @property
    def selected(self) -> FrozenSet[int]:
        return frozenset(self._selected)

    @property
    def hover(self) -> Optional[int]:
        return self._hover

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def clear(self) -> None:
        if not self._selected and self._primary is None and self._hover is None:
            return
        self._selected.clear()
        self._primary = None
        self._hover = None
        self._emit_changed()
        self.hover_changed.emit(None)

    def select(self, index: int, additive: bool = False) -> None:
        """Select an index, optionally adding to the existing set."""
        if index < 0:
            self.clear()
            return

        changed = False
        if additive:
            if index not in self._selected:
                self._selected.add(index)
                changed = True
        else:
            if self._selected != {index}:
                self._selected = {index}
                changed = True

        if self._primary != index:
            self._primary = index
            changed = True

        if changed:
            self._emit_changed()

    def toggle(self, index: int) -> None:
        """Toggle membership of an index; pick a new primary if necessary."""
        if index < 0:
            return

        changed = False
        if index in self._selected:
            self._selected.remove(index)
            changed = True
            if self._primary == index:
                self._primary = next(iter(self._selected), None)
        else:
            self._selected.add(index)
            changed = True
            self._primary = index

        if changed:
            self._emit_changed()

    def set_primary(self, index: Optional[int]) -> None:
        """Force a primary cell (must already be selected if not None)."""
        if index is None:
            if self._primary is None:
                return
            self._primary = None
            self._emit_changed()
            return

        if index not in self._selected:
            self._selected = {index}
            changed = True
        else:
            changed = False
        if self._primary != index:
            self._primary = index
            changed = True
        if changed:
            self._emit_changed()

    def set_hover(self, index: Optional[int]) -> None:
        if self._hover == index:
            return
        self._hover = index
        self.hover_changed.emit(index)

    def replace(self, selected: Iterable[int], primary: Optional[int]) -> None:
        """Replace selection with ``selected`` set and primary index."""
        new_selected = {idx for idx in selected if idx >= 0}
        if primary is not None and primary not in new_selected:
            primary = next(iter(sorted(new_selected)), None)
        if not new_selected:
            primary = None
        if self._selected == new_selected and self._primary == primary:
            return
        self._selected = new_selected
        self._primary = primary
        self._emit_changed()

    # ------------------------------------------------------------------
    def snapshot(self) -> SelectionSnapshot:
        return SelectionSnapshot(
            primary=self._primary,
            selected=frozenset(self._selected),
            hover=self._hover,
        )

    def _emit_changed(self) -> None:
        self.changed.emit(self.snapshot())
