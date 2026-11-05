from __future__ import annotations

from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_current_progress: ContextVar[ProgressStep | None] = ContextVar("_current_progress", default=None)


@contextmanager
def progress_step(
    name: str, weight: float = 1.0, total: float | None = None
) -> Generator[ProgressStep, None, None]:
    """Set ``total`` to the expected sum of child weights for incremental progress."""
    with report_progress(ProgressStep(name, weight, total)) as step:
        yield step


@contextmanager
def _maya_progress_context(
    name: str,
) -> Generator[Callable[[float, str], None] | None, None, None]:
    try:
        from maya import cmds, mel
    except ImportError:
        yield None
        return

    if cmds.about(batch=True):
        yield None
        return

    bar = mel.eval("$tmp = $gMainProgressBar")
    if cmds.progressBar(bar, query=True, visible=True):
        yield None
        return

    cmds.progressBar(
        bar,
        edit=True,
        beginProgress=True,
        status=name,
        progress=0,
        minValue=0,
        maxValue=1000,
        isInterruptable=False,
    )

    def update(value: float, status: str) -> None:
        cmds.progressBar(bar, edit=True, progress=round(value * 1000), status=status)

    try:
        yield update
    finally:
        cmds.progressBar(bar, edit=True, endProgress=True)


@contextmanager
def report_progress(step: ProgressStep) -> Generator[ProgressStep, None, None]:
    """Bind a step, resporting progress to the UI when no parent or callback handles reporting."""
    callback = step._callback
    ui_context = (
        _maya_progress_context(step.name)
        if _current_progress.get() is None and callback is None
        else nullcontext(callback)
    )
    with ui_context as ui_callback:
        step._callback = ui_callback
        try:
            with bind_progress_step(step):
                yield step
        finally:
            step._callback = callback


@contextmanager
def bind_progress_step(step: ProgressStep) -> Generator[ProgressStep, None, None]:
    """Attach a step to the current progress scope and finish it on exit."""
    parent = _current_progress.get()
    if parent:
        parent.add_child_step(step)
    token = _current_progress.set(step)
    try:
        step._propogate_progress()
        yield step
    finally:
        try:
            step.finish_step()
        finally:
            _current_progress.reset(token)


def get_current_progress_step() -> ProgressStep | None:
    return _current_progress.get()


def progress_update(value: float) -> None:
    step = _current_progress.get()
    if step:
        step.update_progress(value)


def finish_step() -> None:
    step = _current_progress.get()
    if step:
        step.finish_step()


class ProgressStep:
    def __init__(
        self,
        name: str,
        weight: float = 1,
        total_weight: float | None = None,
        callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.name = name
        self._weight = weight
        self._progress: float = 0.0
        self._children: list[ProgressStep] = []
        self._parent: ProgressStep | None = None
        self._total_weight: float | None = total_weight
        self._child_weight_sum: float = 0
        self._finished: bool = False
        self._callback = callback

    def get_progress(self) -> float:
        return self._progress

    def add_child_step(self, step: ProgressStep) -> None:
        step._parent = self
        self._children.append(step)
        self._child_weight_sum += step._weight

    def get_child_steps(self) -> list[ProgressStep]:
        return self._children

    def _update_progress_from_children(self) -> None:
        if not self._children:
            return
        denominator = self._total_weight or self._child_weight_sum
        cumulative_progress = 0
        for child in self._children:
            child_progress = child.get_progress()
            scaled_progress = child_progress * (child._weight / denominator)
            cumulative_progress += scaled_progress
        self._progress = cumulative_progress
        self._propogate_progress()

    def _propogate_progress(self) -> None:
        if self._parent is not None:
            self._parent._update_progress_from_children()
        elif self._callback is not None:
            current_step = get_current_progress_step() or self
            name_path = "/".join(s.name for s in current_step.get_ancestors())
            with suppress(Exception):
                self._callback(self._progress, name_path)

    def update_progress(self, progress: float) -> None:
        if self._children:
            return
        if progress == 1:
            self.finish_step()
            return
        self._progress = max(0.0, min(1.0, progress))
        self._propogate_progress()

    def _set_finished(self) -> None:
        self._finished = True
        self._progress = 1

    def finish_step(self) -> None:
        if self._finished:
            return
        for child in self._children:
            child.finish_step()
        self._set_finished()
        self._propogate_progress()

    def get_ancestors(self) -> list[ProgressStep]:
        node: ProgressStep | None = self
        out: list[ProgressStep] = []
        while node is not None:
            out.append(node)
            node = node._parent
        return list(reversed(out))
