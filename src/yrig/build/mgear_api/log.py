import logging
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import Token
from io import TextIOBase

from yrig.build.mgear_api.step import BuildStep
from yrig.build.progress import ProgressStep, _current_progress


class StdoutToLogger(TextIOBase):
    """A write-only stream that forwards each line to a Python logger.

    mGear calls ``sys.stdout.write()`` directly rather than using the
    ``logging`` module, so we temporarily replace ``sys.stdout`` with one
    of these to capture its output.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO) -> None:
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
            self._buffer = ""


@contextmanager
def _redirect_external_logger(
    external_logger: logging.Logger, target_logger: logging.Logger
) -> Generator[logging.Logger, None, None]:
    """Temporarily hooks an external logger into a specified target."""

    # Store original state
    original_parent = external_logger.parent
    original_propagate = external_logger.propagate

    try:
        external_logger.parent = target_logger
        external_logger.propagate = True
        yield external_logger
    finally:
        # Restore original state exactly as it was
        external_logger.parent = original_parent
        external_logger.propagate = original_propagate


@contextmanager
def _capture_mgear_logs(target_logger: logging.Logger) -> Generator[None, None, None]:
    import mgear.pymaya

    def display_info(msg: str) -> None:
        target_logger.info(msg)

    def display_warning(msg: str) -> None:
        target_logger.warning(msg)

    def display_error(msg: str) -> None:
        target_logger.error(msg)

    original_info = mgear.pymaya.displayInfo
    original_warning = mgear.pymaya.displayWarning
    original_error = mgear.pymaya.displayError

    try:
        mgear.pymaya.displayInfo = display_info  # type: ignore
        mgear.pymaya.displayWarning = original_warning
        mgear.pymaya.displayError = display_error  # type: ignore
        yield

    finally:
        mgear.pymaya.displayInfo = original_info
        mgear.pymaya.displayWarning = original_warning
        mgear.pymaya.displayError = original_error


@contextmanager
def _capture_mgear_output(target_logger: logging.Logger) -> Generator[None, None, None]:
    """Redirect sys.stdout and sys.stderr into this module's logger."""

    stdout_logger = StdoutToLogger(target_logger)
    stderr_logger = StdoutToLogger(target_logger, logging.ERROR)

    with (
        redirect_stdout(stdout_logger),
        redirect_stderr(stderr_logger),
    ):
        yield


@contextmanager
def _temporary_log_handler(logger: logging.Logger, handler: logging.Handler) -> Iterator[None]:
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


class BuildStepFilter(logging.Filter):
    def __init__(self, build_steps: Sequence[BuildStep]) -> None:
        super().__init__()
        self._build_prefix_set: set[str] = {f"{step.name} : " for step in build_steps}
        self._custom_step_prefix_set: set[str] = {
            "EXEC: Executing custom step: ",
            "SUCCEED: Custom Shifter Step Class: ",
        }
        self._valid_prefix_set = self._build_prefix_set | self._custom_step_prefix_set

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()

        return any(message.startswith(prefix) for prefix in self._valid_prefix_set)


class ProgressLogHandler(logging.Handler):
    def __init__(
        self,
        pre_steps: Sequence[BuildStep],
        build_steps: Sequence[BuildStep],
        post_steps: Sequence[BuildStep],
        number_of_components: int,
        components: bool = True,
        progress_callback: Callable[[float, str | None], None] | None = None,
    ) -> None:
        super().__init__()
        self.progress_callback = progress_callback
        self.active_steps: dict[str, ProgressStep] = {}
        self.active_step_tokens: dict[str, Token] = {}
        self.pre_steps = pre_steps
        self.build_steps = build_steps
        self.post_steps = post_steps
        self.addFilter(BuildStepFilter(build_steps))
        self.number_of_components = number_of_components

        self.pre_step_names: list[str] = [step.name for step in self.pre_steps]
        self.post_step_names: list[str] = [step.name for step in self.post_steps]

        self.root_step = ProgressStep("Rig Build", callback=progress_callback)
        self.pre_build = ProgressStep("Pre Build", len(pre_steps) * 0.1)
        self.pre_step_map: dict[str, ProgressStep] = {}
        self.root_step.add_child_step(self.pre_build)
        for step in self.pre_steps:
            step_progress = ProgressStep(step.name, weight=step.weight)
            self.pre_build.add_child_step(step_progress)
            self.pre_step_map[step.name] = step_progress
        self.pre_step_finished: bool = False

        # Main Build Steps
        self.build_step = ProgressStep("Main Build", 10)
        self.build_step_map: dict[str, ProgressStep] = {}
        if components:
            self.root_step.add_child_step(self.build_step)
        for step in self.build_steps:
            step_progress = ProgressStep(step.name, weight=step.weight)
            self.build_step.add_child_step(step_progress)
            self.build_step_map[step.name] = step_progress
            # Add leaf steps for components
            for i in range(number_of_components):
                step_progress.add_child_step(ProgressStep(f"{step.name}_component{i}", weight=1))
        self.build_step_finished: bool = False

        self.post_build = ProgressStep("Post Build", len(post_steps))
        self.post_step_map: dict[str, ProgressStep] = {}
        self.root_step.add_child_step(self.post_build)
        for step in self.post_steps:
            step_progress = ProgressStep(step.name, weight=step.weight)
            self.post_build.add_child_step(step_progress)
            self.post_step_map[step.name] = step_progress
        self.post_step_finished: bool = False

        self.build_step_counter: dict[str, int] = {}

    def _start_step(self, name: str) -> ProgressStep | None:
        step = None
        if name in self.pre_step_names:
            step = self.pre_step_map[name]
        if name in self.post_step_names:
            step = self.post_step_map[name]
        if step is not None:
            self.active_steps[name] = step
            token = _current_progress.set(step)
            self.active_step_tokens[name] = token
        return step

    def _finish_step(self, name: str) -> None:
        step = self.active_steps.pop(name, None)
        token = self.active_step_tokens.pop(name, None)
        if step is not None:
            step.finish_step()
        if token is not None:
            _current_progress.reset(token)

    def _update_step_progress(self, name: str, value: float) -> None:
        step = self.active_steps.get(name)
        if step is not None:
            step.update_progress(value)

    def _on_build_step_progress(self, step_name: str) -> None:
        if not self.pre_step_finished:
            self.pre_build.finish_step()
            self.pre_step_finished = True
        if step_name in self.build_step_counter:
            self.build_step_counter[step_name] += 1
        else:
            self.build_step_counter[step_name] = 1
        step = self.build_step_map.get(step_name)
        if step is not None:
            children = step.get_child_steps()
            step_child_index = self.build_step_counter[step_name] - 1
            if step_child_index < len(children):
                children[step_child_index].finish_step()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()

        if message.startswith("EXEC: Executing custom step: "):
            path = message.split(": ")[2]
            step_name = path
            self._start_step(step_name)

        if message.startswith("SUCCEED: Custom Shifter Step Class: "):
            path = message.split(": ")[2].rsplit(".", 1)[0]
            step_name = path
            self._finish_step(step_name)

        prefix = message.partition(" : ")[0]
        step_name = prefix
        self._on_build_step_progress(step_name)
