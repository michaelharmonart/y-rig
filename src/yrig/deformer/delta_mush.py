from maya import cmds


def create_delta_mush(
    geometry: str,
    name: str | None = None,
    *,
    smoothing_iterations: int = 10,
    smoothing_step: float = 0.5,
) -> str:
    resolved_name = name or f"{geometry}_deltaMush"
    delta_mush: str = cmds.deltaMush(  # type: ignore
        geometry,
        name=resolved_name,
        smoothingIterations=smoothing_iterations,
        smoothingStep=smoothing_step,
    )[0]
    return delta_mush
