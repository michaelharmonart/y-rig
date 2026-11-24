from dataclasses import dataclass
from pathlib import Path

from .models import AnimationSettings, FrameRange
from .naming import (
    resolve_animation_name,
    resolve_base_name,
    validate_name,
)
from .storage import save_settings


@dataclass
class ExportJob:
    name: str
    path: Path
    frame_range: FrameRange
    namespace: str
    export_model: bool


def create_export_jobs(
    settings: AnimationSettings,
    shot_name: str,
    export_directory: Path,
) -> list[ExportJob]:

    base_name = resolve_base_name(
        shot_name=shot_name,
        settings=settings,
    )

    jobs: list[ExportJob] = []

    if not settings.namespace:
        raise ValueError("No rig namespace selected.")

    def add_job(
        name: str,
        frame_range: FrameRange,
    ) -> None:

        frame_range.validate()

        jobs.append(
            ExportJob(
                name=name,
                path=export_directory / f"{name}.fbx",
                frame_range=frame_range,
                namespace=settings.namespace,
                export_model=settings.export_model,
            )
        )

    if settings.export_full:
        add_job(
            name=base_name,
            frame_range=settings.frame_range,
        )

    for sequence in settings.sequences:
        if not sequence.enabled:
            continue

        suffix = validate_name(sequence.suffix)

        name = resolve_animation_name(
            base_name=base_name,
            suffix=suffix,
        )

        add_job(
            name=name,
            frame_range=sequence.frame_range,
        )

    if not jobs:
        raise ValueError("No animations selected for export.")

    # Case-insensitive to avoid collisions across platforms.
    names = [job.name.casefold() for job in jobs]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate animation export names.")

    return jobs


def export_animation(job: ExportJob) -> None:
    """Exports a single animation using a clean skeleton."""

    from yrig.ue.anim_export import (
        publish_rig_animation,
        resolve_rig_name,
    )

    from .context import get_available_rigs

    rig = resolve_rig_name(
        namespace=job.namespace,
        rigs=get_available_rigs(),
    )

    job.path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    publish_rig_animation(
        path=job.path,
        rig=rig,
        namespace=job.namespace,
        frame_range=job.frame_range.as_tuple(),
        export_model=job.export_model,
        export_start=0,
    )


def publish_animations(
    settings: AnimationSettings,
    shot_name: str,
    export_directory: Path,
    overwrite: bool = False,
) -> list[Path]:

    jobs = create_export_jobs(
        settings=settings,
        shot_name=shot_name,
        export_directory=export_directory,
    )

    if not overwrite:
        existing = [str(job.path) for job in jobs if job.path.exists()]

        if existing:
            raise FileExistsError("The following animations already exist:\n" + "\n".join(existing))

    exported_paths = []

    for job in jobs:
        export_animation(job)
        exported_paths.append(job.path)

    save_settings(settings)

    return exported_paths
