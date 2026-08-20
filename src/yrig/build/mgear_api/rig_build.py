from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from yrig.build.context import temp_asset_root, temp_build_scope
from yrig.build.core import resolve_build_scope
from yrig.build.mgear_api.log import (
    ProgressLogHandler,
    _capture_mgear_logs,
    _capture_mgear_output,
    _temporary_log_handler,
)
from yrig.build.mgear_api.step import BuildStep
from yrig.build.progress import bind_progress_step
from yrig.build.scope import BuildScope

mgear_api_logger = logging.getLogger("yrig.build.mgear_api")

BUILD_STEPS: list[BuildStep] = [
    BuildStep("Init", 1),
    BuildStep("Objects", 5),
    BuildStep("Properties", 1),
    BuildStep("Operators", 1),
    BuildStep("Connect", 1),
    BuildStep("Joints", 1),
    BuildStep("Finalize", 1),
]


def build_from_shifter_file(  # noqa: ANN201
    file_path: Path,
    dev_build: bool,
    progress_callback: Callable[[float, str | None], None] | None = None,
    components: bool = True,
):
    from mgear.core import curve
    from mgear.shifter import Rig, io

    # Get the guide data from the file
    guide_data: dict = io._import_guide_template(file_path)
    param_values = guide_data["guide_root"]["param_values"]

    # Set WIP mode in the mgear guide data if we're doing a dev build
    param_values["mode"] = 1 if dev_build else 0

    # Get the relevant steps of the build (progress reporting)
    pre_custom_step: dict = json.loads(param_values["preCustomStep"])
    post_custom_step: dict = json.loads(param_values["postCustomStep"])
    pre_custom_steps: list[BuildStep] = [
        BuildStep(item["path"]) for item in pre_custom_step["items"]
    ]
    post_custom_steps: list[BuildStep] = [
        BuildStep(item["path"]) for item in post_custom_step["items"]
    ]
    num_components = len(guide_data["components_list"])

    rig = Rig()
    progress_handler = ProgressLogHandler(
        pre_steps=pre_custom_steps,
        build_steps=BUILD_STEPS,
        post_steps=post_custom_steps,
        number_of_components=num_components,
        components=components,
        progress_callback=progress_callback,
    )
    with (
        _capture_mgear_output(mgear_api_logger),
        _capture_mgear_logs(mgear_api_logger),
        _temporary_log_handler(mgear_api_logger, progress_handler),
        bind_progress_step(progress_handler.root_step),
    ):
        mgear_api_logger.info("\n" + "= SHIFTER RIG SYSTEM " + "=" * 46)

        rig.stopBuild = False

        rig.guide.set_from_dict(guide_data)

        # Build
        mgear_api_logger.info("\n" + "= BUILDING RIG " + "=" * 46)
        # Get merged options early so custom steps use blueprint values
        merged_options = rig.guide.getMergedOptions()
        rig.from_dict_custom_step(merged_options, pre=True)

        # Just build a barebones rig with root if we're doing a custom step only build
        if not components:
            rig.options = rig.guide.getMergedOptions()
            rig.guides = rig.guide.components
            rig.customStepDic["mgearRun"] = rig
            rig.initialHierarchy()
            rig.addToGroup("jnt_org", "deformers")
            rig.finalize()
        else:
            rig.build()

        # Check if build was cancelled
        if rig.stopBuild:
            mgear_api_logger.info("\n" + "= SHIFTER BUILD CANCELLED " + "=" * 40)
            return False

        rig.from_dict_custom_step(merged_options, pre=False)

        # Check if build was cancelled/failed during custom steps
        if rig.stopBuild:
            mgear_api_logger.info("\n" + "= SHIFTER BUILD CANCELLED " + "=" * 40)
            return False

        # controls shapes buffer
        if guide_data["ctl_buffers_dict"]:
            curve.update_curve_from_data(
                guide_data["ctl_buffers_dict"], rplStr=["_controlBuffer", ""]
            )
    return True


def build_from_path(
    rig_root_path: Path,
    dev_build: bool = False,
    build_scope: BuildScope | str | None = None,
    progress_callback: Callable[[float, str | None], None] | None = None,
) -> bool:
    """Build an mGear Shifter rig from a rig path.

    Args:
        rig_root_path: Path to an a rig file structure.
        dev_build: When true the mGear shifter build will be set to WIP mode.
        progress_callback: A function to call at each step of the build.
            It will be called with a float (overall progress from 0-1) and a string (the current step)
        build_scope: An optional BuildScope which will limit the build to that scope.


    Returns:
        bool: True if the build was successful, else False
    """

    guide_path = rig_root_path / "data/guide.sgt"
    resolved_scope = resolve_build_scope(build_scope)
    with temp_asset_root(rig_root_path, dev_build), temp_build_scope(resolved_scope, dev_build):
        mgear_api_logger.info("Starting mGear Shifter build from file: %s", guide_path)
        try:
            components = resolved_scope != BuildScope.FACE
            build_result = build_from_shifter_file(
                guide_path,
                dev_build,
                progress_callback=progress_callback,
                components=components,
            )
        except Exception as e:
            mgear_api_logger.error("mGear build failed: %s", e)
            raise RuntimeError(f"mGear Shifter build failed for '{guide_path.name}': {e}") from e
            return False

        if build_result is not None:
            mgear_api_logger.info("Build from file complete.")
            return True
        else:
            mgear_api_logger.info("Build from file cancelled/failed.")
            return False
