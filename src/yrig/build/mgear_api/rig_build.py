from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from yrig.build.mgear_api.log import (
    ProgressLogHandler,
    _capture_mgear_logs,
    _capture_mgear_output,
    _temporary_log_handler,
)
from yrig.build.mgear_api.step import BuildStep
from yrig.build.progress import bind_progress_step

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
