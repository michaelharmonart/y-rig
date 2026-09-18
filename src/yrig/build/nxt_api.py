from __future__ import annotations

import logging
import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nxt.nxt_layer import CompLayer
from nxt.nxt_node import get_node_enabled
from nxt.runtime import ExitGraph, ExitNode, InvalidNodeError
from nxt.session import Session
from nxt.stage import Stage, logger, run

if TYPE_CHECKING:
    from nxt.nxt_layer import SpecLayer
    from nxt.nxt_node import SpecNode


from yrig.build.progress import progress_step

log = logging.getLogger(__name__)

YRIG_NXT_DIR = (  # Get the path (resolve symlinks first though)
    Path(__file__).resolve().parents[3] / "nxt"
).resolve()
os.environ["YRIG_NXT_DIR"] = str(YRIG_NXT_DIR)

nxt_file_roots = os.environ.get("NXT_FILE_ROOTS", "").split(os.pathsep)
if str(YRIG_NXT_DIR) not in nxt_file_roots:
    nxt_file_roots.append(str(YRIG_NXT_DIR))
    os.environ["NXT_FILE_ROOTS"] = os.pathsep.join(nxt_file_roots)


@contextmanager
def nxt_file_roots(
    file_roots: Iterable[Path], restore: bool = False
) -> Generator[None, None, None]:
    """Temporarily set the NXT_FILE_ROOTS env var, restoring it afterward if restore is True."""
    default_value = os.environ.get("NXT_FILE_ROOTS")
    os.environ["NXT_FILE_ROOTS"] = os.pathsep.join(map(str, file_roots))
    try:
        yield
    finally:
        if restore:
            if default_value is None:
                os.environ.pop("NXT_FILE_ROOTS")
            else:
                os.environ["NXT_FILE_ROOTS"] = default_value


# We wrap the NXT execution so we can have nice progress reporting :)
class ProgressStage(Stage):
    def execute_nodes(self, node_paths, layer, parameters=None):  # noqa: ANN001, ANN201
        """Execute nodes at given `node_paths` using given `layer`. Returns
        runtime layer object that if passed as layer argument to successive
        calls will "continue" execution with the same cached values.
        If parameters are provided they will be applied before the layer node
        runs, unless the layer provided (in the layer arg) is a runtime layer,
        in which case they will be applied before the first node is run.
        :param node_paths: node paths to execute
        :type node_paths: list
        :param layer: CompLayer to execute
        :type layer: CompLayer
        :param parameters: Optional dict of {'/node.attr': value} to be
        applied before execution begins.
        :type parameters: dict
        :raises ValueError: When layer argument has invalid value;
        GraphError: For any exception raised by a node's compute.
        :return: Runtime CompLayer that can be used for continued execution.
        :rtype: CompLayer
        """
        if not isinstance(layer, CompLayer):
            raise ValueError("Execute Nodes requires a comp layer.")  # noqa
        if not layer.runtime:
            dup_comp = self.build_stage(layer.layer_idx())
            runtime_layer: CompLayer = self.setup_runtime_layer(dup_comp, parameters=parameters)
        else:
            runtime_layer: CompLayer = layer
            if parameters:
                self.set_runtime_parameters(parameters, runtime_layer)

        with progress_step("NXT", total=len(node_paths)):
            for path in node_paths:
                curr_node = runtime_layer.lookup(path)
                if get_node_enabled(curr_node) is False:
                    continue
                if not curr_node:
                    raise InvalidNodeError(path)

                logger.execinfo("Executing: " + path, links=[path])  # type: ignore
                runtime_layer.cache_layer.set_node_enter_time(path)
                try:
                    with progress_step(path):
                        run(runtime_layer, stage=self, rt_node=curr_node)
                except ExitNode as exit_node:
                    logger.debug(f"Exited Node {path}: {exit_node}", links=[path])  # type: ignore
                    continue
                except ExitGraph as exit_graph:
                    exit_graph.runtime_layer = runtime_layer
                    logger.execinfo(f"Exited Graph {layer.real_path}: {exit_graph}")  # type: ignore
                    raise
                finally:
                    runtime_layer.cache_layer.set_node_exit_time(path)
                    t = str(round(runtime_layer.cache_layer.get_node_run_time(path)))
                    msg = "Time to execute {}: {} second(s)."
                    logger.execinfo(msg.format(path, t), links=[path])  # type: ignore
        return runtime_layer


def execute_nxt_graph(filepath: Path, parameters: dict[str, Any] | None = None) -> None:
    stage: ProgressStage = ProgressStage.load_from_filepath(str(filepath))
    stage.execute(parameters=parameters)


def _add_world_node(stage: Stage, layer: SpecLayer) -> SpecNode:
    world_node, _ = stage.add_node(
        name="",
        parent="/",
        layer=layer,
        fix_names=False,
    )
    return world_node[0]


def setup_rig_build_nxt_layer(
    filepath: Path, *, rig_path: Path, inherits: Path | None, name: str, color: str
) -> None:
    """
    Create and save an NXT layer for a rig build.

    The layer is given the specified name and color, optionally inherits from
    another layer, and stores the rig path on its world node.

    Args:
        filepath: Path where the NXT layer will be saved.
        rig_path: Path to the rig associated with the layer.
        inherits: Optional path to a layer this layer should inherit from.
        name: Alias to assign to the new layer.
        color: Color to assign to the layer.

    Returns:
        The NXT session containing the newly created layer.
    """
    filepath = filepath.resolve()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    session = Session()
    stage: Stage = session.new_file()
    layer: SpecLayer = stage.top_layer

    layer.set_alias(name)
    layer.color = color

    if inherits is not None:
        layer.add_reference(layer_path=inherits.as_posix())

    world_node = _add_world_node(stage, layer)

    stage.add_node_attr(
        node=world_node,
        attr="rig_path",
        attr_data={"value": rig_path.as_posix()},
        layer=layer,
    )
    layer.save(filepath=filepath.as_posix())
