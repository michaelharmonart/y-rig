from typing import Any

from pxr import Sdf


def spec_fields(spec: Sdf.Spec) -> dict[str, Any]:
    return {key: spec.GetInfo(key) for key in spec.ListInfoKeys()}


def changed_fields(before: Sdf.Spec | None, after: Sdf.Spec) -> dict[str, Any]:
    return {
        key: after.GetInfo(key)
        for key in after.ListInfoKeys()
        if before is None or not before.HasInfo(key) or before.GetInfo(key) != after.GetInfo(key)
    }


def generate_delta_layer(before: Sdf.Layer, after: Sdf.Layer) -> Sdf.Layer:
    """
    Existing prims receive sparse overrides; changed properties are copied in
    full, including unchanged metadata and values. New prim subtrees retain
    their definitions. Inputs must be flattened layers, without composition
    arcs or variants (use ``stage.Flatten()``). Prims that are in `before` but not `after` are deactivated
    at their subtree roots.
    """
    delta = Sdf.Layer.CreateAnonymous("delta.usd")

    def compare(old: Sdf.Spec, new: Sdf.Spec) -> dict[str, Any]:
        if type(old) is not type(new):
            raise ValueError(f"Cannot override spec kind change at {new.path}")
        return changed_fields(old, new)

    def copy_spec(spec: Sdf.Spec) -> None:
        parent_path = spec.path.GetParentPath()
        if parent_path != Sdf.Path.absoluteRootPath:
            Sdf.CreatePrimInLayer(delta, parent_path)
        if not Sdf.CopySpec(after, spec.path, delta, spec.path):
            raise RuntimeError(f"Could not copy {spec.path}")

    def walk(old: Sdf.PrimSpec, new: Sdf.PrimSpec) -> None:
        # Deal with the metadata on the prim
        changed_fields = compare(old, new)
        if changed_fields:
            destination = (
                delta.pseudoRoot
                if new.path == Sdf.Path.absoluteRootPath
                else Sdf.CreatePrimInLayer(delta, new.path)
            )
            for key, value in changed_fields.items():
                destination.SetInfo(key, value)

        # Deal with properties on the prim
        for name, prop in new.properties.items():
            previous = old.properties.get(name)
            if previous is None or compare(previous, prop):
                copy_spec(prop)

        # Deactivate child prims that don't exist in new
        for name, child in old.nameChildren.items():
            if name not in new.nameChildren:
                Sdf.CreatePrimInLayer(delta, child.path).active = False

        # Walk the child prims.
        for name, child in new.nameChildren.items():
            previous = old.nameChildren.get(name)
            if previous is None:
                copy_spec(child)
            else:
                walk(previous, child)

    walk(before.pseudoRoot, after.pseudoRoot)
    return delta
