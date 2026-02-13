def get_short_name(transform: str) -> str:
    """Return the leaf node name from a DAG path, stripping all parent namespaces.

    Maya DAG paths use ``|`` as a separator (e.g. ``|group1|joint1``).
    This function returns only the last component of such a path.

    Args:
        transform: A full or partial Maya DAG path string.

    Returns:
        The short (leaf) name without any leading path components.
    """
    return transform.rsplit("|", 1)[-1]
