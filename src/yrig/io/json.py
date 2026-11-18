import io
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

import msgspec.json

T = TypeVar("T")

_msgspec_encoder = msgspec.json.Encoder()


def _encode_json_key(key: Any) -> bytes:  # noqa: ANN401
    # This is kinda a hack to get a properly encoded json key according to msgspec.
    return _msgspec_encoder.encode({key: None})[1:].rsplit(b":", 1)[0]


def _is_compact(obj: list | tuple | dict, threshold: int) -> bool:
    """Return whether a container should be rendered on a single line."""
    return len(obj) > threshold


def _write_compact_pretty(
    obj: Any,  # noqa:  ANN401
    buffer: BinaryIO,
    indent: int = 0,
    pad: str = "  ",
    threshold: int = 128,
) -> None:
    """Write an object as JSON with large containers kept on a single line.

    Containers at or below ``threshold`` are formatted across multiple lines,
    while larger lists, tuples, and dictionaries are encoded compactly using
    ``msgspec``. Nested containers are formatted recursively.

    Args:
        obj: Object to encode as JSON.
        buffer: Binary file-like object to write the encoded JSON to.
        indent: Current indentation depth.
        pad: String used for one level of indentation.
        threshold: Minimum container length at which compact formatting is used.
    """
    prefix = (pad * indent).encode()
    child_prefix = (pad * (indent + 1)).encode()

    if isinstance(obj, (list, tuple)):
        if not obj:
            buffer.write(b"[]")
            return
        if _is_compact(obj, threshold):
            buffer.write(_msgspec_encoder.encode(obj))
            return
        buffer.write(b"[\n")
        last = len(obj) - 1
        for i, v in enumerate(obj):
            buffer.write(child_prefix)
            _write_compact_pretty(v, buffer, indent + 1, pad, threshold)
            buffer.write(b",\n" if i != last else b"\n")
        buffer.write(prefix)
        buffer.write(b"]")

    elif isinstance(obj, dict):
        if not obj:
            buffer.write(b"{}")
            return
        if _is_compact(obj, threshold):
            buffer.write(_msgspec_encoder.encode(obj))
            return
        n = len(obj)

        buffer.write(b"{\n")
        last = len(obj) - 1
        for i, (k, v) in enumerate(obj.items()):
            buffer.write(child_prefix)
            buffer.write(_encode_json_key(k))
            buffer.write(b": ")
            _write_compact_pretty(v, buffer, indent + 1, pad, threshold)
            buffer.write(b",\n" if i != last else b"\n")
        buffer.write(prefix)
        buffer.write(b"}")

    else:
        buffer.write(_msgspec_encoder.encode(obj))


def _encode_json(obj: Any, pretty: bool = True, compact: bool = True) -> bytes:  # noqa: ANN401
    """Encode an object to JSON bytes using the selected formatting scheme."""
    if not pretty:
        return _msgspec_encoder.encode(obj)

    if compact:
        buffer = io.BytesIO()
        _write_compact_pretty(msgspec.to_builtins(obj), buffer)
        return buffer.getvalue()

    return msgspec.json.format(_msgspec_encoder.encode(obj), indent=2)


def loads_json(data: str | bytes, type: type[T]) -> T:
    """Decode a JSON string (or bytes) into the specified type.

    Args:
        data: JSON text to decode.
        type: Type to decode the JSON data into.

    Returns:
        The decoded object.
    """
    return msgspec.json.decode(data, type=type)


def dumps_json(obj: Any, pretty: bool = False, compact: bool = False) -> str:  # noqa: ANN401
    """Encode an object as a JSON string.

    Args:
        obj: Object to encode as JSON.
        pretty: Whether to use the pretty formatting scheme.
        compact: When True use the compact formatting scheme (only does anything if ``pretty`` is also True).

    Returns:
        The encoded JSON as a string.
    """
    return _encode_json(obj, pretty=pretty, compact=compact).decode()


def load_json(filepath: Path, type: type[T]) -> T:
    """Load and decode a JSON file into the specified type.

    Args:
        filepath: Path to the JSON file.
        type: Type to decode the JSON data into.

    Returns:
        The decoded object.
    """
    data = filepath.read_bytes()
    return msgspec.json.decode(data, type=type)


def export_json(filepath: Path, obj: Any, pretty: bool = True, compact: bool = True) -> None:  # noqa:  ANN401
    """
    Encode an object as JSON and write it to a file.

    When ``compact`` is enabled large containers are kept compact on a single
    line. This produces human-readable JSON while avoiding large,
    noisy diffs for big data (like weights, blendshape deltas, etc).

    Args:
        filepath: Path to the output JSON file.
        obj: Object to encode as JSON.
        pretty: Whether to use the pretty formatting scheme.
        compact: When True use the compact formatting scheme (only does anything if ``pretty`` is also True).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(_encode_json(obj, pretty=pretty, compact=compact))
