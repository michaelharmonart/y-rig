from pathlib import Path

from pxr import Plug

_PLUGIN_INFO = Path(__file__).resolve().parents[3] / "usd/plugin/yrigSchema/resources/plugInfo.json"

if not _PLUGIN_INFO.is_file():
    raise FileNotFoundError(_PLUGIN_INFO)

Plug.Registry().RegisterPlugins(str(_PLUGIN_INFO))
