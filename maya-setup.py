# Run to import yrig and its components
import sys
import os
from pathlib import Path
dev_path = Path("~/maya-python/y-rig").expanduser() # Replace with the path to your y-rig repo
venv_path = (dev_path / Path(".venv/lib/python3.11/site-packages")).resolve()
component_path = (dev_path / Path("components")).resolve()
sys.path.insert(0, str(dev_path))
sys.path.insert(0, str(venv_path))
import yrig
os.environ["MGEAR_SHIFTER_COMPONENT_PATH"] = str(component_path)

# Run to reload yrig during development
modules = [name for name in sys.modules.keys() if name.startswith("yrig")]
for name in modules:
    del sys.modules[name]
import yrig

# Run to open debug port
import debugpy
maya_path = Path(os.environ.get("MAYA_LOCATION")) # type:ignore
mayapy_path = maya_path / Path("bin/mayapy")
debugpy.configure({'python': str(mayapy_path)})
debugpy.listen(5678) # 5678 is the default attach port in the VS Code debug configurations. Host defaults to 127.0.0.1
