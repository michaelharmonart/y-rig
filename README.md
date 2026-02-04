# y-rig: BYU Animation Program Rigging Framework ![y-rig logo](docs/images/y-rig_logo.svg)

---

This repository contains two main pieces:

1. A set of custom mGear Shifter components.
2. A library of tools that can be used by those components or mGear custom step scripts.

The main aim being to expand on the feature set of an established rig build system (mGear),
without needing the high maintanence burden of a fork.

## Developing/Using y-rig

After cloning this repo to a convenient place, run `git config --local core.hooksPath .githooks/` to set up the githooks, these will then install the needed dependencies.  
You'll also need to install mGear. See [mGear Website](https://mgear-framework.com/).
To have LSP support for the mgear module make sure to pull the submodule as well: `git submodule update --init --recursive`

After opening Maya you can configure maya and import y-rig for use with the following script (you can find it in `maya-setup.py`)

```
# Run to import yrig and its components
import sys
import os
from pathlib import Path
dev_path = Path("~/maya-python/y-rig").expanduser() # Replace with the path to your y-rig repo
yrig_path = (dev_path / Path("src")).resolve()
venv_path = (dev_path / Path(".venv/lib/python3.11/site-packages")).resolve()
component_path = (dev_path / Path("shifter/components")).resolve()
sys.path.insert(0, str(yrig_path))
sys.path.insert(0, str(venv_path))
import yrig
os.environ["MGEAR_SHIFTER_COMPONENT_PATH"] = str(component_path)
```

During development if you have made changes to the yrig library, you can re-import to test your changes immediately.

```
# Run to reload yrig during development
import sys
modules = [name for name in sys.modules.keys() if name.startswith("yrig")]
for name in modules:
    del sys.modules[name]
import yrig
```

You can also open a debug port to set breakpoints and test yrig library code.

```
# Run to open debug port
import os
from pathlib import Path
import debugpy
maya_path = Path(os.environ.get("MAYA_LOCATION")) # type:ignore
mayapy_path = (maya_path / Path("bin/mayapy")).resolve()
debugpy.configure({'python': str(mayapy_path)})
debugpy.listen(5678) # 5678 is the default attach port in the VS Code debug configurations. Host defaults to 127.0.0.1
```
