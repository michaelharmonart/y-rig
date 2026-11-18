---
weight: 10
title: "Getting Started Developing with y-rig"
description: ""
icon: "terminal"
date: "2026-10-17T16:50:59-06:00"
lastmod: "2026-10-17T16:50:59-06:00"
toc: true
---

> [!WARNING]
> You probably have important things to do, but before you touch this codebase, please read [this foundational document on the y-rig approach]({{< ref path="docs/approach.md" >}}).

After cloning the y-rig repo to a convenient place, run `git config --local core.hooksPath .githooks/` to set up the githooks,
these will then install the needed dependencies and sync your working directory dependencies on every checkout.
They will also run pre commit checks for code quality/standards.  
You'll also need to install mGear. See [mGear Website](https://mgear-framework.com/).
To have LSP support for the mgear module make sure to pull the submodule as well: `git submodule update --init --recursive`

After opening Maya you can configure maya and import y-rig for use with the following script (you can find it in `maya-setup.py`)
Make sure to run this BEFORE you build a rig with mGear if you're developing a component.
Shifter's reload component function will reload the components but won't update them to the ones in a new path if you change
the MGEAR_SHIFTER_COMPONENT_PATH environment variable at runtime.

```py
# Run to import yrig and its components
import sys
import os
from pathlib import Path
dev_path = Path("~/maya-python/y-rig").expanduser() # Replace with the path to your y-rig repo
yrig_path = (dev_path / Path("src")).resolve()
venv_path = (dev_path / Path(".venv/lib/python3.11/site-packages")).resolve()
if str(yrig_path) not in sys.path:
    sys.path.insert(0, str(yrig_path))
if str(venv_path) not in sys.path:
    sys.path.insert(0, str(venv_path))
import yrig
component_path = (dev_path / Path("shifter/components")).resolve()
os.environ["MGEAR_SHIFTER_COMPONENT_PATH"] = str(component_path)
```

During development if you have made changes to the yrig library, you can re-import to test your changes immediately.

```py
# Run to reload yrig during development
import sys
modules = [name for name in sys.modules.keys() if name.startswith("yrig")]
for name in modules:
    del sys.modules[name]
import yrig
```

You can also open a debug port to set breakpoints and test yrig library code.

```py
# Run to open debug port
import os
from pathlib import Path
import debugpy
maya_path = Path(os.environ.get("MAYA_LOCATION")) # type:ignore
mayapy_path = (maya_path / Path("bin/mayapy")).resolve()
debugpy.configure({'python': str(mayapy_path)})
debugpy.listen(5678) # 5678 is the default attach port in the VS Code debug configurations. Host defaults to 127.0.0.1
```
