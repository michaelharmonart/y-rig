# Shifter Components

This directory is loaded as part of y-rig and is home to our custom mGear Shifter Components.

## Component UI

By default mGear's shifter components are authored as QT Designer XML files, which are compiled into the settingsUI.py file through an mGear tool.
To author component UIs in this codebase run `uvx --from pyside6-essentials pyside6-designer` to launch QT Designer and make the needed changes to the
XML file (settingsUI.py). You can then from that tool or the command line convert it to python code. To keep the type checker happy, change all
`from Pyside6 import` lines to `from Qt import`.

At some point we may automate this but for now that's what's required.
