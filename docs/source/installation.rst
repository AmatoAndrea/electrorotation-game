Installation & setup
====================

This project supports Python 3.10–3.13 and relies on standard scientific
libraries (NumPy, OpenCV, PySide6). These dependencies are installed
automatically when you install the package.

Python version requirements
---------------------------

Before installing this package, ensure you have a compatible Python version (3.10–3.13).
To check your current Python version:

.. code-block:: bash

   python3 --version

If you need to install a compatible Python version, follow the instructions for your
operating system below.

macOS
~~~~~

**Using Homebrew** (recommended):

.. code-block:: bash

   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install Python 3.12
   brew install python@3.12

**Using pyenv** (for managing multiple Python versions):

.. code-block:: bash

   # Install pyenv
   brew install pyenv
   
   # Install Python 3.12
   pyenv install 3.12.7
   
   # Set as local version for this project
   cd /path/to/electrorotation-game
   pyenv local 3.12.7

Linux
~~~~~

**Ubuntu/Debian**:

.. code-block:: bash

   # Update package list
   sudo apt update
   
   # Install Python 3.12
   sudo apt install python3.12 python3.12-venv python3.12-dev

**Fedora/RHEL**:

.. code-block:: bash

   sudo dnf install python3.12 python3.12-devel

**Using pyenv** (recommended for multiple versions):

.. code-block:: bash

   # Install pyenv dependencies
   sudo apt install -y make build-essential libssl-dev zlib1g-dev \
     libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
     libncurses5-dev libncursesw5-dev xz-utils tk-dev
   
   # Install pyenv
   curl https://pyenv.run | bash
   
   # Add to your shell configuration (~/.bashrc or ~/.zshrc)
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv init -)"
   
   # Install Python 3.12
   pyenv install 3.12.7
   pyenv local 3.12.7

Windows
~~~~~~~

**Using the official installer** (recommended):

1. Download Python 3.12 from `python.org <https://www.python.org/downloads/>`_
2. Run the installer
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation:

.. code-block:: powershell

   python --version

**Using pyenv-win**:

.. code-block:: powershell

   # Install pyenv-win using PowerShell
   Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
   
   # Install Python 3.12
   pyenv install 3.12.7
   pyenv local 3.12.7

Installing with Poetry
------------------------

If you manage your environment with `Poetry <https://python-poetry.org/>`, clone
the repository and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/AmatoAndrea/electrorotation-game.git
   cd electrorotation-game
   poetry install --no-interaction

After installation, you can invoke the command-line interface via
``poetry run rot-game``.

Alternative installation using pip
----------------------------------

If you prefer a plain ``pip`` installation, you can install:

.. code-block:: bash

   pip install .

The ``rot-game`` script will then be available on your ``PATH``.

Setting up assets
-----------------

The simulator requires a folder with a sequence of background video frames
and a collection of cell templates with matching masks. Place all assets under
a common directory and specify their locations via environment variables:

.. code-block:: bash

   export ROT_GAME_ASSETS=/absolute/path/to/templates
   export ROT_GAME_OUTPUTS=/absolute/path/to/outputs

Within ``ROT_GAME_ASSETS`` the expected structure is:

.. code-block:: text

   backgrounds/
     <background images or frame folders>
   cell_lines/
     <line_name>/
       cells/   # template images
       masks/   # matching binary masks

See :doc:`quickstart` for examples of how to point the simulator to these assets.
