#!/bin/bash
# Build script to handle distutils compatibility

# Install setuptools first to provide distutils
pip install setuptools==68.2.2

# Install other dependencies
pip install -r requirements.txt

# Create a distutils compatibility module if needed
python -c "
import sys
if sys.version_info >= (3, 12):
    import setuptools
    sys.modules['distutils'] = setuptools
    sys.modules['distutils.util'] = setuptools.util
    sys.modules['distutils.version'] = setuptools.version
    sys.modules['distutils.errors'] = setuptools.errors
    sys.modules['distutils.command'] = setuptools.command
    sys.modules['distutils.command.build'] = setuptools.command.build
    sys.modules['distutils.command.build_ext'] = setuptools.command.build_ext
    sys.modules['distutils.command.build_py'] = setuptools.command.build_py
    sys.modules['distutils.command.install'] = setuptools.command.install
    sys.modules['distutils.command.install_lib'] = setuptools.command.install_lib
    sys.modules['distutils.command.install_scripts'] = setuptools.command.install_scripts
    sys.modules['distutils.command.sdist'] = setuptools.command.sdist
    sys.modules['distutils.core'] = setuptools.core
    sys.modules['distutils.dist'] = setuptools.dist
    sys.modules['distutils.extension'] = setuptools.extension
    sys.modules['distutils.file_util'] = setuptools.file_util
    sys.modules['distutils.log'] = setuptools.log
    sys.modules['distutils.spawn'] = setuptools.spawn
    sys.modules['distutils.sysconfig'] = setuptools.sysconfig
    sys.modules['distutils.text_file'] = setuptools.text_file
    sys.modules['distutils.util'] = setuptools.util
    sys.modules['distutils.version'] = setuptools.version
    print('Distutils compatibility module created')
"
