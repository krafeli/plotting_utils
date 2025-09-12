import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

# automatically import all modules in this package
package_dir = Path(__file__).parent

for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
    module = importlib.import_module(f".{module_name}", package=__name__)
    
    # expose all functions and classes from each module
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isclass(obj):
            globals()[name] = obj

# define __all__ for clarity
__all__ = [name for name, obj in globals().items()
           if inspect.isfunction(obj) or inspect.isclass(obj)]