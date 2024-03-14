import pint
import importlib.resources as pkg_resources

from . import resources

# look up the resource for the units file
unit_file = pkg_resources.files(resources).joinpath("unit_def.txt")

# Setup pint for the package
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
ureg.load_definitions(str(unit_file))
