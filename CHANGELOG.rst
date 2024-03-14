Change Log:
==========

v0.0.3:
-------

- Update the dev dependencies to run with Python 3.12

- Fix resource file loading for changes to pint's load_definitions function

- Modify ``interlayer`` interpolation for changes to scipy interp2d

- Fix a bug in the ``interlayer`` interpolation

- Change setup-tool configuration to pyproject.toml from setup.cfg

v0.0.2:
-------

- Changed the defined glass types (annealed, heat-strengthened, and fully tempered)
  from being derived classes of ``GlassType`` and introduced a registry.

- Changed the ``InterLayer`` class to ``Interlayer``.

v0.0.1:
-------

- Initial release.
