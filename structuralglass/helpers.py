from scipy import interpolate

from . import Q_, ureg


class Roarks4side:
    """
    Roarks four sided simply supported plate calculations.
    """

    @ureg.check(None, "[pressure]", "[length]", "[length]", "[length]")
    def __init__(self, E, dim_x, dim_y, t=Q_(1, "inch")):
        """
        Constructor.

        Parameters
        ----------
        dim_x: :class:`~pint.Quantity` ['length']
            The x dimension of the plate.
        dim_y: :class:`~pint.Quantity` ['length']
            The y dimension of the plate.
        E: :class:`~pint.Quantity` ['pressure']
            The elastic modulus of the plate.
        t: :class:`~pint.Quantity` ['length'], optional
            The plate thickness. The default is 1 inch.
        """

        self._ratio = None
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.t = t
        self.E = E
        self._reset_ratio()

        ratios = [1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 1e16]
        beta = [
            0.2874,
            0.3762,
            0.453,
            0.5172,
            0.5688,
            0.6102,
            0.7134,
            0.741,
            0.7476,
            0.75,
        ]
        alpha = [
            0.044,
            0.0616,
            0.077,
            0.0906,
            0.1017,
            0.111,
            0.1335,
            0.14,
            0.1417,
            0.1421,
        ]
        gamma = [
            0.42,
            0.455,
            0.478,
            0.491,
            0.499,
            0.503,
            0.505,
            0.502,
            0.501,
            0.5,
        ]

        self._beta = interpolate.interp1d(ratios, beta)
        self._alpha = interpolate.interp1d(ratios, alpha)
        self._gamma = interpolate.interp1d(ratios, gamma)

    @ureg.check(None, "[pressure]")
    def stress_max(self, q):
        """
        Calculate the plates max stress.

        Parameters
        ----------
        q : :class:`~pint.Quantity` ['pressure']
            The uniform load on the plate.

        Returns
        -------
        :class:`~pint.Quantity` ['pressure']
            The plates max stress.
        """

        value = self._beta(self._ratio) * q * (self._b / self.t) ** 2
        return value.to_reduced_units()

    @ureg.check(None, "[pressure]")
    def deflection_max(self, q):
        """
        Calculate the plates max deflection.

        Parameters
        ----------
        q : :class:`~pint.Quantity` ['pressure']
            The uniform load on the plate.

        Returns
        -------
        :class:`~pint.Quantity` ['length']
            The plates max deflection.
        """

        value = (
            -self._alpha(self._ratio) * q * self._b**4 / (self.E * self.t**3)
        )
        return value.to_reduced_units()

    @ureg.check(None, "[pressure]")
    def reaction_max(self, q):
        """
        Calculate the plates max reaction force.

        Parameters
        ----------
        q : Quantity['pressure']
            The uniform load on the plate.

        Returns
        -------
        :class:`~pint.Quantity` ['force_per_unit_length']
            The plates max reaction force.
        """

        value = self._gamma(self._ratio) * q * self._b
        return value.to_reduced_units()

    def _reset_ratio(self):
        if self.dim_x > self.dim_y:
            self._a, self._b = (self.dim_x, self.dim_y)
        else:
            self._a, self._b = (self.dim_y, self.dim_x)
        self._ratio = float(self._a / self._b)

    @property
    def dim_x(self):
        """
        Get the x dimension of the plate in :class:`~pint.Quantity` ['length']

        Raises
        ------
        ValueError
            If the dimension is less than 0 inch/mm
        """

        return self._dim_x

    @dim_x.setter
    @ureg.check(None, "[length]")
    def dim_x(self, value):
        if value < Q_(0, "mm"):
            raise ValueError("Dimensions must be greater than zero.")
        self._dim_x = value
        if self._ratio is not None:
            self._reset_ratio()

    @property
    def dim_y(self):
        """
        The y dimension of the plate in :class:`~pint.Quantity` ['length']

        Raises
        ------
        ValueError
            If the dimension is less than 0 inch/mm
        """

        return self._dim_y

    @dim_y.setter
    @ureg.check(None, "[length]")
    def dim_y(self, value):
        if value < Q_(0, "mm"):
            raise ValueError("Dimensions must be greater than zero.")
        self._dim_y = value
        if self._ratio is not None:
            self._reset_ratio()

    @property
    def t(self):
        """
        The thickness of the plate in :class:`~pint.Quantity` ['length'].

        Raises
        ------
        ValueError
            If the thickness is less than 0 inch/mm.
        """

        return self._t

    @t.setter
    @ureg.check(None, "[length]")
    def t(self, value):
        if value < Q_(0, "mm"):
            raise ValueError("Thickness must be greater than zero.")
        self._t = value

    @property
    def E(self):
        """
        The elastic modulus of the plate in
        :class:`~pint.Quantity` ['pressure'].

        Raises
        ------
        ValueError
            If the elastic modulus is less than 0 ksi/MPa
        """

        return self._E

    @E.setter
    @ureg.check(None, "[pressure]")
    def E(self, value):
        if value < Q_(0, "MPa"):
            raise ValueError("Elastic modulus must be greater than zero.")
        self._E = value
