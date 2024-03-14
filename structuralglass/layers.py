"""
There are two layered materials commonly used in structural glass design:

    - Glass plys
    - Interlayers

Typically, interlayer materials are used to bond glass layers into a laminate.
Common commercial applications use either ionoplast (SGP) or polyvinyl butyral
(PVD) as the bonding layers between glass layers.


Glass Plys
----------

It is common for glass layers to be manufactured according to "nominal
thickness". Each nominal thickness has an associated min required thickness
(as documented in E1300). To facilitate this common practice a
:class:`~GlassPly` can be created using
:meth:`~GlassPly.from_nominal_thickness`::

    from structuralglass import Q_
    import structuralglass.layers as lay

    t_nom = Q_(6, "mm")
    ply = lay.GlassPly.from_nominal_thickness(t_nom)

    ply.t_min  # Q_(5.56, "mm")


:class:`~GlassPly` created in this way will have properties for
:attr:`~GlassPly.t_nom` that are not None.

In order to not limit the user to nominal thicknesses, a :class:`~GlassPly`
can be created using :meth:`~GlassPly.from_actual_thickness`::

    from structuralglass import Q_
    import structuralglass.layers as lay

    t_act = Q_(5.56, "mm")
    ply = lay.GlassPly.from_actual_thickness(t_act)

    ply.t_min  # Q_(5.56, "mm")

Interlayers
-----------
An :class:`~structuralglass.layers.Interlayer` can be defined in 2 ways:

- as a static interlayer
- as a dynamic interlayer

Static :class:`~structuralglass.layers.Interlayer` are not backed by
manufactures data. They are static in the sense that the shear modulus is
changed manually. A static :class:`~structuralglass.layers.Interlayer` can be
created via the :meth:`~structuralglass.layers.Interlayer.from_static` class
method.

::

    from structuralglass import Q_
    import structuralglass.layers as lay

    # Interlayer PVB at 30°C for 1 day load duration
    G_pvb = Q_(0.281, "MPa")
    t_pvb = Q_(0.89, "mm")
    interlayer = lay.Interlayer.from_static(t_pvb, G_pvb)

It is common for interlayer manufacturers to provide material properties in
tabulated forms. This is because the laminates are viscoelastic materials and
their material properties depend on load duration and temperature. So, the
manufacture provides tabulated "effective" data for a load duration and
temperature.

Dynamic :class:`~Interlayer` are backed by manufactures tabular data. They are
dynamic in the sense that the shear modulus can be changed by providing a new
temperature and load duration. For gaps in the manufactures data (for example,
the shear modulus is given for 10°C and 20°C and 15°C is set), the tabular
data is interpolated linearly. Extrapolation is not done and capped to
tabulated values. This functionality is provided by scipy's
:class:`~scipy.interpolate.RegularGridInterpolator` function. A dynamic
:class:`~Interlayer` can be created via the
:meth:`~Interlayer.from_product_table` class method.

::

    from structuralglass import Q_
    import structuralglass.layers as lay

    t_pvb = Q_(1.52, "mm")
    product_name = "Ionoplast Interlayer NCSEA"
    interlayer = lay.Interlayer.from_product_table(t_pvb, product_name)

    # set the load duration and temperature
    interlayer.duration = Q_(1, "month")
    interlayer.temperature = Q_(40, "degC")

    # Access the shear modulus of the "Ionoplast Interlayer NCSEA"
    interlayer.G    # Q_(3.29, "MPa")

A background registry holds the manufactures tabular data.
New data can be added via the :func:`~register_interlayer_product` function.
Data can be removed via the :func:`~deregister_interlayer_product` function.

::

    from structuralglass import Q_
    import structuralglass.layers as lay

    name = "product_ID_1"
    data = {
        (Q_(20, "degC"), Q_(3, 'sec')): Q_(240, "MPa"),
        (Q_(30, "degC"), Q_(3, 'sec')): Q_(217, "MPa"),
        (Q_(40, "degC"), Q_(3, 'sec')): Q_(151, "MPa"),

        (Q_(20, "degC"), Q_(10, 'min')): Q_(77.0, "MPa"),
        (Q_(30, "degC"), Q_(10, 'min')): Q_(36.2, "MPa"),
        (Q_(40, "degC"), Q_(10, 'min')): Q_(11.8, "MPa"),
    }
    lay.register_interlayer_product(name, data)

    # choose an interlayer thickness
    t_pvb = Q_(1.52, "mm")
    interlayer = lay.Interlayer.from_product_table(t_pvb, name)

    # set the load duration and temperature
    interlayer.duration = Q_(5, "min")
    interlayer.temperature = Q_(35, "degC")

    interlayer.G    # Q(104.15, "MPa")

"""

import numpy as np
import pint
from scipy import interpolate

from . import Q_, ureg

t_min_lookup_metric = {
    2.0: 1.80,
    2.5: 2.16,
    2.7: 2.59,
    3: 2.92,
    4: 3.78,
    5: 4.57,
    6: 5.56,
    8: 7.42,
    10: 9.02,
    12: 11.91,
    16: 15.09,
    19: 18.26,
    22: 21.44,
    25: 24.61,
}
"""
Lookup for the minimal allowable thickness. Key and value units are in mm.
"""

t_min_lookup_imperial = {
    0.09375: 2.16,
    0.125: 2.92,
    0.15625: 3.78,
    0.1875: 4.57,
    0.25: 5.56,
    0.3125: 7.42,
    0.375: 9.02,
    0.5: 11.91,
    0.625: 15.09,
    0.75: 18.26,
    0.875: 21.44,
    1: 24.61,
}
"""
Lookup for the minimal allowable thickness. Key and value units are in
inches and mm respectively.
"""


class GlassPly:
    """
    A class to represent a glass ply, its thinkess (nominal and minimum
    allowable) and mechanical properties.
    """

    @ureg.check(None, "[length]", None, "[pressure]")
    def __init__(self, t_min, t_nom=None, E=71.7 * ureg.GPa):
        """
        Constructor.

        Parameters
        ----------
        t_min : :class:`~pint.Quantity` [length]
            Min allowable thickness.
        t_nom : :class:`~pint.Quantity` [length], optional
            Nominal thickness, by default None (if using actual thickness)
        E : :class:`~pint.Quantity` [pressure], optional
            Elastic modulus, by default 71.7GPa

        Raises
        ------
        pint.DimensionalityError
            If an input parameters do not meet the :class:`~pint.Quantity`
            requirement.
        TypeError
            The provided nominal thickness is not a :class:`~pint.Quantity` or
            None.
        ValueError
            Actual thickness/elastic modulus/nominal thickness cannot be less
            than zero.
        ValueError
            The provided nominal thickness could not be found in the lookup.
        """

        # The check decorator can not be used to check t_nom (as it can be
        # None)
        if t_nom is not None:
            if isinstance(t_nom, Q_):
                if not t_nom.check("[length]"):
                    dim = t_nom.dimensionality
                    unit = t_nom.units
                    # Is a Quantity but not the corret dim
                    raise pint.DimensionalityError(
                        unit, "inch", dim, "[length]"
                    )
            else:
                raise TypeError("t_nom is not a Quanity['length'] or None.")

        if t_min < Q_(0, "inch"):
            raise ValueError("Actual thickness cannot be less than zero.")

        self.E = 71.7 * ureg.GPa
        self._t_min = t_min
        self._t_nom = t_nom

    @classmethod
    @ureg.check(None, "[length]")
    def from_nominal_thickness(cls, t_nom):
        """
        Class method to creating a :class:`GlassPly` with a nominal thickness.

        Parameters
        ----------
        t_nom : :class:`~pint.Quantity` [length]
            Nominal thickness.

        Returns
        -------
        :class:`GlassPly`
        """

        t_min = cls._find_min_from_nom(t_nom)
        return cls(t_min, t_nom)

    @classmethod
    @ureg.check(None, "[length]")
    def from_actual_thickness(cls, t_act):
        """
        Class method to creating a :class:`GlassPly` with an actual thickness.

        Parameters
        ----------
        t_act : :class:`~pint.Quantity` [length]
            Actual thickness.

        Returns
        -------
        :class:`GlassPly`
        """

        return cls(t_act)

    @staticmethod
    def _find_min_from_nom(t_nom):
        try:
            t_min = t_min_lookup_metric[t_nom.m_as("mm")] * ureg.mm
        except KeyError:
            try:
                t_min = Q_(t_min_lookup_imperial[t_nom.m_as("in")], "mm")
            except KeyError:
                raise ValueError(
                    "Could not find the nominal thickness of "
                    f"{t_nom} in the nominal thickness lookup."
                )
        return t_min

    @property
    def E(self):
        """
        The elastic modulus as :class:`~pint.Quantity` [pressure].

        Raises
        ------
        ValueError
            When set to a value less than 0MPa.
        """

        return self._E

    @E.setter
    @ureg.check(None, "[pressure]")
    def E(self, value):
        if value < Q_(0, "MPa"):
            raise ValueError("Elastic modulus cannot be less than zero.")
        self._E = value

    @property
    def t_nom(self):
        """
        The nominal thickness as :class:`~pint.Quantity` [length]

        Raises
        ------
        ValueError
            When set to a value less than 0mm.
        """

        return self._t_nom

    @t_nom.setter
    @ureg.check(None, "[length]")
    def t_nom(self, value):
        if value < Q_(0, "inch"):
            raise ValueError("Nominal thickness cannot be less than zero.")
        self._t_nom = value
        self._t_min = self._find_min_from_nom(value)

    @property
    def t_min(self):
        """
        The minimum thickness as :class:`~pint.Quantity` [length].

        Raises
        ------
        ValueError
            When set to a value less than 0mm.
        """

        return self._t_min

    @t_min.setter
    @ureg.check(None, "[length]")
    def t_min(self, value):
        if value < Q_(0, "inch"):
            raise ValueError("Actual thickness cannot be less than zero.")
        self._t_min = value
        self._t_nom = None


class Interlayer:
    """
    A class to represent a glass interlayer (e.g. PVB or SG), and its
    mechanical properties. Rate dependent properties can be considered via the
    use of a product table or registered product name.
    """

    def __init__(self, t, *, G=None, G_table=None):
        """
        Constructor

        Parameters
        ----------
        t : :class:`~pint.Quantity` [length]
            Interlayer thickness.
        G : :class:`~pint.Quantity` [pressure]
            Shear modulus for the case of a static layer, do not provide a
            G_table.
        G_table: :class:`dict` ((:class:`~pint.Quantity`, :class:`~pint.Quantity`): :class:`~pint.Quantity`)
            Shear modulus table for the case of using an interlayer product
            table. The keys of the dict are tuples of units [temperature] and
            [time] for the interlayer temperature and load duration,
            respectively. The values of the dict are of units [pressure] for
            the shear modulus. Do not provide a G value.

        Raises
        ------
        ValueError
            If neither G nor G_table are provided.
        ValueError
            If both G and G_table are provided.
        """
        self.t = t
        if G is None and G_table is None:
            raise ValueError("Either G or G_table must be provided.")
        elif G is not None and G_table is not None:
            raise ValueError("Only one of G or G_table must be provided.")
        self._G = G
        self.G_table = G_table
        if self.G_table is not None:
            self._temperature = None
            self._duration = None
            # Create a function that does the interpolation for the product
            # table. Get the unique values for tempereture in the table in °C
            val_x = (ii[0].m_as("degC") for ii in self.G_table.keys())
            G_table_x = np.sort(np.array(list(set(val_x))))
            # Get the unique values for duration in the table in sec
            val_y = (ii[1].m_as("sec") for ii in self.G_table.keys())
            G_table_y = np.sort(np.array(list(set(val_y))))
            # Create a meshgrid for the interpolation process
            x, y = np.meshgrid(G_table_x, G_table_y)
            # vectorize the look up for the tables (this is done as the
            # entries in the table may not be in order)
            vlookup = np.vectorize(
                lambda x, y: self.G_table[Q_(x, "degC"), Q_(y, "sec")].m_as(
                    "MPa"
                )
            )
            # Exicute the vectorized lookup
            G_table_z = vlookup(x, y).T
            # create the interploation function
            G_interp = interpolate.RegularGridInterpolator(
                (np.array(G_table_x), np.array(G_table_y)),
                G_table_z,
                method="linear",
            )
            # use a decorator to add dimensions to the interpolation function

            @ureg.wraps(ureg.MPa, (ureg.degC, ureg.second))
            def call_G_interp(x, y):
                try:
                    result = G_interp([x, y])
                except ValueError as err:
                    raise ValueError(
                        (
                            "Extrapolating G: The temperature-duration "
                            "combination is outside of the data tables bounds."
                        )
                    ) from err
                return result

            self.G_interp_dim = call_G_interp

    @classmethod
    @ureg.check(None, "[length]", None)
    def from_product_table(cls, t, product_name):
        """
        Class method for an :class:`Interlayer` with a product table.

        Parameters
        ----------
        t : :class:`~pint.Quantity` [length]
            The thickness of the interlayer.
        product_name : ``string``
            The registred name of the product.

        Returns
        -------
        :class:`Interlayer`
        """

        if not (t > Q_(0, "mm")):
            raise ValueError(
                "The thickness must be greater than zero " "[lengh]."
            )
        table = _interLayer_registry.get(product_name, None)
        if table is None:
            raise ValueError(
                "The product is not registered in the product " "registry."
            )
        return cls(t, G_table=table)

    @classmethod
    @ureg.check(None, "[length]", "[pressure]")
    def from_static(cls, t, G):
        """
        Class method for an :class:`Interlayer` with a static shear modulus.

        Parameters
        ----------
        t : :class:`~pint.Quantity` [length]
            The thickness of the interlayer.
        G : :class:`~pint.Quantity` [pressure]
            The shear modulus.

        Returns
        -------
        :class:`Interlayer`
        """

        if not (t > Q_(0, "mm")):
            raise ValueError(
                "The thickness must be greater than zero " "[lengh]."
            )
        if not (G > Q_(0, "MPa")):
            raise ValueError(
                "The shear modulus must be greater than zero " "[pressure]."
            )
        return cls(t, G=G)

    @property
    def temperature(self):
        """
        The temperature as :class:`~pint.Quantity` [temperature].

        Raises
        ------
        ValueError
            If no product table is provided.
        """

        if self.G_table is None:
            raise ValueError(
                "No product table provided. Static case being " "used."
            )
        return self._temperature

    @temperature.setter
    @ureg.check(None, "[temperature]")
    def temperature(self, value):
        if self.G_table is None:
            raise ValueError(
                "No product table provided. Static case being " "used."
            )
        self._temperature = value

    @property
    def duration(self):
        """
        The duration as :class:`~pint.Quantity` [time].

        Raises
        ------
        ValueError
            If no product table is provided.
        """

        if self.G_table is None:
            raise ValueError(
                "No product table provided. Static case being " "used."
            )
        return self._duration

    @duration.setter
    @ureg.check(None, "[time]")
    def duration(self, value):
        if self.G_table is None:
            raise ValueError(
                "No product table provided. Static case being " "used."
            )
        self._duration = value

    @property
    def G(self):
        """
        The shear modulus as :class:`~pint.Quantity` [pressure]. Interpolates
        linearly within the domain of the provided table.

        Raises
        ------
        ValueError
            If a product table is being used and the reference temperature
            and/or duration are not set.
        """

        if self._G is not None:
            return self._G
        else:
            try:
                return self.G_table[self.temperature, self.duration]
            except KeyError:
                if self.temperature is None or self.duration is None:
                    raise ValueError(
                        "Reference temperature and/or duration not set."
                    )
                return self.G_interp_dim(self.temperature, self.duration)[0]


_interLayer_registry = {}


def register_interlayer_product(product_name, data):
    """
    Register new interlayer product table.

    Parameters
    ----------
    product_name : :class:`str`
        String identifier
    data : :class:`dict` ((:class:`~pint.Quantity`, :class:`~pint.Quantity`): :class:`~pint.Quantity`)
        The tabulated data of the shear modulus that depends on temperature
        and load duration. The keys of the dict are tuples of units [temp] and
        [time]. The of the dict are of units [pressure]

    Raises
    ------
    ValueError
        If the provided data table is not rectangular. E.g. if shear modulus
        values are given for (20degC, 3s) and (30degC, 10min), then values for
        (30degC, 3s) and (20degC, 10min) must also be provided.
    """

    G_table_tmp = set([ii[0].m_as("degC") for ii in data.keys()])
    G_table_dur = set([ii[1].m_as("sec") for ii in data.keys()])
    G_table_val = list(data.values())
    # check that the provided data is "rectangular"
    if len(G_table_tmp) * len(G_table_dur) != len(G_table_val):
        raise ValueError("The provided data is not rectangular.")
    _interLayer_registry[product_name] = data


def deregister_interlayer_product(product_name):
    """
    Deregister an existing interlayer product table.

    Parameters
    ----------
    product_name : :class:`str`
        Identifier
    """

    _interLayer_registry.pop(product_name, None)


__name_II = "Ionoplast Interlayer NCSEA"
__data_II = {
    (Q_(10, "degC"), Q_(1, "sec")): Q_(240, "MPa"),
    (Q_(20, "degC"), Q_(1, "sec")): Q_(217, "MPa"),
    (Q_(24, "degC"), Q_(1, "sec")): Q_(200, "MPa"),
    (Q_(30, "degC"), Q_(1, "sec")): Q_(151, "MPa"),
    (Q_(40, "degC"), Q_(1, "sec")): Q_(77.0, "MPa"),
    (Q_(50, "degC"), Q_(1, "sec")): Q_(36.2, "MPa"),
    (Q_(60, "degC"), Q_(1, "sec")): Q_(11.8, "MPa"),
    (Q_(70, "degC"), Q_(1, "sec")): Q_(3.77, "MPa"),
    (Q_(80, "degC"), Q_(1, "sec")): Q_(1.55, "MPa"),
    (Q_(10, "degC"), Q_(3, "sec")): Q_(236, "MPa"),
    (Q_(20, "degC"), Q_(3, "sec")): Q_(211, "MPa"),
    (Q_(24, "degC"), Q_(3, "sec")): Q_(193, "MPa"),
    (Q_(30, "degC"), Q_(3, "sec")): Q_(141, "MPa"),
    (Q_(40, "degC"), Q_(3, "sec")): Q_(63.0, "MPa"),
    (Q_(50, "degC"), Q_(3, "sec")): Q_(26.4, "MPa"),
    (Q_(60, "degC"), Q_(3, "sec")): Q_(8.18, "MPa"),
    (Q_(70, "degC"), Q_(3, "sec")): Q_(2.93, "MPa"),
    (Q_(80, "degC"), Q_(3, "sec")): Q_(1.32, "MPa"),
    (Q_(10, "degC"), Q_(1, "min")): Q_(225, "MPa"),
    (Q_(20, "degC"), Q_(1, "min")): Q_(195, "MPa"),
    (Q_(24, "degC"), Q_(1, "min")): Q_(173, "MPa"),
    (Q_(30, "degC"), Q_(1, "min")): Q_(110, "MPa"),
    (Q_(40, "degC"), Q_(1, "min")): Q_(30.7, "MPa"),
    (Q_(50, "degC"), Q_(1, "min")): Q_(11.3, "MPa"),
    (Q_(60, "degC"), Q_(1, "min")): Q_(3.64, "MPa"),
    (Q_(70, "degC"), Q_(1, "min")): Q_(1.88, "MPa"),
    (Q_(80, "degC"), Q_(1, "min")): Q_(0.83, "MPa"),
    (Q_(10, "degC"), Q_(1, "hour")): Q_(206, "MPa"),
    (Q_(20, "degC"), Q_(1, "hour")): Q_(169, "MPa"),
    (Q_(24, "degC"), Q_(1, "hour")): Q_(142, "MPa"),
    (Q_(30, "degC"), Q_(1, "hour")): Q_(59.9, "MPa"),
    (Q_(40, "degC"), Q_(1, "hour")): Q_(9.28, "MPa"),
    (Q_(50, "degC"), Q_(1, "hour")): Q_(4.20, "MPa"),
    (Q_(60, "degC"), Q_(1, "hour")): Q_(1.70, "MPa"),
    (Q_(70, "degC"), Q_(1, "hour")): Q_(0.84, "MPa"),
    (Q_(80, "degC"), Q_(1, "hour")): Q_(0.32, "MPa"),
    (Q_(10, "degC"), Q_(1, "day")): Q_(190, "MPa"),
    (Q_(20, "degC"), Q_(1, "day")): Q_(146, "MPa"),
    (Q_(24, "degC"), Q_(1, "day")): Q_(111, "MPa"),
    (Q_(30, "degC"), Q_(1, "day")): Q_(49.7, "MPa"),
    (Q_(40, "degC"), Q_(1, "day")): Q_(4.54, "MPa"),
    (Q_(50, "degC"), Q_(1, "day")): Q_(2.82, "MPa"),
    (Q_(60, "degC"), Q_(1, "day")): Q_(1.29, "MPa"),
    (Q_(70, "degC"), Q_(1, "day")): Q_(0.59, "MPa"),
    (Q_(80, "degC"), Q_(1, "day")): Q_(0.25, "MPa"),
    (Q_(10, "degC"), Q_(1, "month")): Q_(171, "MPa"),
    (Q_(20, "degC"), Q_(1, "month")): Q_(112, "MPa"),
    (Q_(24, "degC"), Q_(1, "month")): Q_(73.2, "MPa"),
    (Q_(30, "degC"), Q_(1, "month")): Q_(11.6, "MPa"),
    (Q_(40, "degC"), Q_(1, "month")): Q_(3.29, "MPa"),
    (Q_(50, "degC"), Q_(1, "month")): Q_(2.18, "MPa"),
    (Q_(60, "degC"), Q_(1, "month")): Q_(1.08, "MPa"),
    (Q_(70, "degC"), Q_(1, "month")): Q_(0.48, "MPa"),
    (Q_(80, "degC"), Q_(1, "month")): Q_(0.21, "MPa"),
    (Q_(10, "degC"), Q_(10, "year")): Q_(153, "MPa"),
    (Q_(20, "degC"), Q_(10, "year")): Q_(86.6, "MPa"),
    (Q_(24, "degC"), Q_(10, "year")): Q_(26.0, "MPa"),
    (Q_(30, "degC"), Q_(10, "year")): Q_(5.31, "MPa"),
    (Q_(40, "degC"), Q_(10, "year")): Q_(2.95, "MPa"),
    (Q_(50, "degC"), Q_(10, "year")): Q_(2.00, "MPa"),
    (Q_(60, "degC"), Q_(10, "year")): Q_(0.97, "MPa"),
    (Q_(70, "degC"), Q_(10, "year")): Q_(0.45, "MPa"),
    (Q_(80, "degC"), Q_(10, "year")): Q_(0.18, "MPa"),
}

register_interlayer_product(__name_II, __data_II)

__name_PVB = "PVB NCSEA"
__data_PVB = {
    (Q_(20, "degC"), Q_(3, "sec")): Q_(8.060, "MPa"),
    (Q_(30, "degC"), Q_(3, "sec")): Q_(0.971, "MPa"),
    (Q_(40, "degC"), Q_(3, "sec")): Q_(0.610, "MPa"),
    (Q_(50, "degC"), Q_(3, "sec")): Q_(0.440, "MPa"),
    (Q_(20, "degC"), Q_(1, "min")): Q_(1.640, "MPa"),
    (Q_(30, "degC"), Q_(1, "min")): Q_(0.753, "MPa"),
    (Q_(40, "degC"), Q_(1, "min")): Q_(0.455, "MPa"),
    (Q_(50, "degC"), Q_(1, "min")): Q_(0.290, "MPa"),
    (Q_(20, "degC"), Q_(1, "hour")): Q_(0.840, "MPa"),
    (Q_(30, "degC"), Q_(1, "hour")): Q_(0.441, "MPa"),
    (Q_(40, "degC"), Q_(1, "hour")): Q_(0.234, "MPa"),
    (Q_(50, "degC"), Q_(1, "hour")): Q_(0.052, "MPa"),
    (Q_(20, "degC"), Q_(1, "day")): Q_(0.508, "MPa"),
    (Q_(30, "degC"), Q_(1, "day")): Q_(0.281, "MPa"),
    (Q_(40, "degC"), Q_(1, "day")): Q_(0.234, "MPa"),
    (Q_(50, "degC"), Q_(1, "day")): Q_(0.052, "MPa"),
    (Q_(20, "degC"), Q_(1, "month")): Q_(0.372, "MPa"),
    (Q_(30, "degC"), Q_(1, "month")): Q_(0.069, "MPa"),
    (Q_(40, "degC"), Q_(1, "month")): Q_(0.052, "MPa"),
    (Q_(50, "degC"), Q_(1, "month")): Q_(0.052, "MPa"),
    (Q_(20, "degC"), Q_(1, "year")): Q_(0.266, "MPa"),
    (Q_(30, "degC"), Q_(1, "year")): Q_(0.052, "MPa"),
    (Q_(40, "degC"), Q_(1, "year")): Q_(0.052, "MPa"),
    (Q_(50, "degC"), Q_(1, "year")): Q_(0.052, "MPa"),
}

register_interlayer_product(__name_PVB, __data_PVB)
