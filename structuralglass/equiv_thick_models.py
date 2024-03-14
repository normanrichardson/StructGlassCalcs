"""
Equivalent thickness models are a popular treatment of glass mechanics.
The following models are common in practice:

 * monolithic method,
 * non-composite method, and
 * shear transfer coefficient method.

To facilitate these common models :class:`~MonolithicMethod`,
:class:`~NonCompositeMethod`, and :class:`~ShearTransferCoefMethod` can be
used.

::

    from structuralglass import Q_
    import structuralglass.layers as lay
    import structuralglass.equiv_thick_models as et

    t1nom = Q_(6, "mm")
    t2nom = Q_(6, "mm")

    # Plys
    ply1 = lay.GlassPly.from_nominal_thickness(t1nom)
    ply2 = lay.GlassPly.from_nominal_thickness(t2nom)

    # Use the monolithic method
    package = et.MonolithicMethod([ply1, ply2])
    # effective thickness of the package for delfection
    package.h_efs[ply1]     # Q_(11.12, "mm")

    # Use the non-composite method
    package = et.NonCompositeMethod([ply1, ply2])
    # effective thickness of the package for delfection
    package.h_efs[ply1]     # Q_(7.86, "mm")

These classes are derived from :class:`~GlassLiteEquiv`. Unique formulations
of the mechanical behavior can be derived from the abstract class
:class:`~GlassLiteEquiv`. When doing so the following abstract methods need to
be definded:

 * :meth:`~GlassLiteEquiv._calc_equiv_thickness`
 * :meth:`~GlassLiteEquiv._determine_package_E`
 * :meth:`~GlassLiteEquiv._validate`

The :meth:`~GlassLiteEquiv._calc_equiv_thickness` should set the private
variables for the packages effective thickness for displacement
:attr:`~GlassLiteEquiv._h_efw` and stress :attr:`~GlassLiteEquiv._h_efs`.

The :meth:`~GlassLiteEquiv._determine_package_E` should set the private
variables for the packages elastic modulus :attr:`~GlassLiteEquiv._E`.

The :meth:`~GlassLiteEquiv._validate` should perform an arbitaty validation
check on the provided plys. It must return a Tuple(bool, string). The bool is
True if the package is valid. If the package is not valid, the bool is False
and a string message is returned. The message is used to raise a ValueError.
"""

import abc
import itertools

from . import Q_, ureg
from .layers import GlassPly, Interlayer


class GlassLiteEquiv:
    """
    An abstract class to represent the mechanical behavior of a liminate.
    Unique formulations of the mechanical behaviour are derived from this
    class.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, ply):
        """
        Constructor.

        Parameters
        ----------
        ply : :class:`list` (:class:`~structuralglass.layers.GlassPly` and/or :class:`~structuralglass.layers.Interlayer`)
            The list of glass layers and interlayers that form a laminate.
        """
        self._h_efw = None
        """:class:`dict` (:class:`~structuralglass.layers.GlassPly`,
            :class:`~pint.Quantity` [length]): Private attribute for the
            effective thickness for displacement."""
        self._h_efs = None
        """:class:`dict` (:class:`~structuralglass.layers.GlassPly`,
           :class:`~pint.Quantity` [length]): Private attribute for the
           effective thickness for stress"""
        self._E = None
        """:class:`~pint.Quantity` [pressure]: Private attribute for the
        packages elastic modulus"""
        self.ply = ply

    @abc.abstractmethod
    def _calc_equiv_thickness(self):
        """Abstract method to determine the equivalent thicknesses."""

        return

    @abc.abstractmethod
    def _validate(self, plys):
        """
        Abstract method to valadate the plys.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly` and/or :class:`~structuralglass.layers.Interlayer`)

        Returns
        -------
        (:class:`bool`, :class:`str`)
            True for a valid formulation. A mesage for an error.
        """

        return

    @abc.abstractmethod
    def _determine_package_E(self):
        """
        Abstract method to determine the packages elastic modulus.
        """

        return

    @property
    def h_efw(self):
        """
        The equivalent laminate thickness for displacement as a
        :class:`~pint.Quantity` [length]
        """

        return self._h_efw

    @property
    def h_efs(self):
        """
        The equivalent laminate thickness for the stress in the associated
        plys as a :class:`dict` (:class:`~structuralglass.layers.GlassPly`,
        :class:`~pint.Quantity` [length])
        """

        return self._h_efs

    @property
    def ply(self):
        """
        A list of glass layers and interlayers that form a laminate as a
        :class:`list` (:class:`~structuralglass.layers.GlassPly`
        and/or :class:`~structuralglass.layers.Interlayer`)

        Raises
        ------
        ValueError
            If the provided list fails the formulations validation function.
        """

        return self._ply

    @property
    def E(self):
        """
        The elastic modulus for the formulation as a
        :class:`~pint.Quantity` [pressure]
        """

        return self._E

    @ply.setter
    def ply(self, value):
        valid, msg = self._validate(value)
        if not valid:
            raise ValueError(f"Ply validation failed: {msg}")
        self._ply = value
        self._determine_package_E()
        self._calc_equiv_thickness()


class MonolithicMethod(GlassLiteEquiv):
    """
    A basic/naive mechanical behavior that assumes the laminates effective
    thickness is the sum of the plys minimum thicknesses. This assumption
    applies to both stress and deflection effective thickness. This models
    validity decreases with load duration.
    """

    def __init__(self, plys):
        """
        Constructor for a liminate's behaviour.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly`)
            The list of glass layers that form a laminate.
        """

        super(MonolithicMethod, self).__init__(plys)

    def _calc_equiv_thickness(self):
        """
        Calculates the effective thicknesses.

        The sum of the plys minimum thicknesses is used.
        """

        self._h_efw = sum((ii.t_min for ii in self._ply))
        self._h_efs = dict(
            zip(self._ply, itertools.repeat(self.h_efw, len(self._ply)))
        )

    def _determine_package_E(self):
        """
        Assign the elastic modulus for the formulation.

        As the validation function is already checks the elastic modulus
        of all :class:`~structuralglass.layers.GlassPly` are the same, take
        the 1st plys elastic modulus.
        """

        self._E = self.ply[0].E

    def _validate(self, plys):
        """
        Validate the data in the plys.

        1. Checks the list is only contains type
           :class:`~structuralglass.layers.GlassPly`.
        2. Checks the elastic modulus of all
           :class:`~structuralglass.layers.GlassPly` are the same.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly`)

        Returns
        -------
        (:class:`bool`, :class:`str`)
            True for a valid formulation. A mesage for an error.
        """

        msg = ""
        valid = all((isinstance(ii, GlassPly) for ii in plys))
        if valid:
            set_E = set([ii.E for ii in plys])
            valid = len(set_E) == 1
            if not valid:
                msg = "The plys must have the same elastic modulus."
        else:
            msg = "Method is only formulated for a list of GlassPly."
        return valid, msg


class NonCompositeMethod(GlassLiteEquiv):
    """
    A simple mechanical behavior that assumes the laminates effective
    thickness is:

        * for deflection: the square root of the minimum thicknesses squared
        * for stress: the cubed root of the minimum thicknesses cubed

    This models assumes a non-composite behavior.
    """

    def __init__(self, plys):
        """
        Constructor for a liminate's behaviour.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly`)
            The list of glass layers that form a laminate.
        """

        super(NonCompositeMethod, self).__init__(plys)

    def _calc_equiv_thickness(self):
        """
        Calculate the effective thicknesses.

        The laminates effective thickness is for:

            * deflection: the square root of the sum of the minimum
              thicknesses squared
            * stress: cubed root of the sum of the minimum thicknesses cubed

        """

        def func(n):
            return sum(ii.t_min**n for ii in self.ply) ** (1 / n)

        self._h_efs = dict(
            zip(self.ply, itertools.repeat(func(2), len(self.ply)))
        )
        self._h_efw = func(3)

    def _determine_package_E(self):
        """Elastic modulus for the formulation.

        As the validation function is already checks the elastic modulus
        of all :class:`~structuralglass.layers.GlassPly` are the same, take
        the 1st plys elastic modulus.
        """

        self._E = self.ply[0].E

    def _validate(self, plys):
        """
        Method that validates the data in the plys.

        1. Checks the list is only contains type
           :class:`~structuralglass.layers.GlassPly`
        2. Checks the elastic modulus of all
           :class:`~structuralglass.layers.GlassPly` are the same.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly`)

        Returns
        -------
        (:class:`bool`, :class:`str`)
            True for a valid formulation. A mesage for an error.
        """

        msg = ""
        valid = all((isinstance(ii, GlassPly) for ii in plys))
        if valid:
            set_E = set([ii.E for ii in plys])
            valid = len(set_E) == 1
            if not valid:
                msg = "The plys must have the same elastic modulus."
        else:
            msg = "Method is only formulated for a list of GlassPly."
        return valid, msg


class ShearTransferCoefMethod(GlassLiteEquiv):
    """
    A mechanical behavior that takes the shear modulus of the laminate into
    consideration. This method was originally proposed by Bennison-Wolfel and
    referenced in ASTM E1300. It is only valid for a two glass ply laminate.
    Ref literature for limitations.
    """

    def __init__(self, plys, panel_min_dim):
        """
        Constructor for a liminate's behaviour.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly` and :class:`~structuralglass.layers.Interlayer`)
            The list of glass layers and Interlayers that form a laminate.
        panel_min_dim : :class:`~pint.Quantity` [length]
            Minimum dimension of the rectangular panel
        """

        self._panel_min_dim = panel_min_dim
        self.beta = 9.6
        super(ShearTransferCoefMethod, self).__init__(plys)

    def _validate(self, plys):
        """
        Validate the data in the plys.

         1. This formulation is only valid for plys with that are
            [:class:`~structuralglass.layers.GlassPly`,
            :class:`~structuralglass.layers.Interlayer`,
            :class:`~structuralglass.layers.GlassPly`].
         2. The :class:`~structuralglass.layers.GlassPly`'s must have the same
            elastic modulus.

        Parameters
        ----------
        plys : :class:`list` (:class:`~structuralglass.layers.GlassPly` and :class:`~structuralglass.layers.Interlayer`)
            The list to be validated.

        Returns
        -------
        (:class:`bool`, :class:`str`)
            True for a valid formulation. A mesage for an error.
        """

        valid, msg = False, ""
        if len(plys) == 3:
            if isinstance(plys[0], GlassPly):
                if isinstance(plys[2], GlassPly):
                    if isinstance(plys[1], Interlayer):
                        valid = True
        if valid:
            set_E = set([plys[0].E, plys[2].E])
            if len(set_E) != 1:
                msg = "The plys must have the same elastic modulus."
        else:
            msg = (
                "Method is only valid a list of [GlassPly, Interlayer, "
                "GlassPly]."
            )
        return valid, msg

    def _calc_equiv_thickness(self):
        """
        Calculate the effective thicknesses.

        This method was originally proposed by Bennison-Wolfel and referenced
        in ASTM E1300.
        """

        h_1 = self.ply[0].t_min
        h_2 = self.ply[2].t_min
        h_v = self.ply[1].t
        G_interlayer = self.ply[1].G
        E_glass = self.E
        h_s = 0.5 * (h_1 + h_2) + h_v
        h_s1 = h_s * h_1 / (h_1 + h_2)
        h_s2 = h_s * h_2 / (h_1 + h_2)
        I_s = h_1 * h_s2**2 + h_2 * h_s1**2
        self.Gamma = 1.0 / (
            1.0
            + self.beta
            * E_glass
            * I_s
            * h_v
            / (G_interlayer * h_s**2 * self.panel_min_dim**2)
        )
        self._h_efw = (h_1**3 + h_2**3 + 12 * self.Gamma * I_s) ** (1 / 3.0)
        self._h_efs = {}
        self._h_efs[self.ply[0]] = (
            self.h_efw**3 / (h_1 + 2 * self.Gamma * h_s2)
        ) ** (0.5)
        self._h_efs[self.ply[2]] = (
            self.h_efw**3 / (h_2 + 2 * self.Gamma * h_s1)
        ) ** (0.5)

    def _determine_package_E(self):
        """
        Method to determine the elastic modulus for the formulation.
        As the validation function is already checks the elastic modulus
        of all :class:`~structuralglass.layers.GlassPly` are the same, take
        the 1st plys elastic modulus.
        """

        self._E = self.ply[0].E

    @property
    def beta(self):
        """
        The beta value of the formulation.
        A coefficient  dependent  on  the  boundary  and  loading  conditions.

        Returns
        -------
        :class:`float`
        """

        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    @property
    def panel_min_dim(self):
        """
        The minimum dimension of the rectangular panel.

        Returns
        -------
        :class:`~pint.Quantity` [length]

        Raises
        ------
        ValueError
            If the minimum panel dimension is less than 0m.
        """

        return self._panel_min_dim

    @panel_min_dim.setter
    @ureg.check(None, "[length]")
    def panel_min_dim(self, value):
        if value < Q_(0, "ft"):
            raise ValueError(
                "The minimum panel dimension must be greater than 0ft"
            )
        self._panel_min_dim = value
        self._calc_equiv_thickness()
