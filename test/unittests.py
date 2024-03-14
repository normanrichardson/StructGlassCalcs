import unittest

import pint

import structuralglass.demands as dem
import structuralglass.equiv_thick_models as et
import structuralglass.glass_types as gt
import structuralglass.helpers as hp
import structuralglass.layers as lay
from structuralglass import Q_, ureg


class TestGlassPly(unittest.TestCase):
    def test_from_nominal_thickness_metric(self):
        tnom = 8 * ureg.mm
        tmin = lay.t_min_lookup_metric[tnom.m_as("mm")] * ureg.mm
        ply = lay.GlassPly.from_nominal_thickness(tnom)
        self.assertIsInstance(ply, lay.GlassPly)
        self.assertEqual(ply.t_nom, tnom)
        self.assertEqual(ply.t_min, tmin)
        self.assertEqual(ply.E, 71.7 * ureg.GPa)

    def test_from_nominal_thickness_imperial(self):
        tnom = 3 / 8 * ureg.inch
        tmin = lay.t_min_lookup_imperial[tnom.m_as("in")] * ureg.mm
        ply = lay.GlassPly.from_nominal_thickness(tnom)
        self.assertIsInstance(ply, lay.GlassPly)
        self.assertEqual(ply.t_nom, tnom)
        self.assertEqual(ply.t_min, tmin)
        self.assertEqual(ply.E, 71.7 * ureg.GPa)

    def test_from_actual_thickness(self):
        tmin = 8 * ureg.mm
        ply = lay.GlassPly.from_actual_thickness(tmin)
        self.assertIsInstance(ply, lay.GlassPly)
        self.assertIsNone(ply.t_nom)
        self.assertEqual(ply.t_min, tmin)
        self.assertEqual(ply.E, 71.7 * ureg.GPa)

    def test_invalid_lookup_from_nominal_thickness(self):
        tnom = 8.5 * ureg.mm
        with self.assertRaises(ValueError) as cm:
            lay.GlassPly.from_nominal_thickness(tnom)
        self.assertEqual(
            str(cm.exception),
            f"Could not find the nominal thickness of {tnom} in "
            "the nominal thickness lookup.",
        )

    def test_invalid_no_unit_from_nominal_thickness(self):
        tnom = 8
        with self.assertRaises(pint.DimensionalityError) as cm:
            lay.GlassPly.from_nominal_thickness(tnom)
        self.assertEqual(
            str(cm.exception),
            "Cannot convert from '8' (dimensionless) to 'a "
            "quantity of' ([length])",
        )

    def test_invalid_unit_from_nominal_thickness(self):
        tnom = 8 * ureg.mm ** 2
        with self.assertRaises(pint.DimensionalityError) as cm:
            lay.GlassPly.from_nominal_thickness(tnom)
        self.assertEqual(
            str(cm.exception),
            "Cannot convert from '8 millimeter ** 2' ([length] **"
            " 2) to 'a quantity of' ([length])",
        )


class TestInterlayerStatic(unittest.TestCase):
    def setUp(self) -> None:
        self.G_pvb = 0.281 * ureg.MPa
        self.t_pvb = 0.89 * ureg.mm
        self.interlayer = lay.Interlayer.from_static(self.t_pvb, self.G_pvb)
        lay.Interlayer

    def test_data(self):
        self.assertIsInstance(self.interlayer, lay.Interlayer)
        self.assertEqual(self.interlayer.t, self.t_pvb)
        self.assertEqual(self.interlayer.G, self.G_pvb)

    def test_invalid_request_for_duration(self):
        with self.assertRaises(ValueError) as cm:
            self.interlayer.duration
        self.assertEqual(
            str(cm.exception),
            "No product table provided. Static case being used.",
        )

    def test_invalid_request_for_temperature(self):
        with self.assertRaises(ValueError) as cm:
            self.interlayer.temperature
        self.assertEqual(
            str(cm.exception),
            "No product table provided. Static case being used.",
        )


class TestInterlayerProductRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # save the registry before testing
        cls._registry = lay._interLayer_registry

    @classmethod
    def tearDownClass(cls):
        # reset the registry after testing
        lay._interLayer_registry = cls._registry

    def test_valid_product_data(self):
        name = "nice product"
        data = {
            (Q_(20, "degC"), Q_(3, "sec")): Q_(240, "MPa"),
            (Q_(30, "degC"), Q_(3, "sec")): Q_(217, "MPa"),
            (Q_(40, "degC"), Q_(3, "sec")): Q_(151, "MPa"),
            (Q_(20, "degC"), Q_(10, "min")): Q_(77.0, "MPa"),
            (Q_(30, "degC"), Q_(10, "min")): Q_(36.2, "MPa"),
            (Q_(40, "degC"), Q_(10, "min")): Q_(11.8, "MPa"),
        }
        lay.register_interlayer_product(name, data)
        self.assertDictEqual(lay._interLayer_registry[name], data)

    def test_invalid_product_data(self):
        name = "not so nice product"
        invalid_data = {
            (Q_(20, "degC"), Q_(3, "sec")): Q_(240, "MPa"),
            (Q_(30, "degC"), Q_(3, "sec")): Q_(217, "MPa"),
            (Q_(40, "degC"), Q_(3, "sec")): Q_(151, "MPa"),
            (Q_(30, "degC"), Q_(10, "min")): Q_(36.2, "MPa"),
            (Q_(40, "degC"), Q_(10, "min")): Q_(11.8, "MPa"),
        }
        with self.assertRaises(ValueError) as cm:
            lay.register_interlayer_product(name, invalid_data)
        self.assertEqual(
            str(cm.exception), "The provided data is not rectangular."
        )


class TestInterlayerProductTable(unittest.TestCase):
    def setUp(self):
        self.t_pvb = 1.52 * ureg.mm
        product_name = "Ionoplast Interlayer NCSEA"
        self.interlayer = lay.Interlayer.from_product_table(
            self.t_pvb, product_name
        )

    def test_data(self):
        self.interlayer.temperature = Q_(40, "degC")
        self.interlayer.duration = Q_(1, "month")
        self.assertEqual(self.interlayer.t, self.t_pvb)
        self.assertEqual(self.interlayer.G, Q_(3.29, "MPa"))

    def test_interpolation_data(self):
        self.interlayer.temperature = Q_(35, "degC")
        self.interlayer.duration = Q_(1, "day")
        self.assertEqual(self.interlayer.t, self.t_pvb)
        self.assertEqual(self.interlayer.G, Q_(26.5, "MPa"))

    def test_interpolation_out_of_bounds(self):
        self.interlayer.temperature = Q_(35, "degC")
        self.interlayer.duration = Q_(20, "year")
        with self.assertRaises(ValueError) as cm:
            G = self.interlayer.G
        self.assertEqual(
            str(cm.exception), ("Extrapolating G: The temperature-duration "
            "combination is outside of the data tables bounds.")
        )

    def test_not_setting_ref_temp(self):
        self.interlayer.duration = Q_(1, "month")
        self.assertEqual(self.interlayer.t, self.t_pvb)
        with self.assertRaises(ValueError) as cm:
            self.interlayer.G
        self.assertEqual(
            str(cm.exception),
            "Reference temperature and/or duration not set.",
        )

    def test_not_setting_ref_duration(self):
        self.interlayer.temperature = Q_(40, "degC")
        self.assertEqual(self.interlayer.t, self.t_pvb)
        with self.assertRaises(ValueError) as cm:
            self.interlayer.G
        self.assertEqual(
            str(cm.exception),
            "Reference temperature and/or duration not set.",
        )


class TestInvalidInterlayerClassMethods(unittest.TestCase):
    def test_invalid_product_name(self):
        t_pvb = 1.52 * ureg.mm
        product_name = "product 1"
        with self.assertRaises(ValueError) as cm:
            lay.Interlayer.from_product_table(t_pvb, product_name)
        self.assertEqual(
            str(cm.exception),
            "The product is not registered in the product " "registry.",
        )

    def test_invalid_product_thickness(self):
        t_pvb = -1.52 * ureg.mm
        product_name = "PVB NCSEA"
        with self.assertRaises(ValueError) as cm:
            lay.Interlayer.from_product_table(t_pvb, product_name)
        self.assertEqual(
            str(cm.exception),
            "The thickness must be greater than zero [lengh].",
        )

    def test_invalid_static_shear_mod(self):
        t_pvb = 1.52 * ureg.mm
        G_pvb = -0.281 * ureg.MPa
        with self.assertRaises(ValueError) as cm:
            lay.Interlayer.from_static(t_pvb, G_pvb)
        self.assertEqual(
            str(cm.exception),
            "The shear modulus must be greater than zero " "[pressure].",
        )

    def test_invalid_static_thickness(self):
        t_pvb = -1.52 * ureg.mm
        G_pvb = 0.281 * ureg.MPa
        with self.assertRaises(ValueError) as cm:
            lay.Interlayer.from_static(t_pvb, G_pvb)
        self.assertEqual(
            str(cm.exception),
            "The thickness must be greater than zero [lengh].",
        )


class TestMonolithicMethod(unittest.TestCase):
    def setUp(self):
        self.t1nom = 8 * ureg.mm
        self.t2nom = 10 * ureg.mm
        self.ply1 = lay.GlassPly.from_nominal_thickness(self.t1nom)
        self.ply2 = lay.GlassPly.from_nominal_thickness(self.t2nom)
        package = [self.ply1, self.ply2]
        self.buildup = [et.MonolithicMethod(package)]

    def test_h_efw(self):
        t_act = (
            lay.t_min_lookup_metric[self.t1nom.m_as("mm")]
            + lay.t_min_lookup_metric[self.t2nom.m_as("mm")]
        ) * ureg.mm
        self.assertAlmostEqual(self.buildup[0].h_efw, t_act)

    def test_h_efs(self):
        t_act = (
            lay.t_min_lookup_metric[self.t1nom.m_as("mm")]
            + lay.t_min_lookup_metric[self.t2nom.m_as("mm")]
        ) * ureg.mm
        self.assertAlmostEqual(self.buildup[0].h_efs[self.ply1], t_act)


class TestNonCompositeMethod(unittest.TestCase):
    def setUp(self):
        self.t1nom = 8 * ureg.mm
        self.t2nom = 6 * ureg.mm
        self.t3nom = 8 * ureg.mm
        self.ply1 = lay.GlassPly.from_nominal_thickness(self.t1nom)
        self.ply2 = lay.GlassPly.from_nominal_thickness(self.t2nom)
        self.ply3 = lay.GlassPly.from_nominal_thickness(self.t2nom)
        package = [self.ply1, self.ply2, self.ply3]
        self.buildup = [et.NonCompositeMethod(package)]

    def test_h_efw(self):
        t_act = 9.09479121 * ureg.mm
        self.assertAlmostEqual(self.buildup[0].h_efw, t_act)

    def test_h_efs(self):
        t_act = 10.8112719 * ureg.mm
        self.assertAlmostEqual(self.buildup[0].h_efs[self.ply3], t_act)


class TestShearTransferCoefMethod(unittest.TestCase):
    def setUp(self):
        self.a = 1000 * ureg.mm
        self.t1nom = 8 * ureg.mm
        self.t2nom = 6 * ureg.mm
        G_pvb = 0.44 * ureg.MPa
        t_pvb = 1.52 * ureg.mm
        self.ply1 = lay.GlassPly.from_nominal_thickness(self.t1nom)
        self.ply2 = lay.GlassPly.from_nominal_thickness(self.t2nom)
        self.interlayer = lay.Interlayer.from_static(t_pvb, G_pvb)
        package = [self.ply1, self.interlayer, self.ply2]
        self.buildup = [et.ShearTransferCoefMethod(package, self.a)]

    def test_h_efw(self):
        t_act = 9.53304352 * ureg.mm
        self.assertAlmostEqual(self.buildup[0].h_efw, t_act)

    def test_h_efs(self):
        t_act1 = 10.2650672 * ureg.mm
        t_act2 = 11.4310515 * ureg.mm
        self.assertAlmostEqual(self.buildup[0].h_efs[self.ply1], t_act1)
        self.assertAlmostEqual(self.buildup[0].h_efs[self.ply2], t_act2)

    def test_invalid_packages(self):
        # invalid package
        ply3 = lay.GlassPly.from_nominal_thickness(self.t1nom)
        pac_invalid_1 = [self.ply1, self.ply2]
        pac_invalid_2 = [self.ply1, self.ply2, self.interlayer]
        pac_invalid_3 = [
            self.ply1,
            self.interlayer,
            self.ply2,
            self.interlayer,
            ply3,
        ]

        with self.assertRaises(ValueError) as cm:
            [et.ShearTransferCoefMethod(pac_invalid_1, self.a)]
        self.assertEqual(
            str(cm.exception),
            "Ply validation failed: Method is only valid a list "
            "of [GlassPly, Interlayer, GlassPly].",
        )

        with self.assertRaises(ValueError) as cm:
            [et.ShearTransferCoefMethod(pac_invalid_2, self.a)]
        self.assertEqual(
            str(cm.exception),
            "Ply validation failed: Method is only valid a list "
            "of [GlassPly, Interlayer, GlassPly].",
        )

        with self.assertRaises(ValueError) as cm:
            [et.ShearTransferCoefMethod(pac_invalid_3, self.a)]
        self.assertEqual(
            str(cm.exception),
            "Ply validation failed: Method is only valid a list "
            "of [GlassPly, Interlayer, GlassPly].",
        )


class TestGlassTypeRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # save the registry before testing
        cls._registry = gt._glass_type_registry

    @classmethod
    def tearDownClass(cls):
        # reset the registry after testing
        gt._glass_type_registry = cls._registry

    def test_valid_glass_type_add(self):
        data = {
            "stress_surface": 93.1 * ureg.MPa,
            "stress_edge": 73.0 * ureg.MPa,
            "duration_factor": 47.5,
            "coef_variation": 0.2,
            "surf_factors": {
                "None": 1,
                "Fritted": 1,
                "Acid etching": 0.5,
                "Sandblasting": 0.5,
            },
        }
        gt.register_glass_type(name="Fully Tempered 2", **data, abbr="FT2")
        self.assertDictEqual(gt._glass_type_registry["Fully Tempered 2"], data)

    def test_invalid_glass_type_add_1(self):
        data = {
            "stress_surface": 93.1 * ureg.MPa,
            "stress_edge": 73.0 * ureg.MPa,
            "duration_factor": 47.5,
            "coef_variation": 0.2,
            "surf_factors": {
                "None": 1,
                "Fritted": 1,
                "Acid etching": 0.5,
                "Sandblasting": 0.5,
            },
        }

        with self.assertRaises(ValueError) as cm:
            gt.register_glass_type(name="Fully Tempered", **data, abbr="FT")
        self.assertEqual(
            str(cm.exception),
            "Name identifier already in use. Deregister `Fully Tempered` "
            "first.",
        )

    def test_invalid_glass_type_add_2(self):
        data = {
            "stress_surface": 93.1 * ureg.MPa,
            "stress_edge": 73.0 * ureg.MPa,
            "duration_factor": 47.5,
            "coef_variation": 0.2,
            "surf_factors": {
                "None": 1,
                "Fritted": 1,
                "Acid etching": 0.5,
                "Sandblasting": 0.5,
            },
        }

        with self.assertRaises(ValueError) as cm:
            gt.register_glass_type(name="Fully Tempered 2", **data, abbr="FT")
        self.assertEqual(
            str(cm.exception),
            "Abbreviation identifier already in use. Deregister "
            "`Fully Tempered` first.",
        )

    def test_remove_glass_type(self):
        name = "Annealed"
        abbr = "AN"
        gt.deregister_glass_type(name)
        with self.assertRaises(KeyError) as cm:
            gt._glass_type_registry[name]
        self.assertEqual(str(cm.exception), "'Annealed'")
        with self.assertRaises(KeyError) as cm:
            gt._glass_type_abbr[abbr]
        self.assertEqual(str(cm.exception), "'AN'")


class TestAnnealedGlassType(unittest.TestCase):
    def setUp(self) -> None:
        self.an = gt.GlassType.from_abbr("AN")

    def test_prob_breakage_factor(self):
        self.assertAlmostEqual(
            self.an.prob_breakage_factor(1 / 1000), 0.681114447, 4
        )
        self.assertAlmostEqual(
            self.an.prob_breakage_factor(5 / 1000), 0.9218781646, 4
        )
        self.assertAlmostEqual(self.an.prob_breakage_factor(8 / 1000), 1, 4)
        self.assertAlmostEqual(
            self.an.prob_breakage_factor(10 / 1000), 1.038646695, 4
        )

    def test_load_duration_factor(self):
        self.assertAlmostEqual(
            self.an.load_duration_factor(Q_(10, "year")), 0.315, 3
        )
        self.assertAlmostEqual(
            self.an.load_duration_factor(Q_(12, "hour")), 0.550, 3
        )

    def test_surf_factors(self):
        self.assertAlmostEqual(self.an.surf_factors["None"], 1)


class TestHeatStrengthenedGlassType(unittest.TestCase):
    def setUp(self) -> None:
        self.hs = gt.GlassType.from_abbr("HS")

    def test_prob_breakage_factor(self):
        self.assertAlmostEqual(
            self.hs.prob_breakage_factor(1 / 1000), 0.8399834341, 4
        )
        self.assertAlmostEqual(
            self.hs.prob_breakage_factor(3 / 1000), 0.9204133016, 4
        )
        self.assertAlmostEqual(self.hs.prob_breakage_factor(8 / 1000), 1, 4)
        self.assertAlmostEqual(
            self.hs.prob_breakage_factor(9 / 1000), 1.010169699, 4
        )

    def test_load_duration_factor(self):
        self.assertAlmostEqual(
            self.hs.load_duration_factor(10 * ureg.year), 0.558, 3
        )
        self.assertAlmostEqual(
            self.hs.load_duration_factor(12 * ureg.hour), 0.739, 3
        )

    def test_surf_factors(self):
        self.assertAlmostEqual(self.hs.surf_factors["Fritted"], 1)


class TestFullyTemperedGlassType(unittest.TestCase):
    def setUp(self) -> None:
        self.ft = gt.GlassType.from_abbr("FT")

    def test_prob_breakage_factor(self):
        self.assertAlmostEqual(
            self.ft.prob_breakage_factor(1 / 1000), 0.9102486076, 4
        )
        self.assertAlmostEqual(
            self.ft.prob_breakage_factor(4 / 1000), 0.9679689847, 4
        )
        self.assertAlmostEqual(self.ft.prob_breakage_factor(8 / 1000), 1, 4)
        self.assertAlmostEqual(
            self.ft.prob_breakage_factor(10 / 1000), 1.01087724, 4
        )

    def test_prob_breakage_factor_in_NCSEA(self):
        # NCSEA uses a coef of var of 0.2 irrespective of the glass type
        self.ft.coef_variation = 0.2
        self.assertAlmostEqual(
            self.ft.prob_breakage_factor(1 / 1000), 0.7370555907, 4
        )
        self.assertAlmostEqual(
            self.ft.prob_breakage_factor(4 / 1000), 0.9061588218, 4
        )
        self.assertAlmostEqual(self.ft.prob_breakage_factor(8 / 1000), 1, 4)
        self.assertAlmostEqual(
            self.ft.prob_breakage_factor(10 / 1000), 1.031867021, 4
        )

    def test_load_duration_factor(self):
        self.assertAlmostEqual(
            self.ft.load_duration_factor(10 * ureg.year), 0.678, 3
        )
        self.assertAlmostEqual(
            self.ft.load_duration_factor(12 * ureg.hour), 0.817, 3
        )

    def test_surf_factors(self):
        self.assertAlmostEqual(self.ft.surf_factors["Acid etching"], 0.5)


class TestRoarks4sidePlate(unittest.TestCase):
    def setUp(self):
        self.rk4s = hp.Roarks4side(
            Q_(71.7, "GPa"), Q_(5, "ft"), Q_(10, "ft"), Q_(0.5, "inch")
        )

    def test_data(self):
        self.assertEqual(self.rk4s.dim_x, Q_(5, "ft"))
        self.assertEqual(self.rk4s.dim_y, Q_(10, "ft"))

    def test_stress_max(self):
        self.assertAlmostEqual(
            self.rk4s.stress_max(Q_(20, "psf")), Q_(8.41436178451202, "MPa"), 3
        )

    def test_deflection_max(self):
        self.assertAlmostEqual(
            self.rk4s.deflection_max(Q_(20, "psf")), Q_(-0.153704045, "in")
        )

    def test_reaction_max(self):
        self.assertAlmostEqual(
            self.rk4s.reaction_max(Q_(20, "psf")), Q_(50.3, "plf")
        )

    def test_change_properties(self):
        self.rk4s.dim_x = Q_(15, "ft")
        self.rk4s.dim_y = Q_(5, "ft")
        self.rk4s.t = Q_(0.25, "in")
        self.rk4s.E = Q_(200, "GPa")
        self.assertAlmostEqual(
            self.rk4s.stress_max(Q_(20, "psf")), Q_(39.349758824, "MPa"), 0
        )


class TestRoarks4sidePlateInvalid(unittest.TestCase):
    def test_invalid_dimension_1(self):
        with self.assertRaises(ValueError) as cm:
            hp.Roarks4side(
                Q_(71.7, "GPa"), Q_(-5, "ft"), Q_(10, "ft"), Q_(0.5, "inch")
            )
        self.assertEqual(
            str(cm.exception), "Dimensions must be greater than zero."
        )

    def test_invalid_dimension_2(self):
        with self.assertRaises(ValueError) as cm:
            hp.Roarks4side(
                Q_(71.7, "GPa"),
                Q_(5, "ft"),
                Q_(-10, "ft"),
                Q_(0.5, "inch"),
            )
        self.assertEqual(
            str(cm.exception), "Dimensions must be greater than zero."
        )

    def test_invalid_thickness(self):
        with self.assertRaises(ValueError) as cm:
            hp.Roarks4side(
                Q_(71.7, "GPa"), Q_(5, "ft"), Q_(10, "ft"), Q_(-0.5, "inch")
            )
        self.assertEqual(
            str(cm.exception), "Thickness must be greater than zero."
        )

    def test_invalid_elastic_mod(self):
        with self.assertRaises(ValueError) as cm:
            hp.Roarks4side(
                Q_(-71.7, "GPa"), Q_(5, "ft"), Q_(10, "ft"), Q_(0.5, "inch")
            )
        self.assertEqual(
            str(cm.exception), "Elastic modulus must be greater than zero."
        )

    def test_invalid_dimension_prop_1(self):
        rk4s = hp.Roarks4side(
            Q_(71.7, "GPa"), Q_(5, "ft"), Q_(10, "ft"), Q_(0.5, "inch")
        )
        with self.assertRaises(ValueError) as cm:
            rk4s.dim_x = Q_(-10, "ft")
        self.assertEqual(
            str(cm.exception), "Dimensions must be greater than zero."
        )

    def test_invalid_dimension_prop_2(self):
        rk4s = hp.Roarks4side(
            Q_(71.7, "GPa"), Q_(5, "ft"), Q_(10, "ft"), Q_(0.5, "inch")
        )
        with self.assertRaises(ValueError) as cm:
            rk4s.dim_y = Q_(-5, "ft")
        self.assertEqual(
            str(cm.exception), "Dimensions must be greater than zero."
        )

    def test_invalid_thickness_prop(self):
        rk4s = hp.Roarks4side(
            Q_(71.7, "GPa"), Q_(5, "ft"), Q_(10, "ft"), Q_(0.5, "inch")
        )
        with self.assertRaises(ValueError) as cm:
            rk4s.t = Q_(-0.75, "inch")
        self.assertEqual(
            str(cm.exception), "Thickness must be greater than zero."
        )

    def test_invalid_elastic_mod_prop(self):
        rk4s = hp.Roarks4side(
            Q_(71.7, "GPa"),
            Q_(5, "ft"),
            Q_(10, "ft"),
            Q_(0.5, "inch"),
        )
        with self.assertRaises(ValueError) as cm:
            rk4s.E = Q_(-200, "GPa")
        self.assertEqual(
            str(cm.exception), "Elastic modulus must be greater than zero."
        )


class TestIGUWindDemands(unittest.TestCase):
    def setUp(self):
        self.dim_x = Q_(4000, "mm")
        self.dim_y = Q_(2000, "mm")
        t1nom = Q_(6, "mm")
        t2nom = Q_(6, "mm")
        t3nom = Q_(6, "mm")

        # Panel force
        self.wind_load = Q_(1.436, "kPa")

        # Plys
        self.ply1 = lay.GlassPly.from_nominal_thickness(t1nom)
        self.ply2 = lay.GlassPly.from_nominal_thickness(t2nom)
        self.ply3 = lay.GlassPly.from_nominal_thickness(t3nom)

        # Package specifying the model type
        self.package1 = et.MonolithicMethod([self.ply1, self.ply2])
        self.package2 = et.MonolithicMethod([self.ply3])
        buildup = [self.package1, self.package2]

        # Panel
        self.panel = dem.IGUWindDemands(
            buildup, self.wind_load, self.dim_x, self.dim_y
        )
        self.panel.solve()

    def test_data(self):
        self.panel.dim_x
        self.assertListEqual(
            self.panel.buildup, [self.package1, self.package2]
        )
        self.assertEqual(self.panel.dim_x, self.dim_x)
        self.assertEqual(self.panel.dim_y, self.dim_y)
        self.assertEqual(self.panel.wind_load, self.wind_load)

    def test_load_sharing(self):
        self.assertEqual(sum(self.panel.LSF.values()), 1)
        self.assertAlmostEqual(self.panel.LSF[self.package1], 0.888888889)
        self.assertAlmostEqual(self.panel.LSF[self.package2], 0.111111111)

    def test_stress_demand(self):
        self.assertAlmostEqual(
            self.panel.stress[self.ply1], Q_(25.1955903, "MPa"), 4
        )
        self.assertAlmostEqual(
            self.panel.stress[self.ply2], Q_(25.1955903, "MPa"), 4
        )
        self.assertAlmostEqual(
            self.panel.stress[self.ply3], Q_(12.5977951, "MPa"), 4
        )

    def test_deflection_demand(self):
        self.assertAlmostEqual(
            self.panel.deflection[self.package1], Q_(-22.9938324, "mm"), 4
        )
        self.assertAlmostEqual(
            self.panel.deflection[self.package2], Q_(-22.9938324, "mm"), 4
        )


if __name__ == "__main__":
    unittest.main()
