#include <drjit/dynamic.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/python/python.h>
#include <mitsuba/render/bounded_ggx_vndf.h>

#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

MI_PY_EXPORT(BoundedGGX) {
    MI_PY_IMPORT_TYPES(BoundedGGX);

    nb::class_<BoundedGGX>(m, "BoundedGGX")
        .def(
            "__init__",
            [](BoundedGGX *alloc, ScalarFloat alpha, bool bounded_ggx,
               bool relative_warp, bool z_square, ScalarFloat epsilon) {
                new (alloc)
                    BoundedGGX(alpha, bounded_ggx, relative_warp, z_square, epsilon);
            },
            "alpha"_a, "bounded_ggx"_a = true, "relative_warp"_a = true,
            "z_square"_a = false, "epsilon"_a = 1e-3)
        .def(nb::init<float, bool, bool, bool, float>(), "alpha"_a,
             "bounded_ggx"_a = true, "relative_warp"_a = true,
             "z_square"_a = false, "epsilon"_a = 1e-3)
        .def("sample", &BoundedGGX::sample, "wi"_a, "sample_phi"_a,
             "sample_theta"_a)
        .def("kiz_root", &BoundedGGX::kiz_root, "wi"_a)
        .def("pdf", &BoundedGGX::pdf, "wi"_a, "wo"_a)
        .def("pdf_m", &BoundedGGX::pdf_m, "wi"_a, "m"_a)
        .def("invert", &BoundedGGX::invert, "wi"_a, "m"_a)
        .def("lambda_", &BoundedGGX::lambda, "theta"_a)
        .def("sigma", &BoundedGGX::sigma, "theta"_a)
        .def("smith_g", &BoundedGGX::smith_g, "wi"_a, "wo"_a, "m"_a)
        .def("elevation", &BoundedGGX::elevation, "d"_a)
        .def("smith_g1", &BoundedGGX::smith_g1, "wo"_a, "m"_a)
        .def("sigma_inv", &BoundedGGX::sigma_inv, "sigma"_a)
        .def("ndf", &BoundedGGX::ndf, "m"_a)
        .def("ndf_supplementary", &BoundedGGX::ndf_supplementary, "m"_a)
        .def("alpha", &BoundedGGX::alpha)
        .def("epsilon", &BoundedGGX::epsilon)
        .def("warp_microfacet", &BoundedGGX::warp_microfacet, "wi"_a, "m"_a)
        .def("unwarp_microfacet", &BoundedGGX::unwarp_microfacet, "wi"_a, "m"_a)
        .def("warp_microfacet_absolute", &BoundedGGX::warp_microfacet_absolute,
             "m"_a)
        .def("unwarp_microfacet_absolute",
             &BoundedGGX::unwarp_microfacet_absolute, "m"_a)
        .def("warp_microfacet_relative", &BoundedGGX::warp_microfacet_relative,
             "wi"_a, "m"_a)
        .def("unwarp_microfacet_relative",
             &BoundedGGX::unwarp_microfacet_relative, "wi"_a, "m"_a)
        .def("spherical_to_cartesian", &BoundedGGX::spherical_to_cartesian,
             "spherical"_a)
        .def("cartesian_to_spherical", &BoundedGGX::cartesian_to_spherical,
             "cartesian"_a)
        .def("theta_jacobian", &BoundedGGX::theta_jacobian, "wi"_a, "m"_a,
             "m_prime"_a)
        .def("theta_jacobian_absolute", &BoundedGGX::theta_jacobian_absolute,
             "m"_a, "m_prime"_a)
        .def("theta_jacobian_relative", &BoundedGGX::theta_jacobian_relative,
             "wi"_a, "m"_a, "m_prime"_a)
        .def("sample_unbounded", &BoundedGGX::sample_unbounded, "wi"_a, "u1"_a,
             "u2"_a)
        .def("invert_sample_unbounded", &BoundedGGX::invert_unbounded, "wi"_a,
             "m"_a)
        .def("theta_max", &BoundedGGX::theta_max, "wi"_a)
        .def("square_redistribution", &BoundedGGX::square_redistribution, "x"_a,
             "low"_a, "high"_a)
        .def("root_redistribution", &BoundedGGX::root_redistribution, "x"_a,
             "low"_a, "high"_a)
        .def_repr(BoundedGGX);
}