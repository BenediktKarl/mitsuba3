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
            [](BoundedGGX *alloc, ScalarFloat alpha, ScalarFloat epsilon) {
                new (alloc) BoundedGGX(alpha, epsilon);
            },
            "alpha"_a, "epsilon"_a = 1e-3)
        .def(nb::init<float, float>(), "alpha"_a, "epsilon"_a = 1e-3)
        .def("sample", &BoundedGGX::sample, "wi"_a, "sample_phi"_a,
             "sample_theta"_a)
        .def("kiz_root", &BoundedGGX::kiz_root, "wi"_a)
        .def("pdf", &BoundedGGX::pdf, "wi"_a, "wo"_a)
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
        .def("warp_microfacet", &BoundedGGX::warp_microfacet, "m"_a)
        .def("unwarp_microfacet", &BoundedGGX::unwarp_microfacet, "m"_a)
        .def("spherical_to_cartesian", &BoundedGGX::spherical_to_cartesian,
             "spherical"_a)
        .def("cartesian_to_spherical", &BoundedGGX::cartesian_to_spherical,
             "cartesian"_a)
        .def_repr(BoundedGGX);
}