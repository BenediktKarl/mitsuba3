#include <mitsuba/core/properties.h>
#include <mitsuba/render/bounded_ggx_vndf.h>
#include <mitsuba/python/python.h>
#include <drjit/dynamic.h>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

MI_PY_EXPORT(BoundedGGX) {
    MI_PY_IMPORT_TYPES(BoundedGGX);

    nb::class_<BoundedGGX>(m, "BoundedGGX")
        .def("__init__", [](BoundedGGX* alloc, ScalarFloat alpha, ScalarFloat epsilon) {
            new (alloc) BoundedGGX(alpha, epsilon);
        }, "alpha"_a, "epsilon"_a=1e-3)
    ;
}