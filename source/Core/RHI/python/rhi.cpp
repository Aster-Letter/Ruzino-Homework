//#include <nanobind/nanobind.h>
//
//#include <RHI/rhi.hpp>
//
//namespace nb = nanobind;
//
//using namespace USTC_CG;
//
//NB_MODULE(RHI_py, m)
//{
//    m.def(
//        "init",
//        &RHI::init,
//        nb::arg("with_window") = false,
//        nb::arg("use_dx12") = false);
//    m.def("shutdown", &RHI::shutdown);
//    m.def("get_device", &RHI::get_device, nb::rv_policy::reference);
//    m.def("get_backend", &RHI::get_backend);
//}
