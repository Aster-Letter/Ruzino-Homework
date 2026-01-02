#include <nanobind/nanobind.h>

#include <RHI/rhi.hpp>

namespace nb = nanobind;

using namespace Ruzino;

// Tell nanobind to treat IDevice as an opaque pointer type
NB_MAKE_OPAQUE(nvrhi::IDevice*);

NB_MODULE(RHI_py, m)
{
    // Bind the IDevice interface as an opaque type
    auto idevice = nb::class_<nvrhi::IDevice>(m, "IDevice");
    
    // Bind the GraphicsAPI enum
    nb::enum_<nvrhi::GraphicsAPI>(m, "GraphicsAPI")
        .value("D3D11", nvrhi::GraphicsAPI::D3D11)
        .value("D3D12", nvrhi::GraphicsAPI::D3D12)
        .value("VULKAN", nvrhi::GraphicsAPI::VULKAN);

    // Bind the functions
    m.def("init", &RHI::init, 
          nb::arg("with_window") = false,
          nb::arg("use_dx12") = false);
    m.def("shutdown", &RHI::shutdown);
    m.def("get_device", &RHI::get_device, nb::rv_policy::reference);
    m.def("get_backend", &RHI::get_backend);
}
