#include "nodes/core/def/node_def.hpp"
#include "render_node_base.h"

#if USTC_CG_WITH_CUDA
#include "RHI/internal/cuda_extension.hpp"
#include "hd_USTC_CG/render_global_payload.hpp"

USTC_CG_NAMESPACE_OPEN_SCOPE

static void node_nvrhi_to_cuda_declare(NodeDeclarationBuilder& b)
{
    b.add_input<nvrhi::TextureHandle>("Texture")
        .description("NVRHI texture to convert to CUDA memory");
    b.add_output<cuda::CUDALinearBufferHandle>("Buffer")
        .description("Output CUDA linear buffer");
}

static void node_nvrhi_to_cuda_exec(ExeParams params)
{
    auto texture = params.get_input<nvrhi::TextureHandle>("Texture");
    
    if (!texture) {
        throw std::runtime_error("Invalid input texture");
    }
    
    auto& global_payload = params.get_global_payload<RenderGlobalPayload&>();
    auto device = global_payload.nvrhi_device;
    
    const auto& desc = texture->getDesc();
    
    // Determine element size based on format
    uint32_t element_size = 0;
    switch (desc.format) {
        case nvrhi::Format::RGBA32_FLOAT:
        case nvrhi::Format::RGBA32_UINT:
        case nvrhi::Format::RGBA32_SINT:
            element_size = 16;
            break;
        case nvrhi::Format::RGB32_FLOAT:
        case nvrhi::Format::RGB32_UINT:
        case nvrhi::Format::RGB32_SINT:
            element_size = 12;
            break;
        case nvrhi::Format::RG32_FLOAT:
        case nvrhi::Format::RG32_UINT:
        case nvrhi::Format::RG32_SINT:
            element_size = 8;
            break;
        case nvrhi::Format::R32_FLOAT:
        case nvrhi::Format::R32_UINT:
        case nvrhi::Format::R32_SINT:
            element_size = 4;
            break;
        default:
            element_size = 16; // Default to RGBA32
            break;
    }
    
    // Convert texture to CUDA linear buffer
    auto buffer = cuda::copy_texture_to_linear_buffer_with_cleanup(
        device, texture.Get(), element_size);
    
    params.set_output("Buffer", buffer);
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "NVRHI to CUDA");
    strcpy_s(ntype.id_name, "nvrhi_to_cuda");

    geo_node_type_base(&ntype);
    ntype.node_execute = node_nvrhi_to_cuda_exec;
    ntype.declare = node_nvrhi_to_cuda_declare;
    ntype.is_conversion_node = true;

    register_node_type(&ntype);
}

NOD_REGISTER_NODE(node_register)

USTC_CG_NAMESPACE_CLOSE_SCOPE

#endif  // USTC_CG_WITH_CUDA
