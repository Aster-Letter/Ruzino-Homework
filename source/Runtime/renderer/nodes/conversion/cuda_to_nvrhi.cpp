#include "nodes/core/def/node_def.hpp"
#include "render_node_base.h"

#if RUZINO_WITH_CUDA
#include "RHI/internal/cuda_extension.hpp"
#include "hd_RUZINO/render_global_payload.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE

static void node_cuda_to_nvrhi_declare(NodeDeclarationBuilder& b)
{
    b.add_input<cuda::CUDALinearBufferHandle>("Buffer")
        .description("CUDA linear buffer to convert");
    b.add_input<int>("Width").default_val(1920)
        .description("Texture width");
    b.add_input<int>("Height").default_val(1080)
        .description("Texture height");
    b.add_output<nvrhi::TextureHandle>("Texture")
        .description("Output NVRHI texture");
}

static void node_cuda_to_nvrhi_exec(ExeParams params)
{
    auto buffer = params.get_input<cuda::CUDALinearBufferHandle>("Buffer");
    int width = params.get_input<int>("Width");
    int height = params.get_input<int>("Height");
    
    if (!buffer) {
        throw std::runtime_error("Invalid input buffer");
    }
    
    auto& global_payload = params.get_global_payload<RenderGlobalPayload&>();
    auto device = global_payload.nvrhi_device;
    
    // Create texture descriptor
    nvrhi::TextureDesc desc = nvrhi::TextureDesc{}
        .setWidth(width)
        .setHeight(height)
        .setFormat(nvrhi::Format::RGBA32_FLOAT)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setIsUAV(true);
    
    // Convert CUDA buffer to NVRHI texture
    auto texture = cuda::cuda_linear_buffer_to_nvrhi_texture(
        device, buffer, desc);
    
    params.set_output("Texture", texture);
}

static void node_register()
{
    static NodeTypeInfo ntype;

    strcpy(ntype.ui_name, "CUDA to NVRHI");
    strcpy_s(ntype.id_name, "cuda_to_nvrhi");

    geo_node_type_base(&ntype);
    ntype.node_execute = node_cuda_to_nvrhi_exec;
    ntype.declare = node_cuda_to_nvrhi_declare;
    ntype.is_conversion_node = true;

    register_node_type(&ntype);
}

NOD_REGISTER_NODE(node_register)

RUZINO_NAMESPACE_CLOSE_SCOPE

#endif  // RUZINO_WITH_CUDA
