#include "gpu_compute.h"

#include "GPUContext/compute_context.hpp"
#include "GPUContext/program_vars.hpp"
#include "RHI/internal/resources.hpp"
#include "RHI/rhi.hpp"
#include "nvrhi/nvrhi.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/vec3f.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using namespace pxr;

// Define the macro locally if not available
template<typename T>
class RAII_resource_cleaner_local {
   public:
    RAII_resource_cleaner_local(ResourceAllocator& allocator)
        : allocator_(allocator)
    {
    }
    T set_data(T handle)
    {
        data = handle;
        return data;
    }
    ~RAII_resource_cleaner_local()
    {
    }

   private:
    ResourceAllocator& allocator_;
    T data;
};
#define MARK_DESTROY_NVRHI_RESOURCE(resource)                           \
    RAII_resource_cleaner_local<decltype(resource)> resource##_cleaner( \
        get_instance().sa_resource_allocator);                          \
    resource##_cleaner.set_data(resource)

GPUSceneAssember GPUSceneAssember::instance;

void GPUSceneAssember::fill_instances(
    const pxr::GfMatrix4f& parent_transform,
    const pxr::VtIntArray& instance_indices,
    const pxr::GfVec3f* translations,
    const pxr::GfQuatf* rotations,
    const pxr::GfVec3f* scales,
    const pxr::GfMatrix4d* instanceTransforms,
    DeviceMemoryPool<GeometryInstanceData>::MemoryHandle
        geometry_instance_buffer,
    DeviceMemoryPool<nvrhi::rt::InstanceDesc>::MemoryHandle rt_instance_buffer,
    uint64_t BLAS_address,
    const pxr::GfMatrix4f& prototype_transform,
    unsigned material_id,
    unsigned geometry_id)
{
    if (instance_indices.empty()) {
        return;
    }

    auto program_desc =
        ProgramDesc()
            .add_path(GPU_ASSEMBLER_SHADER_DIR "instancer.slang")
            .set_entry_name("main")
            .set_shader_type(nvrhi::ShaderType::Compute);

    if (translations) {
        program_desc.define("HAS_TRANSLATIONS", "1");
    }

    if (rotations) {
        program_desc.define("HAS_ROTATIONS", "1");
    }

    if (scales) {
        program_desc.define("HAS_SCALES", "1");
    }

    if (instanceTransforms) {
        program_desc.define("HAS_INSTANCE_TRANSFORMS", "1");
    }

    ProgramHandle filler_program =
        get_instance().sa_resource_allocator.create(program_desc);
    MARK_DESTROY_NVRHI_RESOURCE(filler_program);

    ProgramVars filler_program_vars(
        get_instance().sa_resource_allocator, filler_program);

    auto device = RHI::get_device();

    // Create index buffer
    nvrhi::BufferDesc index_desc =
        nvrhi::BufferDesc{}
            .setByteSize(instance_indices.size() * sizeof(int))
            .setStructStride(sizeof(int))
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true)
            .setDebugName("instance_indices");
    auto index_buffer = device->createBuffer(index_desc);
    MARK_DESTROY_NVRHI_RESOURCE(index_buffer);

    auto cmd = device->createCommandList();
    cmd->open();
    cmd->writeBuffer(
        index_buffer,
        instance_indices.data(),
        instance_indices.size() * sizeof(int));

    // Create optional buffers
    nvrhi::BufferHandle translations_buffer;
    if (translations) {
        nvrhi::BufferDesc translations_desc =
            nvrhi::BufferDesc{}
                .setByteSize(instance_indices.size() * sizeof(GfVec3f))
                .setStructStride(sizeof(GfVec3f))
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("translations");
        translations_buffer = device->createBuffer(translations_desc);
        MARK_DESTROY_NVRHI_RESOURCE(translations_buffer);
        cmd->writeBuffer(
            translations_buffer,
            translations,
            instance_indices.size() * sizeof(GfVec3f));
    }

    nvrhi::BufferHandle rotations_buffer;
    if (rotations) {
        nvrhi::BufferDesc rotations_desc =
            nvrhi::BufferDesc{}
                .setByteSize(instance_indices.size() * sizeof(GfQuatf))
                .setStructStride(sizeof(GfQuatf))
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("rotations");
        rotations_buffer = device->createBuffer(rotations_desc);
        MARK_DESTROY_NVRHI_RESOURCE(rotations_buffer);
        cmd->writeBuffer(
            rotations_buffer,
            rotations,
            instance_indices.size() * sizeof(GfQuatf));
    }

    nvrhi::BufferHandle scales_buffer;
    if (scales) {
        nvrhi::BufferDesc scales_desc =
            nvrhi::BufferDesc{}
                .setByteSize(instance_indices.size() * sizeof(GfVec3f))
                .setStructStride(sizeof(GfVec3f))
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("scales");
        scales_buffer = device->createBuffer(scales_desc);
        MARK_DESTROY_NVRHI_RESOURCE(scales_buffer);
        cmd->writeBuffer(
            scales_buffer, scales, instance_indices.size() * sizeof(GfVec3f));
    }

    nvrhi::BufferHandle instance_transforms_buffer;
    if (instanceTransforms) {
        nvrhi::BufferDesc instance_transforms_desc =
            nvrhi::BufferDesc{}
                .setByteSize(instance_indices.size() * sizeof(GfMatrix4d))
                .setStructStride(sizeof(GfMatrix4d))
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true)
                .setDebugName("instance_transforms");
        instance_transforms_buffer =
            device->createBuffer(instance_transforms_desc);
        MARK_DESTROY_NVRHI_RESOURCE(instance_transforms_buffer);
        cmd->writeBuffer(
            instance_transforms_buffer,
            instanceTransforms,
            instance_indices.size() * sizeof(GfMatrix4d));
    }

    cmd->close();
    device->executeCommandList(cmd);

    // Set up constant buffer parameters
    struct InstanceParams {
        GfMatrix4f parent_transform;
        GfMatrix4f prototype_transform;
        unsigned material_id;
        unsigned geometry_id;
        unsigned instance_count;
        unsigned base_instance_index;
        uint64_t BLAS_address;
    };

    InstanceParams params;
    params.parent_transform = parent_transform;
    params.prototype_transform = prototype_transform;
    params.material_id = material_id;
    params.geometry_id = geometry_id;
    params.instance_count = instance_indices.size();
    params.base_instance_index = geometry_instance_buffer->index();
    params.BLAS_address = BLAS_address;

    nvrhi::BufferDesc params_desc =
        nvrhi::BufferDesc{}
            .setByteSize(sizeof(InstanceParams))
            .setIsConstantBuffer(true)
            .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
            .setKeepInitialState(true)
            .setDebugName("instance_params");
    auto params_buffer = device->createBuffer(params_desc);
    MARK_DESTROY_NVRHI_RESOURCE(params_buffer);

    cmd = device->createCommandList();
    cmd->open();
    cmd->writeBuffer(params_buffer, &params, sizeof(InstanceParams));
    cmd->close();
    device->executeCommandList(cmd);

    // Set up program variables
    filler_program_vars["g_Params"] = params_buffer;
    filler_program_vars["g_InstanceIndices"] = index_buffer;
    filler_program_vars["g_GeometryInstances"] =
        geometry_instance_buffer->get_descriptor();
    filler_program_vars["g_RTInstances"] = rt_instance_buffer->get_descriptor();

    if (translations) {
        filler_program_vars["g_Translations"] = translations_buffer;
    }
    if (rotations) {
        filler_program_vars["g_Rotations"] = rotations_buffer;
    }
    if (scales) {
        filler_program_vars["g_Scales"] = scales_buffer;
    }
    if (instanceTransforms) {
        filler_program_vars["g_InstanceTransforms"] =
            instance_transforms_buffer;
    }

    filler_program_vars.finish_setting_vars();

    ComputeContext compute_context(
        get_instance().sa_resource_allocator, filler_program_vars);
    compute_context.finish_setting_pso();

    compute_context.begin();
    compute_context.dispatch(
        {}, filler_program_vars, instance_indices.size(), 64);
    compute_context.finish();

    device->waitForIdle();
}

USTC_CG_NAMESPACE_CLOSE_SCOPE