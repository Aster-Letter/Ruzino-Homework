#pragma once
#include "RHI/ResourceManager/resource_allocator.hpp"
#include "api.h"
#include "nvrhi/nvrhi.h"
#include "program_vars.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE

class GPUCONTEXT_API GPUContext {
   public:
    virtual ~GPUContext();

    GPUContext(ResourceAllocator& r, ProgramVars& vars);

    virtual void begin();
    virtual void finish();

    void set_resource_state(
        nvrhi::IResource* resource,
        nvrhi::ResourceStates state);

    void uav_barrier(nvrhi::ITexture* texture) const;
    void uav_barrier(nvrhi::IBuffer* buffer) const;

    void write_buffer(
        nvrhi::IBuffer* buffer,
        const void* data,
        size_t dataSize,
        uint64_t destOffsetBytes = 0) const;

    void clear_buffer(
        nvrhi::IBuffer* buffer,
        uint32_t clear_value = 0,
        const nvrhi::BufferRange& range = nvrhi::EntireBuffer);

    void clear_texture(
        nvrhi::ITexture* texture,
        nvrhi::Color color = { 0 },
        const nvrhi::TextureSubresourceSet& subresources =
            nvrhi::AllSubresources);

   protected:
    ResourceAllocator& resource_allocator_;
    ProgramVars& vars_;
    nvrhi::CommandListHandle commandList_;
};

RUZINO_NAMESPACE_CLOSE_SCOPE
