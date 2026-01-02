

#include "GPUContext/context.hpp"

#include "RHI/internal/nvrhi_patch.hpp"
#include "nvrhi/utils.h"

RUZINO_NAMESPACE_OPEN_SCOPE
GPUContext::~GPUContext()
{
    resource_allocator_.destroy(commandList_);
}

GPUContext::GPUContext(ResourceAllocator& resource_allocator, ProgramVars& vars)
    : resource_allocator_(resource_allocator),
      vars_(vars)
{
    commandList_ = resource_allocator_.create(nvrhi::CommandListDesc{});
}

void GPUContext::begin()
{
    commandList_->open();
}

void GPUContext::finish()
{
    commandList_->close();
    resource_allocator_.device->executeCommandList(commandList_);
}

void GPUContext::set_resource_state(
    nvrhi::IResource* resource,
    nvrhi::ResourceStates state)
{
    auto* texture = dynamic_cast<nvrhi::ITexture*>(resource);
    if (texture) {
        commandList_->setTextureState(
            texture, nvrhi::TextureSubresourceSet(), state);
        return;
    }

    auto* buffer = dynamic_cast<nvrhi::IBuffer*>(resource);
    if (buffer) {
        commandList_->setBufferState(buffer, state);
        return;
    }

    auto* accelStruct = dynamic_cast<nvrhi::rt::IAccelStruct*>(resource);
    if (accelStruct) {
        commandList_->setAccelStructState(accelStruct, state);
        return;
    }
    commandList_->commitBarriers();
    // If we get here, it's an unknown resource type
    assert(false && "Unknown resource type in set_resource_state");
}

void GPUContext::uav_barrier(nvrhi::ITexture* texture) const
{
    nvrhi::utils::TextureUavBarrier(commandList_, texture);
}

void GPUContext::uav_barrier(nvrhi::IBuffer* buffer) const
{
    nvrhi::utils::BufferUavBarrier(commandList_, buffer);
}

void GPUContext::write_buffer(
    nvrhi::IBuffer* buffer,
    const void* data,
    size_t dataSize,
    uint64_t destOffsetBytes) const
{
    return commandList_->writeBuffer(buffer, data, dataSize, destOffsetBytes);
}

void GPUContext::clear_buffer(
    nvrhi::IBuffer* buffer,
    uint32_t clear_value,
    const nvrhi::BufferRange& range)
{
    commandList_->clearBufferUInt(buffer, clear_value);
}

void GPUContext::clear_texture(
    nvrhi::ITexture* texture,
    nvrhi::Color color,
    const nvrhi::TextureSubresourceSet& subresources)
{
    commandList_->clearTextureFloat(texture, subresources, color);
}

RUZINO_NAMESPACE_CLOSE_SCOPE
