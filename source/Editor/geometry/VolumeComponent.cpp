#include "GCore/Components/VolumeComponent.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
GeometryComponentHandle VolumeComponent::copy(Geometry* operand) const
{
    auto new_component = std::make_shared<VolumeComponent>(operand);
    new_component->add_grid(grid);
    return new_component;
}

std::string VolumeComponent::to_string() const
{
    return std::string("VolumeComponet");
}

void VolumeComponent::add_grid(openvdb::FloatGrid::Ptr grid)
{
    this->grid = grid;
}

void VolumeComponent::apply_transform(const pxr::GfMatrix4d& transform)
{
    throw std::runtime_error(
        "VolumeComponent::apply_transform not implemented");
}

void VolumeComponent::write_disk(const std::string& file_name)
{
    openvdb::io::File file(file_name);
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

openvdb::GridBase::Ptr VolumeComponent::get_grid()
{
    return grid;
}

USTC_CG_NAMESPACE_CLOSE_SCOPE
