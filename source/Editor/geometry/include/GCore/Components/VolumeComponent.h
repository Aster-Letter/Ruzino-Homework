#pragma once

#include "GCore/Components.h"
#include "GCore/GOP.h"
#include "GCore/api.h"
#include "GCore/geom_payload.hpp"
#include "openvdb/openvdb.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
class GEOMETRY_API VolumeComponent : public GeometryComponent {
   public:
    explicit VolumeComponent(Geometry* attached_operand)
        : GeometryComponent(attached_operand)
    {
    }

    GeometryComponentHandle copy(Geometry* operand) const override;
    std::string to_string() const override;
    void add_grid(openvdb::FloatGrid::Ptr grid);

    void apply_transform(const pxr::GfMatrix4d& transform) override;
    void write_disk(const std::string& file_name);
    openvdb::GridBase::Ptr get_grid();

private:
    std::unordered_map<std::string, openvdb::FloatGrid::Ptr> grids_;
    openvdb::FloatGrid::Ptr grid;
};
USTC_CG_NAMESPACE_CLOSE_SCOPE
