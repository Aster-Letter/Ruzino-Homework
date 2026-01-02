#pragma once

#include "RHI/ResourceManager/resource_allocator.hpp"
#include "pxr/usd/sdf/path.h"

namespace Ruzino {
class Hd_RUZINO_RenderInstanceCollection;
class Hd_RUZINO_Light;
class Hd_RUZINO_Camera;
class Hd_RUZINO_Mesh;
class Hd_RUZINO_Material;
}  // namespace Ruzino

namespace Ruzino {

struct RenderGlobalPayloadGL {
    RenderGlobalPayloadGL()
    {
    }
    RenderGlobalPayloadGL(
        pxr::VtArray<Hd_RUZINO_Camera*>* cameras,
        pxr::VtArray<Hd_RUZINO_Light*>* lights,
        pxr::VtArray<Hd_RUZINO_Mesh*>* meshes,
        pxr::TfHashMap<pxr::SdfPath, Hd_RUZINO_Material*, pxr::TfHash>*
            materials)
        : cameras(cameras),
          lights(lights),
          meshes(meshes),
          materials(materials)
    {
    }

    RenderGlobalPayloadGL(const RenderGlobalPayloadGL& rhs)
        : cameras(rhs.cameras),
          lights(rhs.lights),
          meshes(rhs.meshes),
          materials(rhs.materials)
    {
    }

    RenderGlobalPayloadGL& operator=(const RenderGlobalPayloadGL& rhs)
    {
        cameras = rhs.cameras;
        lights = rhs.lights;
        meshes = rhs.meshes;
        materials = rhs.materials;
        return *this;
    }

    ResourceAllocator resource_allocator;

    auto& get_cameras() const
    {
        return *cameras;
    }

    auto& get_lights() const
    {
        return *lights;
    }

    auto& get_materials() const
    {
        return *materials;
    }

    auto& get_meshes() const
    {
        return *meshes;
    }

   private:
    pxr::VtArray<Hd_RUZINO_Camera*>* cameras;
    pxr::VtArray<Hd_RUZINO_Light*>* lights;
    pxr::TfHashMap<pxr::SdfPath, Hd_RUZINO_Material*, pxr::TfHash>* materials;
    pxr::VtArray<Hd_RUZINO_Mesh*>* meshes;
};

}  // namespace Ruzino
