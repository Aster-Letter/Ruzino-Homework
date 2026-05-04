#pragma once
#include "../camera.h"
#include "../geometries/mesh.h"
#include "../global_payload_gl.hpp"
#include "../light.h"
#include "../material.h"
#include "GL/GLResources.hpp"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/frustum.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/glf/simpleLight.h"
#include "pxr/imaging/hd/tokens.h"

RUZINO_NAMESPACE_OPEN_SCOPE
inline void render_node_type_base(NodeTypeInfo* ntype)
{
    ntype->color[0] = 114 / 255.f;
    ntype->color[1] = 94 / 255.f;
    ntype->color[2] = 29 / 255.f;
    ntype->color[3] = 1.0f;
}
#define global_payload      params.get_global_payload<RenderGlobalPayloadGL&>()
#define instance_collection global_payload.InstanceCollection
inline ResourceAllocator& get_resource_allocator(ExeParams& params)
{
    return global_payload.resource_allocator;
}

#define resource_allocator get_resource_allocator(params)
#define shader_factory     get_shader_factory(params)
inline Hd_RUZINO_Camera* get_free_camera(ExeParams& params)
{
    auto& cameras = global_payload.get_cameras();

    Hd_RUZINO_Camera* free_camera;
    for (auto camera : cameras) {
        if (camera->GetId() != SdfPath::EmptyPath()) {
            free_camera = camera;
            break;
        }
    }
    return free_camera;
}

struct ShadowCameraInfo {
    pxr::GfMatrix4f light_projection = pxr::GfMatrix4f(1.0f);
    pxr::GfMatrix4f light_view = pxr::GfMatrix4f(1.0f);
    int shadow_map_id = -1;
};

struct ShadowCameraSettings {
    float fov_degrees = 120.0f;
    float near_plane = 1.0f;
    float far_plane = 25.0f;
    pxr::GfVec3f look_at_target = pxr::GfVec3f(0.0f, 0.0f, 0.0f);
};

template <typename Builder>
inline void DeclareShadowCameraInputs(Builder& b)
{
    b.add_input<float>("Shadow Camera FOV")
        .min(10.0f)
        .max(170.0f)
        .default_val(120.0f);
    b.add_input<float>("Shadow Camera Near")
        .min(0.01f)
        .max(100.0f)
        .default_val(1.05f);
    b.add_input<float>("Shadow Camera Far")
        .min(0.1f)
        .max(500.0f)
        .default_val(25.0f);
    b.add_input<float>("Shadow Target X")
        .min(-100.0f)
        .max(100.0f)
        .default_val(0.0f);
    b.add_input<float>("Shadow Target Y")
        .min(-100.0f)
        .max(100.0f)
        .default_val(0.0f);
    b.add_input<float>("Shadow Target Z")
        .min(-100.0f)
        .max(100.0f)
        .default_val(0.0f);
}

inline ShadowCameraSettings ReadShadowCameraSettings(ExeParams& params)
{
    ShadowCameraSettings settings;
    settings.fov_degrees = params.get_input<float>("Shadow Camera FOV");
    settings.near_plane = params.get_input<float>("Shadow Camera Near");
    settings.far_plane = params.get_input<float>("Shadow Camera Far");
    settings.look_at_target = pxr::GfVec3f(
        params.get_input<float>("Shadow Target X"),
        params.get_input<float>("Shadow Target Y"),
        params.get_input<float>("Shadow Target Z"));
    return settings;
}

inline bool TryBuildShadowCameraInfo(
    const Hd_RUZINO_Light& light,
    const pxr::GlfSimpleLight& light_params,
    const ShadowCameraSettings& settings,
    int shadow_map_id,
    ShadowCameraInfo& out)
{
    if (light.GetLightType() != pxr::HdPrimTypeTokens->sphereLight) {
        return false;
    }

    // Current HW7 shadow setup approximates a sphere light with a single
    // perspective shadow camera that always looks toward the scene origin.
    // This is simple to debug, but it can introduce projection distortion
    // when the useful scene region does not align with that frustum.
    pxr::GfFrustum frustum;
    pxr::GfVec3f light_position = { light_params.GetPosition()[0],
                                    light_params.GetPosition()[1],
                                    light_params.GetPosition()[2] };

    out.light_view = pxr::GfMatrix4f().SetLookAt(
        light_position,
        settings.look_at_target,
        pxr::GfVec3f(0, 0, 1));
    frustum.SetPerspective(
        settings.fov_degrees,
        1.0f,
        settings.near_plane,
        settings.far_plane);
    out.light_projection =
        pxr::GfMatrix4f(frustum.ComputeProjectionMatrix());
    out.shadow_map_id = shadow_map_id;
    return true;
}

#define materials global_payload.get_materials()
#define meshes    global_payload.get_meshes()
#define lights    global_payload.get_lights()

RUZINO_NAMESPACE_CLOSE_SCOPE
