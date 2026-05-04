// #define __GNUC__

#include "../camera.h"
#include "../light.h"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/glf/simpleLight.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "rich_type_buffer.hpp"
#include "utils/draw_fullscreen.h"
NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(deferred_lighting)
{
    b.add_input<GLTextureHandle>("Position");
    b.add_input<GLTextureHandle>("diffuseColor");
    b.add_input<GLTextureHandle>("MetallicRoughness");
    b.add_input<GLTextureHandle>("Normal");
    b.add_input<GLTextureHandle>("Shadow Maps");

    // Shared shadow camera parameters. Keep these consistent with the
    // shadow_mapping node so shadow generation and lookup use identical
    // light-space transforms.
    DeclareShadowCameraInputs(b);

    // Constant ambient fill light added after deferred shading.
    // Keep this low, otherwise shadow contrast becomes hard to judge.
    b.add_input<float>("Ambient Strength")
        .min(0.0f)
        .max(1.0f)
        .default_val(0.03f);

    // Slope-scaled shadow bias term. Increase when acne dominates;
    // too large a value causes detached shadows (peter-panning).
    b.add_input<float>("Shadow Bias Scale")
        .min(0.0f)
        .max(0.05f)
        .default_val(0.005f);

    // Radius multiplier for 3x3 PCF filtering in shadow map texel space.
    // 0 disables filtering; larger values soften edges but can blur contact shadows.
    b.add_input<float>("Shadow PCF Radius")
        .min(0.0f)
        .max(4.0f)
        .default_val(1.0f);

    // Effective PCSS light size scale. This controls penumbra growth and is
    // intentionally separated from Shadow PCF Radius so softness and filter
    // footprint can be tuned independently.
    b.add_input<float>("Shadow Light Size")
        .min(0.0f)
        .max(8.0f)
        .default_val(1.5f);

    // Enable a lightweight post AA pass in the fullscreen lighting shader.
    // This is cheaper to integrate than MSAA for the current deferred pipeline.
    b.add_input<bool>("Enable Antialiasing").default_val(true);

    // Edge sensitivity for post AA. Smaller values smooth more jagged edges,
    // larger values keep more detail but may miss thin stair-steps.
    b.add_input<float>("AA Edge Threshold")
        .min(0.0f)
        .max(0.5f)
        .default_val(0.08f);

    // Blend strength toward neighbor colors on detected edges.
    b.add_input<float>("AA Blend Strength")
        .min(0.0f)
        .max(1.0f)
        .default_val(0.75f);

    // Toggle inverse-square attenuation for point-like light falloff.
    b.add_input<bool>("Enable Distance Attenuation").default_val(true);

    b.add_input<std::string>("Lighting Shader")
        .default_val("shaders/blinn_phong.fs");
    b.add_output<GLTextureHandle>("Color");
}

struct LightInfo {
    GfMatrix4f light_projection;
    GfMatrix4f light_view;
    GfVec3f position;
    float radius;
    GfVec3f color;
    int shadow_map_id;
};

struct DeferredLightingSettings {
    float ambient_strength;
    float shadow_bias_scale;
    float shadow_pcf_radius;
    float shadow_light_size;
    bool enable_antialiasing;
    float aa_edge_threshold;
    float aa_blend_strength;
    bool enable_distance_attenuation;
};

NODE_EXECUTION_FUNCTION(deferred_lighting)
{
    // Fetch all the information

    auto read_deferred_lighting_settings = [&params]() {
        DeferredLightingSettings settings;
        settings.ambient_strength =
            params.get_input<float>("Ambient Strength");
        settings.shadow_bias_scale =
            params.get_input<float>("Shadow Bias Scale");
        settings.shadow_pcf_radius =
            params.get_input<float>("Shadow PCF Radius");
        settings.shadow_light_size =
            params.get_input<float>("Shadow Light Size");
        settings.enable_antialiasing =
            params.get_input<bool>("Enable Antialiasing");
        settings.aa_edge_threshold =
            params.get_input<float>("AA Edge Threshold");
        settings.aa_blend_strength =
            params.get_input<float>("AA Blend Strength");
        settings.enable_distance_attenuation =
            params.get_input<bool>("Enable Distance Attenuation");
        return settings;
    };

    auto build_deferred_light_buffer = [&]() {
        std::vector<LightInfo> light_vector;
        light_vector.reserve(lights.size());
        auto shadow_camera_settings = ReadShadowCameraSettings(params);

        for (int i = 0; i < lights.size(); ++i) {
            if (lights[i]->GetId().IsEmpty()) {
                continue;
            }

            GlfSimpleLight light_params =
                lights[i]->Get(HdTokens->params).Get<GlfSimpleLight>();
            auto diffuse4 = light_params.GetDiffuse();
            pxr::GfVec3f color(diffuse4[0], diffuse4[1], diffuse4[2]);
            auto position4 = light_params.GetPosition();
            pxr::GfVec3f position(position4[0], position4[1], position4[2]);

            float radius = 1.0f;
            if (lights[i]->Get(HdLightTokens->radius).IsHolding<float>()) {
                radius = lights[i]->Get(HdLightTokens->radius).Get<float>();
            }

            ShadowCameraInfo shadow_camera;
            TryBuildShadowCameraInfo(
                *lights[i],
                light_params,
                shadow_camera_settings,
                i,
                shadow_camera);

            light_vector.emplace_back(
                shadow_camera.light_projection,
                shadow_camera.light_view,
                position,
                radius,
                color,
                shadow_camera.shadow_map_id);
        }

        return light_vector;
    };

    auto position_texture = params.get_input<GLTextureHandle>("Position");
    auto diffuseColor_texture =
        params.get_input<GLTextureHandle>("diffuseColor");

    auto metallic_roughness =
        params.get_input<GLTextureHandle>("MetallicRoughness");
    auto normal_texture = params.get_input<GLTextureHandle>("Normal");

    auto shadow_maps = params.get_input<GLTextureHandle>("Shadow Maps");
    auto settings = read_deferred_lighting_settings();

    Hd_RUZINO_Camera* free_camera = get_free_camera(params);
    // Creating output textures.
    auto size = position_texture->desc.size;
    GLTextureDesc color_output_desc;
    color_output_desc.format = HdFormatFloat32Vec4;
    color_output_desc.size = size;
    auto color_texture = resource_allocator.create(color_output_desc);

    unsigned int VBO, VAO;
    CreateFullScreenVAO(VAO, VBO);

    auto shaderPath = params.get_input<std::string>("Lighting Shader");

    GLShaderDesc shader_desc;
    shader_desc.set_vertex_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path("shaders/fullscreen.vs"));

    shader_desc.set_fragment_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path(shaderPath));
    auto shader = resource_allocator.create(shader_desc);
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        color_texture->texture_id,
        0);

    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader->shader.use();
    shader->shader.setVec2("iResolution", size);

    shader->shader.setInt("diffuseColorSampler", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, diffuseColor_texture->texture_id);

    shader->shader.setInt("normalMapSampler", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normal_texture->texture_id);

    shader->shader.setInt("metallicRoughnessSampler", 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, metallic_roughness->texture_id);

    shader->shader.setInt("shadow_maps", 3);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D_ARRAY, shadow_maps->texture_id);

    shader->shader.setInt("position", 4);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, position_texture->texture_id);

    GfVec3f camPos =
        GfMatrix4f(free_camera->GetTransform()).ExtractTranslation();
    shader->shader.setVec3("camPos", camPos);
    shader->shader.setFloat("ambientStrength", settings.ambient_strength);
    shader->shader.setFloat("shadowBiasScale", settings.shadow_bias_scale);
    shader->shader.setFloat("shadowPcfRadius", settings.shadow_pcf_radius);
    shader->shader.setFloat("shadowLightSize", settings.shadow_light_size);
    shader->shader.setBool("enableAntialiasing", settings.enable_antialiasing);
    shader->shader.setFloat("aaEdgeThreshold", settings.aa_edge_threshold);
    shader->shader.setFloat("aaBlendStrength", settings.aa_blend_strength);
    shader->shader.setBool(
        "enableDistanceAttenuation", settings.enable_distance_attenuation);

    GLuint lightBuffer;
    glGenBuffers(1, &lightBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightBuffer);
    glViewport(0, 0, size[0], size[1]);
    std::vector<LightInfo> light_vector = build_deferred_light_buffer();

    shader->shader.setInt("light_count", static_cast<int>(light_vector.size()));

    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        light_vector.size() * sizeof(LightInfo),
        light_vector.data(),
        GL_STATIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lightBuffer);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    DestroyFullScreenVAO(VAO, VBO);

    auto shader_error = shader->shader.get_error();

    resource_allocator.destroy(shader);
    glDeleteBuffers(1, &lightBuffer);
    glDeleteFramebuffers(1, &framebuffer);
    params.set_output("Color", color_texture);
    if (!shader_error.empty()) {
        return false;
    }
    return true;
}

NODE_DECLARATION_UI(deferred_lighting);
NODE_DEF_CLOSE_SCOPE
