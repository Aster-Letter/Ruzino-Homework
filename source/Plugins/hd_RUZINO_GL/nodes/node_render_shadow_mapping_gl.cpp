

#include "../geometries/mesh.h"
#include "../light.h"
#include "nodes/core/def/node_def.hpp"
#include "pxr/imaging/glf/simpleLight.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "rich_type_buffer.hpp"
#include "utils/draw_fullscreen.h"
NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(shadow_mapping)
{
    // Shadow map resolution. Raise this first when edge aliasing dominates,
    // but note that resolution alone won't fix projection distortion.
    b.add_input<int>("resolution").default_val(1024).min(256).max(4096);

    // Shared shadow camera parameters. These should match the values used by
    // deferred_lighting so the shadow test uses the same light frustum.
    DeclareShadowCameraInputs(b);

    // Shader override for quick experiments with depth visualization.
    b.add_input<std::string>("Shader").default_val("shaders/shadow_mapping.fs");

    b.add_output<GLTextureHandle>("Shadow Maps");
}

NODE_EXECUTION_FUNCTION(shadow_mapping)
{
    auto resolution = params.get_input<int>("resolution");
    auto shadow_camera_settings = ReadShadowCameraSettings(params);

    GLTextureDesc texture_desc;
    texture_desc.array_size = lights.size();
    // texture_desc.array_size = 1;
    texture_desc.size = GfVec2i(resolution);
    texture_desc.format = HdFormatFloat32;
    auto shadow_map_texture = resource_allocator.create(texture_desc);

    auto shaderPath = params.get_input<std::string>("Shader");

    GLShaderDesc shader_desc;
    shader_desc.set_vertex_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path("shaders/shadow_mapping.vs"));

    shader_desc.set_fragment_path(
        std::filesystem::path(RENDER_NODES_FILES_DIR) /
        std::filesystem::path(shaderPath));
    auto shader_handle = resource_allocator.create(shader_desc);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    std::vector<GLTextureHandle> depth_textures;
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glViewport(0, 0, resolution, resolution);
    int NN = lights.size();
    auto tmp = global_payload;
    for (int light_id = 0; light_id < NN; ++light_id) {
        shader_handle->shader.use();
        if (!lights[light_id]->GetId().IsEmpty()) {
            GlfSimpleLight light_params =
                lights[light_id]->Get(HdTokens->params).Get<GlfSimpleLight>();

            ShadowCameraInfo shadow_camera;
            if (!TryBuildShadowCameraInfo(
                    *lights[light_id],
                    light_params,
                    shadow_camera_settings,
                    light_id,
                    shadow_camera)) {
                continue;
            }

            shader_handle->shader.setMat4(
                "light_view", shadow_camera.light_view);
            shader_handle->shader.setMat4(
                "light_projection", shadow_camera.light_projection);

            glFramebufferTextureLayer(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                shadow_map_texture->texture_id,
                0,
                light_id);

            texture_desc.format = HdFormatFloat32UInt8;
            texture_desc.array_size = 1;
            auto depth_texture_for_opengl =
                resource_allocator.create(texture_desc);
            depth_textures.push_back(depth_texture_for_opengl);

            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_DEPTH_STENCIL_ATTACHMENT,
                GL_TEXTURE_2D,
                depth_texture_for_opengl->texture_id,
                0);

            glClearColor(0.f, 0.f, 0.f, 1.0f);
            glClear(
                GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
                GL_STENCIL_BUFFER_BIT);

            for (int mesh_id = 0; mesh_id < meshes.size(); ++mesh_id) {
                auto mesh = meshes[mesh_id];

                shader_handle->shader.setMat4("model", mesh->transform);

                mesh->RefreshGLBuffer();

                glBindVertexArray(mesh->VAO);
                glDrawElements(
                    GL_TRIANGLES,
                    static_cast<unsigned int>(
                        mesh->triangulatedIndices.size() * 3),
                    GL_UNSIGNED_INT,
                    0);
                glBindVertexArray(0);
            }
        }
    }

    for (auto&& depth_texture : depth_textures) {
        resource_allocator.destroy(depth_texture);
    }

    auto shader_error = shader_handle->shader.get_error();

    resource_allocator.destroy(shader_handle);
    glDeleteFramebuffers(1, &framebuffer);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    params.set_output("Shadow Maps", shadow_map_texture);
    if (!shader_error.empty()) {
        return false;
    }
    return true;
}

NODE_DECLARATION_UI(shadow_mapping);
NODE_DEF_CLOSE_SCOPE
