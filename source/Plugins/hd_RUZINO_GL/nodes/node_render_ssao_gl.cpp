

#include "../camera.h"
#include "../light.h"
#include "nodes/core/def/node_def.hpp"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/imaging/hd/tokens.h"
#include "render_node_base.h"
#include "rich_type_buffer.hpp"
#include "utils/draw_fullscreen.h"

#ifndef RENDER_NODES_FILES_DIR
#define RENDER_NODES_FILES_DIR "."
#endif

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(ssao)
{
    b.add_input<GLTextureHandle>("Color");
    b.add_input<GLTextureHandle>("Position");
    b.add_input<GLTextureHandle>("Depth");
    b.add_input<GLTextureHandle>("Normal");

    // Sample radius in world/view neighborhood space.
    b.add_input<float>("Radius").min(0.05f).max(3.0f).default_val(0.35f);

    // Overall AO influence. Since this node post-multiplies the lit color,
    // keep the range compact to avoid turning AO into fake hard shadows.
    b.add_input<float>("Strength")
        .min(0.0f)
        .max(2.0f)
        .default_val(0.9f);
    b.add_output<GLTextureHandle>("Color");
}

NODE_EXECUTION_FUNCTION(ssao)
{
    auto color = params.get_input<GLTextureHandle>("Color");
    auto position = params.get_input<GLTextureHandle>("Position");
    auto normal = params.get_input<GLTextureHandle>("Normal");
    Hd_RUZINO_Camera* free_camera = get_free_camera(params);

    auto size = color->desc.size;

    unsigned int VBO, VAO;

    CreateFullScreenVAO(VAO, VBO);

    GLTextureDesc texture_desc;
    texture_desc.size = size;
    texture_desc.format = HdFormatFloat32Vec4;
    auto color_texture = resource_allocator.create(texture_desc);

    const std::string shaderPath = "shaders/ssao.fs";

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

    shader->shader.setInt("colorSampler", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color->texture_id);

    shader->shader.setInt("positionSampler", 1);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, position->texture_id);

    shader->shader.setInt("normalSampler", 2);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, normal->texture_id);

    GfMatrix4f projection(free_camera->_projMatrix);
    shader->shader.setMat4("view", GfMatrix4f(free_camera->_viewMatrix));
    shader->shader.setMat4("projection", projection);
    shader->shader.setFloat("radius", params.get_input<float>("Radius"));
    shader->shader.setFloat("strength", params.get_input<float>("Strength"));

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    DestroyFullScreenVAO(VAO, VBO);

    auto shader_error = shader->shader.get_error();

    resource_allocator.destroy(shader);

    params.set_output("Color", color_texture);
    if (!shader_error.empty()) {
        return false;
    }
    return true;
}

NODE_DECLARATION_UI(ssao);
NODE_DEF_CLOSE_SCOPE
