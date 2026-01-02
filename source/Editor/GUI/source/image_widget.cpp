#include "GUI/image_widget.hpp"

#include <string>

#include "imgui.h"

RUZINO_NAMESPACE_OPEN_SCOPE

ImageWidget::ImageWidget(nvrhi::ITexture* texture)
    : texture_(texture),
      texture_handle_{},
      use_handle_(false)
{
    width = 512;
    height = 512;
}

ImageWidget::ImageWidget(nvrhi::TextureHandle texture_handle)
    : texture_(nullptr),
      texture_handle_(texture_handle),
      use_handle_(true)
{
    width = 512;
    height = 512;
}

bool ImageWidget::BuildUI()
{
    if (use_handle_) {
        // 使用TextureHandle渲染
        if (texture_handle_) {
            ImVec2 content_region = ImGui::GetContentRegionAvail();
            ImVec2 image_size = content_region;

            // 保持纵横比
            if (content_region.x > 0 && content_region.y > 0) {
                float aspect = content_region.x / content_region.y;
                if (aspect > 1.0f) {
                    image_size.x = content_region.y;
                    image_size.y = content_region.y;
                }
                else {
                    image_size.x = content_region.x;
                    image_size.y = content_region.x;
                }
            }

            // 居中显示图像
            ImVec2 cursor_pos = ImGui::GetCursorPos();
            cursor_pos.x += (content_region.x - image_size.x) * 0.5f;
            cursor_pos.y += (content_region.y - image_size.y) * 0.5f;
            ImGui::SetCursorPos(cursor_pos);

            ImGui::Image(texture_handle_.Get(), image_size);
        }
        else {
            ImGui::Text("Invalid texture handle");
        }
    }
    else {
        // 使用ITexture*渲染
        if (texture_) {
            ImVec2 content_region = ImGui::GetContentRegionAvail();
            ImVec2 image_size = content_region;

            // 保持纵横比
            if (content_region.x > 0 && content_region.y > 0) {
                float aspect = content_region.x / content_region.y;
                if (aspect > 1.0f) {
                    image_size.x = content_region.y;
                    image_size.y = content_region.y;
                }
                else {
                    image_size.x = content_region.x;
                    image_size.y = content_region.x;
                }
            }

            // 居中显示图像
            ImVec2 cursor_pos = ImGui::GetCursorPos();
            cursor_pos.x += (content_region.x - image_size.x) * 0.5f;
            cursor_pos.y += (content_region.y - image_size.y) * 0.5f;
            ImGui::SetCursorPos(cursor_pos);

            ImGui::Image(texture_, image_size);
        }
        else {
            ImGui::Text("No texture provided");
        }
    }

    return true;
}

const char* ImageWidget::GetWindowName()
{
    return "Image Viewer";
}

RUZINO_NAMESPACE_CLOSE_SCOPE
