#include <gtest/gtest.h>
#include <nvrhi/nvrhi.h>

#include "GUI/image_widget.hpp"
#include "GUI/widget.h"
#include "GUI/window.h"
#include "RHI/rhi.hpp"
#include "imgui.h"

using namespace USTC_CG;

TEST(CreateRHI, window)
{
    Window window;
    window.run();
}

class Widget : public IWidget {
   public:
    explicit Widget(const char* title) : title(title)
    {
    }

    bool BuildUI() override
    {
        ImGui::Text("Hello, world!");
        return true;
    }

   private:
    std::string title;
};

class WidgetFactory : public IWidgetFactory {
   public:
    std::unique_ptr<IWidget> Create(
        const std::vector<std::unique_ptr<IWidget>>& others) override
    {
        return std::make_unique<Widget>("widget");
    }
};

TEST(CreateRHI, widget_factory)
{
    Window window;
    window.register_openable_widget(
        std::make_unique<WidgetFactory>(), { "File", "Open", "widget" });
    window.run();
}

TEST(CreateRHI, widget)
{
    Window window;
    std::unique_ptr<IWidget> widget = std::make_unique<Widget>("widget");
    window.register_widget(std::move(widget));
    window.run();
}

TEST(CreateRHI, multiple_widgets)
{
    Window window;
    window.register_widget(std::make_unique<Widget>("widget"));
    window.register_widget(std::make_unique<Widget>("widget2"));
    window.run();
}

#include "GUI/ImGuiFileDialog.h"

class FileWidget : public IWidget {
   public:
    explicit FileWidget(const char* title) : title(title)
    {
    }

    bool BuildUI() override
    {
        auto instance = IGFD::FileDialog::Instance();

        instance->Display("SelectFile");

        return true;
    }

   protected:
    const char* GetWindowName() override
    {
        return title.c_str();
    }

   private:
    std::string title;
};

TEST(FileDialog, create_dialog)
{
    Window window;
    window.register_widget(std::make_unique<FileWidget>("file"));
    window.run();
}

// TEST(ImageWidget, red_texture)
int main()
{
#if USTC_CG_WITH_CUDA
    // Initialize RHI and CUDA
    RHI::init(true);
    USTC_CG::cuda::cuda_init();

    Window window;

    // Create a 256x256 texture using CUDA linear buffer
    const int width = 256;
    const int height = 256;
    const int pixel_count = width * height;

    // Create CUDA linear buffer for RGBA data
    auto cuda_buffer =
        USTC_CG::cuda::create_cuda_linear_buffer<uint32_t>(pixel_count);

    // Fill the buffer with red color using CUDA (validation of CUDA filling)
    std::vector<uint32_t> red_data(pixel_count);
    for (int i = 0; i < pixel_count; ++i) {
        // RGBA format as uint32_t: 0xAABBGGRR
        red_data[i] = 0xFF007FFF;  // Alpha=255, Blue=0, Green=127, Red=255
    }
    cuda_buffer->assign_host_vector(red_data);

    // Create texture description
    nvrhi::TextureDesc texture_desc;
    texture_desc.width = width;
    texture_desc.height = height;
    texture_desc.format = nvrhi::Format::RGBA8_UNORM;
    texture_desc.debugName = "cuda_red_texture";
    texture_desc.isRenderTarget = false;
    texture_desc.isShaderResource = true;

    // Convert CUDA linear buffer to NVRHI texture (validating the conversion)
    auto device = RHI::get_device();
    auto texture_handle = USTC_CG::cuda::cuda_linear_buffer_to_nvrhi_texture(
        device, cuda_buffer, texture_desc);

    // Create and register the image widget
    auto image_widget = std::make_unique<ImageWidget>(texture_handle);
    window.register_widget(std::move(image_widget));

    window.run();

#else
    // Fallback for non-CUDA builds
    RHI::init(true);

    Window window;

    // Create a simple red texture without CUDA
    const int width = 256;
    const int height = 256;
    std::vector<uint8_t> red_data(width * height * 4);

    for (int i = 0; i < width * height; ++i) {
        red_data[i * 4 + 0] = 255;  // Red
        red_data[i * 4 + 1] = 127;  // Green
        red_data[i * 4 + 2] = 0;    // Blue
        red_data[i * 4 + 3] = 255;  // Alpha
    }

    nvrhi::TextureDesc texture_desc;
    texture_desc.width = width;
    texture_desc.height = height;
    texture_desc.format = nvrhi::Format::RGBA8_UNORM;
    texture_desc.debugName = "fallback_red_texture";
    texture_desc.isRenderTarget = false;
    texture_desc.isShaderResource = true;

    auto [texture_handle, staging_handle] =
        RHI::load_texture(texture_desc, red_data.data());

    auto image_widget = std::make_unique<ImageWidget>(texture_handle);
    window.register_widget(std::move(image_widget));

    window.run();

    RHI::shutdown();
#endif
}
