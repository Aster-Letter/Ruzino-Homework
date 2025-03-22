#include "material.h"

#include <MaterialXGenShader/TypeDesc.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/usdImaging/usdImaging/tokens.h>

#include <filesystem>
#include <fstream>

#include "MaterialX/SlangShaderGenerator.h"
#include "MaterialXCore/Document.h"
#include "MaterialXFormat/Util.h"
#include "MaterialXGenShader/Shader.h"
#include "MaterialXGenShader/Util.h"
#include "RHI/Hgi/format_conversion.hpp"
#include "api.h"
#include "hdMtlxFast.h"
#include "materialFilter.h"
#include "nvrhi/nvrhi.h"
#include "pxr/base/arch/fileSystem.h"
#include "pxr/base/arch/hash.h"
#include "pxr/base/arch/library.h"
#include "pxr/imaging/hd/changeTracker.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/ar/resolver.h"
#include "pxr/usd/sdr/registry.h"
#include "pxr/usd/sdr/shaderNode.h"

USTC_CG_NAMESPACE_OPEN_SCOPE
using namespace MaterialX;
/// Shared pointer to a BindlessContext
using BindlessContextPtr = std::shared_ptr<class BindlessContext>;

/// @class BindlessContext
/// Class representing a resource binding for Slang shader resources.
class BindlessContext : public HwResourceBindingContext {
   public:
    BindlessContext(
        size_t uniformBindingLocation,
        size_t samplerBindingLocation);

    static BindlessContextPtr create(
        size_t uniformBindingLocation = 0,
        size_t samplerBindingLocation = 0)
    {
        return std::make_shared<BindlessContext>(
            uniformBindingLocation, samplerBindingLocation);
    }

    void initialize() override
    {
        fetch_data = "";
        data_location = 0;
    }

    void emitDirectives(GenContext& context, ShaderStage& stage) override
    {
        const ShaderGenerator& generator = context.getShaderGenerator();
        generator.emitLine("import Scene.BindlessMaterial", stage);
    }

    // Emit uniforms with binding information
    void emitResourceBindings(
        GenContext& context,
        const VariableBlock& uniforms,
        ShaderStage& stage) override;

    // Emit structured uniforms with binding information and align members where
    // possible
    void emitStructuredResourceBindings(
        GenContext& context,
        const VariableBlock& uniforms,
        ShaderStage& stage,
        const std::string& structInstanceName,
        const std::string& arraySuffix) override;

    std::string get_data_code()
    {
        return fetch_data;
    }

   private:
    std::string fetch_data = "";
    unsigned int data_location = 0;
};

//
// BindlessContext
//
BindlessContext::BindlessContext(
    size_t uniformBindingLocation,
    size_t samplerBindingLocation)
{
}

void BindlessContext::emitResourceBindings(
    GenContext& context,
    const VariableBlock& uniforms,
    ShaderStage& stage)
{
    const ShaderGenerator& generator = context.getShaderGenerator();
    const Syntax& syntax = generator.getSyntax();

    // First, emit all value uniforms in a block with single layout binding
    bool hasValueUniforms = false;
    for (auto uniform : uniforms.getVariableOrder()) {
        if (uniform->getType() != Type::FILENAME) {
            hasValueUniforms = true;
            break;
        }
    }
    if (hasValueUniforms && uniforms.getName() == HW::PUBLIC_UNIFORMS) {
        for (auto uniform : uniforms.getVariableOrder()) {
            auto type = uniform->getType();

            auto& syntax = generator.getSyntax();

            if (type != Type::FILENAME) {
                if (type == Type::VECTOR3) {
                    string float3_string =
                        "asfloat(data.data[" + std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) + "])";

                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + float3_string +
                                  ";\n";
                }
                else if (type == Type::FLOAT) {
                    string float_string = "asfloat(data.data[" +
                                          std::to_string(data_location++) +
                                          "])";
                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + float_string +
                                  ";\n";
                }
                else if (type == Type::INTEGER) {
                    string int_string = "asint(data.data[" +
                                        std::to_string(data_location++) + "])";
                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + int_string +
                                  ";\n";
                }
                else if (type == Type::MATRIX44) {
                    std::ostringstream matrix_stream;
                    matrix_stream << "float4x4(";

                    // Create matrix with 4 rows of 4 elements each
                    for (int row = 0; row < 4; row++) {
                        for (int col = 0; col < 4; col++) {
                            matrix_stream << "asfloat(data.data["
                                          << data_location++ << "])";
                            // Add comma after every element except the last
                            // one
                            if (!(row == 3 && col == 3)) {
                                matrix_stream << ", ";
                            }
                        }
                    }

                    matrix_stream << ")";
                    string matrix_string = matrix_stream.str();

                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + matrix_string +
                                  ";\n";
                }
                else if (type == Type::COLOR3) {
                    string float3_string =
                        "asfloat(data.data[" + std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) + "])";

                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + float3_string +
                                  ";\n";
                }
                else if (type == Type::COLOR4) {
                    string float4_string =
                        "asfloat(data.data[" + std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) + "])";

                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + float4_string +
                                  ";\n";
                }
                else if (type == Type::VECTOR2) {
                    string float2_string =
                        "asfloat(data.data[" + std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) + "])";

                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + float2_string +
                                  ";\n";
                }
                else if (type == Type::DISPLACEMENTSHADER) {
                    // First load the float3 and float parameters for the
                    // displacement shader
                    string float3_string =
                        "asfloat(data.data[" + std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) +
                        "]), asfloat(data.data[" +
                        std::to_string(data_location++) + "])";

                    string float_string = "asfloat(data.data[" +
                                          std::to_string(data_location++) +
                                          "])";

                    // Initialize the displacement shader with the float3 and
                    // float parameters
                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " +
                                  "displacementshader(float3(" + float3_string +
                                  "), " + float_string + ");\n";
                }
                else if (type == Type::STRING) {
                    string int_string = "asint(data.data[" +
                                        std::to_string(data_location++) + "])";
                    fetch_data += syntax.getTypeName(type) + " " +
                                  uniform->getName() + " = " + int_string +
                                  ";\n";
                }

                else {
                    log::warning(
                        ("Unsupported uniform type: " + type->getName())
                            .c_str());
                }

                // generator.emitLineBegin(stage);
                // generator.emitVariableDeclaration(
                //     uniform, EMPTY_STRING, context, stage, true);
                // generator.emitString(Syntax::SEMICOLON, stage);
                // generator.emitLineEnd(stage, false);
            }
        }

        // Second, emit all sampler uniforms as separate uniforms with separate
        // layout bindings
        for (auto uniform : uniforms.getVariableOrder()) {
            if (*uniform->getType() == *Type::FILENAME) {
                // generator.emitString(
                //     "layout (binding=" +
                //         std::to_string(
                //             _separateBindingLocation ?
                //             _hwUniformBindLocation++
                //                                      :
                //                                      _hwSamplerBindLocation++)
                //                                      +
                //         ") " + syntax.getUniformQualifier() + " ",
                //     stage);
                // generator.emitVariableDeclaration(
                //    uniform, EMPTY_STRING, context, stage, true);
                // generator.emitLineEnd(stage, true);

                fetch_data += "Sampler2D " + uniform->getName() + " = " +
                              " t_BindlessTextures[$" + uniform->getName() +
                              "_id];\n";
            }
        }
    }
    generator.emitLineBreak(stage);
}

void BindlessContext::emitStructuredResourceBindings(
    GenContext& context,
    const VariableBlock& uniforms,
    ShaderStage& stage,
    const std::string& structInstanceName,
    const std::string& arraySuffix)
{
    const ShaderGenerator& generator = context.getShaderGenerator();
    const Syntax& syntax = generator.getSyntax();

    // Slang structures need to be aligned. We make a best effort to base align
    // struct members and add padding if required.
    // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_uniform_buffer_object.txt

    const size_t baseAlignment = 16;
    std::unordered_map<const TypeDesc*, size_t> alignmentMap(
        { { Type::FLOAT, baseAlignment / 4 },
          { Type::INTEGER, baseAlignment / 4 },
          { Type::BOOLEAN, baseAlignment / 4 },
          { Type::COLOR3, baseAlignment },
          { Type::COLOR4, baseAlignment },
          { Type::VECTOR2, baseAlignment },
          { Type::VECTOR3, baseAlignment },
          { Type::VECTOR4, baseAlignment },
          { Type::MATRIX33, baseAlignment * 4 },
          { Type::MATRIX44, baseAlignment * 4 } });

    // Get struct alignment and size
    // alignment, uniform member index
    vector<std::pair<size_t, size_t>> memberOrder;
    size_t structSize = 0;
    for (size_t i = 0; i < uniforms.size(); ++i) {
        auto it = alignmentMap.find(uniforms[i]->getType());
        if (it == alignmentMap.end()) {
            structSize += baseAlignment;
            memberOrder.push_back(std::make_pair(baseAlignment, i));
        }
        else {
            structSize += it->second;
            memberOrder.push_back(std::make_pair(it->second, i));
        }
    }

    // Align up and determine number of padding floats to add
    const size_t numPaddingfloats =
        (((structSize + (baseAlignment - 1)) & ~(baseAlignment - 1)) -
         structSize) /
        4;

    // Sort order from largest to smallest
    std::sort(
        memberOrder.begin(),
        memberOrder.end(),
        [](const std::pair<size_t, size_t>& a,
           const std::pair<size_t, size_t>& b) { return a.first > b.first; });

    // Emit the struct
    generator.emitLine("struct " + uniforms.getName(), stage, false);
    generator.emitScopeBegin(stage);

    for (size_t i = 0; i < uniforms.size(); ++i) {
        size_t variableIndex = memberOrder[i].second;
        generator.emitLineBegin(stage);
        generator.emitVariableDeclaration(
            uniforms[variableIndex], EMPTY_STRING, context, stage, false);
        generator.emitString(Syntax::SEMICOLON, stage);
        generator.emitLineEnd(stage, false);
    }

    // Emit padding
    for (size_t i = 0; i < numPaddingfloats; ++i) {
        generator.emitLine("float pad" + std::to_string(i), stage, true);
    }
    generator.emitScopeEnd(stage, true);

    // Emit binding information
    generator.emitLineBreak(stage);
    generator.emitLine(
        syntax.getUniformQualifier() + " " + uniforms.getName() + "_" +
            stage.getName(),
        stage,
        false);
    generator.emitScopeBegin(stage);
    generator.emitLine(
        uniforms.getName() + " " + structInstanceName + arraySuffix, stage);
    generator.emitScopeEnd(stage, true);
}

namespace mx = MaterialX;

MaterialX::GenContextPtr Hd_USTC_CG_Material::shader_gen_context_ =
    std::make_shared<mx::GenContext>(mx::SlangShaderGenerator::create());
MaterialX::DocumentPtr Hd_USTC_CG_Material::libraries = mx::createDocument();

std::mutex Hd_USTC_CG_Material::shadergen_mutex;
std::mutex Hd_USTC_CG_Material::texture_mutex;
std::mutex Hd_USTC_CG_Material::material_data_handle_mutex;

std::once_flag Hd_USTC_CG_Material::shader_gen_initialized_;

Hd_USTC_CG_Material::Hd_USTC_CG_Material(SdfPath const& id) : HdMaterial(id)
{
    std::call_once(shader_gen_initialized_, []() {
        mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
        searchPath.append(mx::FileSearchPath("usd/hd_USTC_CG/resources"));

        loadLibraries({ "libraries" }, searchPath, libraries);
        mx::loadLibraries(
            { "usd/hd_USTC_CG/resources/libraries" }, searchPath, libraries);
        shader_gen_context_->registerSourceCodeSearchPath(searchPath);

        shader_gen_context_->pushUserData(
            mx::HW::USER_DATA_BINDING_CONTEXT, BindlessContext::create());
    });
}

TF_DEFINE_PRIVATE_TOKENS(_tokens, (file));
void Hd_USTC_CG_Material::CollectTextures(
    HdMaterialNetwork2Interface netInterface,
    HdMtlxTexturePrimvarData hdMtlxData)
{
    // Collect texture names and paths into a vector.
    for (const SdfPath& texturePath : hdMtlxData.hdTextureNodes) {
        TfToken textureNodeName = texturePath.GetToken();
        // Get the file parameter from the node.
        VtValue vFile =
            netInterface.GetNodeParameterValue(textureNodeName, _tokens->file);
        std::string path;
        if (vFile.IsHolding<SdfAssetPath>()) {
            path = vFile.Get<SdfAssetPath>().GetResolvedPath();
            if (path.empty()) {
                path = vFile.Get<SdfAssetPath>().GetAssetPath();
            }
        }
        else if (vFile.IsHolding<std::string>()) {
            path = vFile.Get<std::string>();
        }
        texturePaths[textureNodeName.GetString()] = path;

        // Load the texture immediately
        if (!pxr::HioImage::IsSupportedImageFile(path)) {
            TF_WARN(
                "Texture '%s': unsupported file format '%s'.",
                textureNodeName.GetString().c_str(),
                path.c_str());
            continue;
        }

        HioImageSharedPtr image = pxr::HioImage::OpenForReading(path);
        if (!image) {
            TF_WARN(
                "Texture '%s': failed to load image from file '%s'.",
                textureNodeName.GetString().c_str(),
                path.c_str());
            continue;
        }

        textureResources[textureNodeName.GetString()].filePath = path;
        textureResources[textureNodeName.GetString()].image = image;
    }
}

void Hd_USTC_CG_Material::MtlxGenerateShader(
    MaterialX::DocumentPtr mtlx_document,
    HdMaterialNetwork2Interface netInterface,
    HdMtlxTexturePrimvarData& hdMtlxData)
{
    _UpdateTextureNodes(
        &netInterface, hdMtlxData.hdTextureNodes, mtlx_document);

    auto renderable = findRenderableElements(mtlx_document);

    _FixOmittedConnections(mtlx_document, renderable);
    using namespace mx;

    auto element = renderable[0];

    std::string elementName(element->getNamePath());
    elementName = mx::createValidName(elementName);

    ShaderGenerator& shader_generator_ =
        shader_gen_context_->getShaderGenerator();
    {
        std::lock_guard lock(shadergen_mutex);
        auto shader = shader_generator_.generate(
            elementName, element, *shader_gen_context_);
        shader_source = shader->getSourceCode(mx::Stage::PIXEL);

        BindlessContextPtr context =
            shader_gen_context_->getUserData<BindlessContext>(
                mx::HW::USER_DATA_BINDING_CONTEXT);
        get_data_code = context->get_data_code();
    }
}

HdMaterialNetwork2Interface Hd_USTC_CG_Material::FetchMaterialNetwork(
    HdSceneDelegate* sceneDelegate,
    HdMaterialNetwork2& hdNetwork,
    SdfPath& materialPath,
    SdfPath& surfTerminalPath,
    HdMaterialNode2 const*& surfTerminal)
{
    VtValue material = sceneDelegate->GetMaterialResource(GetId());
    HdMaterialNetworkMap networkMap = material.Get<HdMaterialNetworkMap>();

    bool isVolume;
    hdNetwork = HdConvertToHdMaterialNetwork2(networkMap, &isVolume);

    materialPath = GetId();

    auto netInterface = HdMaterialNetwork2Interface(materialPath, &hdNetwork);
    _FixNodeTypes(&netInterface);
    _FixNodeValues(&netInterface);

    const TfToken& terminalNodeName = HdMaterialTerminalTokens->surface;

    surfTerminal =
        _GetTerminalNode(hdNetwork, terminalNodeName, &surfTerminalPath);
    return netInterface;
}
void Hd_USTC_CG_Material::BuildGPUTextures(Hd_USTC_CG_RenderParam* render_param)
{
    auto descriptor_table =
        render_param->InstanceCollection->get_descriptor_table();

    for (auto& texture_resource : textureResources) {
        // Create a thread for asynchronous processing
        std::thread texture_thread(
            [&texture_resource, this, descriptor_table, render_param]() {
                auto device = RHI::get_device();

                auto image = texture_resource.second.image;

                nvrhi::TextureDesc desc;
                desc.width = image->GetWidth();
                desc.height = image->GetHeight();
                desc.format = RHI::ConvertFromHioFormat(image->GetFormat());

                desc.initialState = nvrhi::ResourceStates::ShaderResource;
                desc.isRenderTarget = false;

                texture_resource.second.texture = device->createTexture(desc);

                auto texture_name =
                    std::filesystem::path(texture_resource.first)
                        .filename()
                        .string();

                auto storage_byte_size = image->GetBytesPerPixel();
                if (image->GetFormat() == HioFormatUNorm8Vec3srgb)
                    storage_byte_size = 4;

                std::vector<uint8_t> data(
                    image->GetWidth() * image->GetHeight() * storage_byte_size,
                    0);

                HioImage::StorageSpec storageSpec;
                storageSpec.width = image->GetWidth();
                storageSpec.height = image->GetHeight();
                storageSpec.format = image->GetFormat();
                storageSpec.flipped = false;
                storageSpec.data = data.data();

                // Read the image data asynchronously
                texture_resource.second.image->Read(storageSpec);

                {
                    std::lock_guard lock(texture_mutex);
                    auto [gpu_texture, staging] =
                        RHI::load_texture(desc, storageSpec.data);
                    texture_resource.second.texture = gpu_texture;
                }

                texture_resource.second.descriptor =
                    descriptor_table->CreateDescriptorHandle(
                        nvrhi::BindingSetItem::Texture_SRV(
                            0, texture_resource.second.texture, desc.format));

                if (texture_resource.second.texture) {
                    auto texture_id = texture_resource.second.descriptor.Get();

                    // Replace the "$"+ texture_name+"_id" with the actual
                    // texture_id in the get_data_code string
                    std::string to_replace = "$" + texture_name + "_file_id";
                    std::string replace_with = std::to_string(texture_id);

                    size_t pos = get_data_code.find(to_replace);
                    if (pos != std::string::npos) {
                        get_data_code.replace(
                            pos, to_replace.length(), replace_with);
                    }
                }
            });

        // Add the thread to the render_param for tracking
        render_param->texture_loading_threads.push_back(
            std::move(texture_thread));
    }
}

void Hd_USTC_CG_Material::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    auto param = static_cast<Hd_USTC_CG_RenderParam*>(renderParam);

    ensure_material_data_handle(param);

    HdMaterialNetwork2 hdNetwork;
    SdfPath materialPath;

    SdfPath surfTerminalPath;
    HdMaterialNode2 const* surfTerminal;
    HdMaterialNetwork2Interface netInterface = FetchMaterialNetwork(
        sceneDelegate, hdNetwork, materialPath, surfTerminalPath, surfTerminal);

    HdMtlxTexturePrimvarData hdMtlxData;

    MaterialX::DocumentPtr mtlx_document =
        HdMtlxCreateMtlxDocumentFromHdNetworkFast(
            hdNetwork,
            *surfTerminal,
            surfTerminalPath,
            materialPath,
            libraries,
            &hdMtlxData);
    assert(mtlx_document);
    CollectTextures(netInterface, hdMtlxData);

    MtlxGenerateShader(mtlx_document, netInterface, hdMtlxData);

    BuildGPUTextures(param);

    *dirtyBits = HdChangeTracker::Clean;
}

HdDirtyBits Hd_USTC_CG_Material::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::AllDirty;
}

void Hd_USTC_CG_Material::Finalize(HdRenderParam* renderParam)
{
    HdMaterial::Finalize(renderParam);
}

std::vector<TextureHandle> Hd_USTC_CG_Material::GetTextures() const
{
    std::vector<TextureHandle> textures;
    for (const auto& tex : textureResources) {
        textures.push_back(tex.second.texture);
    }
    return textures;
}

void Hd_USTC_CG_Material::ensure_material_data_handle(
    Hd_USTC_CG_RenderParam* render_param)
{
    std::lock_guard<std::mutex> lock(material_data_handle_mutex);
    if (!material_data_handle) {
        if (!render_param) {
            throw std::runtime_error("Render param is null.");
        }
        material_data_handle =
            render_param->InstanceCollection->material_pool.allocate(1);
    }
}

unsigned Hd_USTC_CG_Material::GetMaterialLocation() const
{
    if (!material_data_handle) {
        return -1;
    }
    return material_data_handle->index();
}

// HLSL callable shader
static std::string slang_source_code = R"(
// Simple callable shader that returns a red color

ConstantBuffer<float> cb : register(b0);

struct CallableData
{
    float4 color;
};

[shader("callable")]
void getColor(inout CallableData data)
{
        float4 color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    // Set the payload
        data.color = color;

    return;
}

)";

void Hd_USTC_CG_Material::ensure_shader_compiled(const ShaderFactory& factory)
{
    if (!program) {
        if (!shader_source.empty()) {
            ProgramDesc mtlx_desc;
            mtlx_desc.set_source_code(shader_source);
            auto mtlx_program = factory.createProgram(mtlx_desc);
        }

        // replace the $BindlessDataLoading in shader_source with the
        // get_data_code
        std::string to_replace = "$BindlessDataLoading";
        size_t pos = shader_source.find(to_replace);
        if (pos != std::string::npos) {
            shader_source.replace(pos, to_replace.length(), get_data_code);
        }
#ifndef _NDEBUG
        std::ofstream out(
            "generated_shaders/" + std::to_string(this->GetMaterialLocation()) +
            ".slang");
        out << shader_source;
        out.close();
#endif

        ProgramDesc desc;
        desc.set_source_code(slang_source_code).set_entry_name("getColor");

        program = factory.createProgram(desc);
    }
}

std::shared_ptr<ProgramVars> Hd_USTC_CG_Material::GetShader(
    const ShaderFactory& factory,
    ResourceAllocator& allocator)
{
    ensure_shader_compiled(factory);
    return std::make_shared<ProgramVars>(allocator, program);
}

USTC_CG_NAMESPACE_CLOSE_SCOPE