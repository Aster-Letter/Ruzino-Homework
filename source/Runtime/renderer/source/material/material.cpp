#include "material.h"

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

class BinglessContext : public MaterialX::HwResourceBindingContext {
   public:
    ~BinglessContext() override;
    void initialize() override;
    void emitDirectives(mx::GenContext& context, mx::ShaderStage& stage)
        override;
    void emitResourceBindings(
        mx::GenContext& context,
        const mx::VariableBlock& uniforms,
        mx::ShaderStage& stage) override;
    void emitStructuredResourceBindings(
        mx::GenContext& context,
        const mx::VariableBlock& uniforms,
        mx::ShaderStage& stage,
        const std::string& structInstanceName,
        const std::string& arraySuffix) override;
};

BinglessContext::~BinglessContext()
{
}

void BinglessContext::initialize()
{
}

void BinglessContext::emitDirectives(
    mx::GenContext& context,
    mx::ShaderStage& stage)
{
}

void BinglessContext::emitResourceBindings(
    mx::GenContext& context,
    const mx::VariableBlock& uniforms,
    mx::ShaderStage& stage)
{
    std::string blockName = uniforms.getName();

    context.getShaderGenerator().emitLine("struct " + blockName, stage, false);
    context.getShaderGenerator().emitScopeBegin(stage);

    std::cout << "Uniform block: " << blockName << std::endl;

    for (const auto& uniform : uniforms.getVariableOrder()) {
        std::string uniformName = uniform->getName();
        std::string uniformType = uniform->getType()->getName();
        std::string uniformValue = uniform->getValueString();

        std::cout << "Uniform: " << uniformName << " " << uniformType << " "
                  << uniformValue << std::endl;
    }
    context.getShaderGenerator().emitScopeEnd(stage, true);
}

void BinglessContext::emitStructuredResourceBindings(
    mx::GenContext& context,
    const mx::VariableBlock& uniforms,
    mx::ShaderStage& stage,
    const std::string& structInstanceName,
    const std::string& arraySuffix)
{
}

namespace mx = MaterialX;

MaterialX::GenContextPtr Hd_USTC_CG_Material::shader_gen_context_ =
    std::make_shared<mx::GenContext>(mx::SlangShaderGenerator::create());
MaterialX::DocumentPtr Hd_USTC_CG_Material::libraries = mx::createDocument();

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

        //shader_gen_context_->pushUserData(
        //    mx::HW::USER_DATA_BINDING_CONTEXT,
        //    std::make_shared<BinglessContext>());
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
    }
}

void Hd_USTC_CG_Material::LoadTextures()
{
    for (const auto& tex : texturePaths) {
        const std::string& textureName = tex.first;
        const std::string& filePath = tex.second;

        if (!pxr::HioImage::IsSupportedImageFile(filePath)) {
            TF_WARN(
                "Texture '%s': unsupported file format '%s'.",
                textureName.c_str(),
                filePath.c_str());
            continue;
        }

        HioImageSharedPtr image = pxr::HioImage::OpenForReading(filePath);
        if (!image) {
            TF_WARN(
                "Texture '%s': failed to load image from file '%s'.",
                textureName.c_str(),
                filePath.c_str());
            continue;
        }

        textureResources[textureName].filePath = filePath;
        textureResources[textureName].image = image;
    }
}

void Hd_USTC_CG_Material::BuildGPUTextures(Hd_USTC_CG_RenderParam* render_param)
{
    auto descriptor_table =
        render_param->InstanceCollection->get_descriptor_table();
    auto device = RHI::get_device();

    auto command_list = device->createCommandList();

    for (auto& texture_resource : textureResources) {
        auto image = texture_resource.second.image;
        nvrhi::TextureDesc desc;
        desc.width = image->GetWidth();
        desc.height = image->GetHeight();
        desc.format = RHI::ConvertFromHioFormat(image->GetFormat());

        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.isRenderTarget = false;

        texture_resource.second.texture = device->createTexture(desc);

        auto storage_byte_size = image->GetBytesPerPixel();
        if (image->GetFormat() == HioFormatUNorm8Vec3srgb)
            storage_byte_size = 4;

        std::vector<uint8_t> data(
            image->GetWidth() * image->GetHeight() * storage_byte_size, 0);

        HioImage::StorageSpec storageSpec;

        storageSpec.width = image->GetWidth();
        storageSpec.height = image->GetHeight();
        storageSpec.format = image->GetFormat();
        storageSpec.flipped = false;
        storageSpec.data = data.data();

        texture_resource.second.image->Read(storageSpec);

        auto [gpu_texture, staging] = RHI::load_texture(desc, storageSpec.data);

        texture_resource.second.texture = gpu_texture;

        texture_resource.second.descriptor =
            descriptor_table->CreateDescriptorHandle(
                nvrhi::BindingSetItem::Texture_SRV(
                    0, texture_resource.second.texture, desc.format));
    }
}

void Hd_USTC_CG_Material::MtlxGenerateShader(
    HdMaterialNetwork2 hdNetwork,
    SdfPath materialPath,
    HdMaterialNetwork2Interface netInterface,
    SdfPath surfTerminalPath,
    HdMaterialNode2 const* surfTerminal,
    HdMtlxTexturePrimvarData& hdMtlxData)
{
    MaterialX::DocumentPtr mtlx_document =
        HdMtlxCreateMtlxDocumentFromHdNetwork(
            hdNetwork,
            *surfTerminal,
            surfTerminalPath,
            materialPath,
            libraries,
            &hdMtlxData);
    assert(mtlx_document);

    _UpdateTextureNodes(
        &netInterface, hdMtlxData.hdTextureNodes, mtlx_document);

    using namespace mx;
    auto materials = mtlx_document->getMaterialNodes();

    auto shaders = mtlx_document->getNodesOfType(SURFACE_SHADER_TYPE_STRING);

    auto renderable = mx::findRenderableElements(mtlx_document);
    auto element = renderable[0];

    mx::OutputPtr output = element->asA<mx::Output>();
    mx::NodePtr outputNode = element->asA<mx::Node>();
    if (output) {
        outputNode = output->getConnectedNode();
    }

    mx::NodeDefPtr nodeDef = outputNode->getNodeDef();

    std::string elementName(element->getNamePath());
    elementName = mx::createValidName(elementName);

    ShaderGenerator& shader_generator_ =
        shader_gen_context_->getShaderGenerator();
    shader =
        shader_generator_.generate(elementName, element, *shader_gen_context_);

#ifndef NDEBUG
    // Ensure the generated_shaders directory exists
    std::filesystem::create_directories("./generated_shaders");
    std::string shaderPath = "./generated_shaders/" + elementName + ".slang";
    std::ofstream shaderFile(shaderPath);
    if (!shaderFile) {
        TF_WARN("Failed to open shader file: %s", shaderPath.c_str());
    }
    else {
        shaderFile << shader->getSourceCode();
        shaderFile.close();
    }
    shaderFile.close();
#endif
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

void Hd_USTC_CG_Material::Sync(
    HdSceneDelegate* sceneDelegate,
    HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    HdMaterialNetwork2 hdNetwork;
    SdfPath materialPath;

    SdfPath surfTerminalPath;
    HdMaterialNode2 const* surfTerminal;
    HdMaterialNetwork2Interface netInterface = FetchMaterialNetwork(
        sceneDelegate, hdNetwork, materialPath, surfTerminalPath, surfTerminal);

    if (surfTerminal) {
        HdMtlxTexturePrimvarData hdMtlxData;
        MtlxGenerateShader(
            hdNetwork,
            materialPath,
            netInterface,
            surfTerminalPath,
            surfTerminal,
            hdMtlxData);

        CollectTextures(netInterface, hdMtlxData);
        LoadTextures();
        BuildGPUTextures(static_cast<Hd_USTC_CG_RenderParam*>(renderParam));
    }
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

USTC_CG_NAMESPACE_CLOSE_SCOPE