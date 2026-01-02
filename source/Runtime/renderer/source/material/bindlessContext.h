#pragma once
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include <filesystem>

#include "MaterialX/SlangShaderGenerator.h"
#include "MaterialXGenShader/Shader.h"
#include "material.h"

RUZINO_NAMESPACE_OPEN_SCOPE
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
        fetch_data = "\n VertexData vd; \n";
        data_location = 0;
    }

    void emitDirectives(GenContext& context, ShaderStage& stage) override
    {
        const ShaderGenerator& generator = context.getShaderGenerator();
        generator.emitLine("import Scene.BindlessMaterial", stage);
        generator.emitLine("import Scene.VertexInfo", stage);
    }

    // Emit uniforms with binding information
    void emitResourceBindings(
        GenContext& context,
        const VariableBlock& resources,
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

    MaterialDataBlob& get_material_data()
    {
        return material_data;
    }

   private:
    std::string fetch_data = "";
    unsigned int data_location = 0;
    MaterialDataBlob material_data;
};

RUZINO_NAMESPACE_CLOSE_SCOPE
