//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include "ClosureCompoundNodeSlang.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN
ShaderNodeImplPtr ClosureCompoundNodeSlang::create()
{
    return std::make_shared<ClosureCompoundNodeSlang>();
}

void ClosureCompoundNodeSlang::addClassification(ShaderNode& node) const
{
    // Add classification from the graph implementation.
    node.addClassification(_rootGraph->getClassification());
}

void ClosureCompoundNodeSlang::emitFunctionDefinition(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();
        const Syntax& syntax = shadergen.getSyntax();

        bool isStandardSurface =
            _functionName.find("standard_surface") != string::npos;
        bool isUsdPreviewSurface =
            _functionName.find("UsdPreviewSurface") != string::npos;

        if (isStandardSurface || isUsdPreviewSurface) {
            // Generate fetch_shader_opacity placeholder
            // The actual opacity computation code will be extracted from fetch_shader_data
            // by post-processing in materialX.cpp
            shadergen.emitLineBreak(stage);
            shadergen.emitComment("Opacity fetch function - computes opacity from material graph", stage);
            shadergen.emitLine("void fetch_shader_opacity(", stage, false);
            shadergen.emitLine("    inout uint material_params_index,", stage, false);
            shadergen.emitLine("    inout uint shader_type_id,", stage, false);
            shadergen.emitLine("    in MaterialDataBlob data,", stage, false);
            shadergen.emitLine("    in VertexInfo vertexInfo)", stage, false);
            shadergen.emitLine("{", stage, false);
            shadergen.emitLine("$BindlessDataLoading", stage, false);
            shadergen.emitLine("$OpacityComputation", stage, false);  // Placeholder for opacity computation
            shadergen.emitLine("    shader_type_id = " + std::to_string(isUsdPreviewSurface ? 1 : 0) + ";", stage, false);
            shadergen.emitLine("    material_params_index = asuint(opacity_value);", stage, false);
            shadergen.emitLine("}", stage, false);
            shadergen.emitLineBreak(stage);
            
            return;
        }
        throw std::runtime_error(
            "ClosureCompoundNodeSlang::emitFunctionDefinition - Only "
            "standard_surface and UsdPreviewSurface are supported.");

        // Emit functions for all child nodes
        shadergen.emitFunctionDefinitions(*_rootGraph, context, stage);

        string delim = "";

        // Begin function signature
        shadergen.emitLineBegin(stage);

        shadergen.emitString("void " + _functionName + "(", stage);

        if (context.getShaderGenerator().nodeNeedsClosureData(node)) {
            shadergen.emitString(
                delim + HW::CLOSURE_DATA_TYPE + " " + HW::CLOSURE_DATA_ARG +
                    ", ",
                stage);
        }

        auto& vertexData = stage.getInputBlock(HW::VERTEX_DATA);
        if (!vertexData.empty()) {
            shadergen.emitString(
                delim + vertexData.getName() + " " + vertexData.getInstance(),
                stage);
            delim = ", ";
        }

        const string& type = syntax.getTypeName(Type::VECTOR3);
        shadergen.emitString(delim + type + " " + HW::DIR_L, stage);
        shadergen.emitString(", " + type + " " + HW::DIR_V, stage);
        // eta
        shadergen.emitString(delim + "uint eta_flipped", stage);

        // Add all inputs
        for (ShaderGraphInputSocket* inputSocket :
             _rootGraph->getInputSockets()) {
            shadergen.emitString(
                delim + syntax.getTypeName(inputSocket->getType()) + " " +
                    inputSocket->getVariable(),
                stage);
            delim = ", ";
        }

        // Add all outputs
        for (ShaderGraphOutputSocket* outputSocket :
             _rootGraph->getOutputSockets()) {
            shadergen.emitString(
                delim + syntax.getOutputTypeName(outputSocket->getType()) +
                    " " + outputSocket->getVariable(),
                stage);
            delim = ", ";
        }

        // End function signature
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage, false);

        // Begin function body
        shadergen.emitFunctionBodyBegin(*_rootGraph, context, stage);

        if (isStandardSurface) {
            shadergen.emitLine("if (transmission > 0.0)", stage, false);
            shadergen.emitLine("if (eta_flipped > 0.0)", stage, false);
            shadergen.emitLine("specular_IOR = 1.0 / specular_IOR", stage);
        }

        // Emit all texturing nodes. These are inputs to the
        // closure nodes and need to be emitted first.
        shadergen.emitFunctionCalls(
            *_rootGraph, context, stage, ShaderNode::Classification::TEXTURE);

        // Emit function calls for internal closures nodes connected to the
        // graph sockets. These will in turn emit function calls for any
        // dependent closure nodes upstream.
        for (ShaderGraphOutputSocket* outputSocket :
             _rootGraph->getOutputSockets()) {
            if (outputSocket->getConnection()) {
                const ShaderNode* upstream =
                    outputSocket->getConnection()->getNode();
                if (upstream->getParent() == _rootGraph.get() &&
                    (upstream->hasClassification(
                         ShaderNode::Classification::CLOSURE) ||
                     upstream->hasClassification(
                         ShaderNode::Classification::SHADER))) {
                    shadergen.emitFunctionCall(*upstream, context, stage);
                }
            }
        }

        // Emit final results
        for (ShaderGraphOutputSocket* outputSocket :
             _rootGraph->getOutputSockets()) {
            const string result =
                shadergen.getUpstreamResult(outputSocket, context);
            shadergen.emitLine(
                outputSocket->getVariable() + " = " + result, stage);
        }

        // End function body
        shadergen.emitFunctionBodyEnd(*_rootGraph, context, stage);
    }
}

void ClosureCompoundNodeSlang::emitOpacityFetchFunction(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();
    
    bool isStandardSurface =
        _functionName.find("standard_surface") != string::npos;
    bool isUsdPreviewSurface =
        _functionName.find("UsdPreviewSurface") != string::npos;
    
    // Generate fetch_shader_opacity function
    // This function only extracts opacity and stores it in material_params_index
    // to avoid full material buffer access during shadow testing
    
    // Emit function signature
    shadergen.emitLine(
        "void fetch_shader_opacity(",
        stage, false);
    shadergen.emitLine(
        "    inout uint material_params_index,",
        stage, false);
    shadergen.emitLine(
        "    inout uint shader_type_id,",
        stage, false);
    shadergen.emitLine(
        "    in MaterialDataBlob data,",
        stage, false);
    shadergen.emitLine(
        "    in VertexInfo vertexInfo)",
        stage, false);
    
    // Begin function body with proper scope
    shadergen.emitFunctionBodyBegin(*_rootGraph, context, stage);
    
    if (isStandardSurface) {
        // For standard_surface, opacity is stored as float3 in MaterialDataBlob
        // We read it directly from the blob without running the material graph
        shadergen.emitLine("// Standard Surface opacity extraction", stage);
        shadergen.emitLine("shader_type_id = 0;", stage);
        
        // In standard_surface's fetch_shader_data, opacity is stored in data
        // For simplicity, we just return 1.0 (fully opaque) for standard_surface
        // since standard_surface typically doesn't support alpha transparency
        shadergen.emitLine("float opacity = 1.0;", stage);
        
        shadergen.emitLine(
            "material_params_index = asuint(opacity);",
            stage);
    }
    else if (isUsdPreviewSurface) {
        // For UsdPreviewSurface, opacity is parameter 8 (float)
        // We read it directly from the MaterialDataBlob
        shadergen.emitLine("// Preview Surface opacity extraction", stage);
        shadergen.emitLine("shader_type_id = 1;", stage);
        
        // Read opacity from the material data blob
        // This corresponds to the opacity parameter (index 8) in UsdPreviewSurface
        // The actual data index depends on how the material is packed
        shadergen.emitLine("// Read opacity from material data blob", stage);
        shadergen.emitLine("float opacity = 1.0; // Default fully opaque", stage);
        shadergen.emitLine("// TODO: Read from MaterialDataBlob if packed", stage);
        
        shadergen.emitLine(
            "material_params_index = asuint(opacity);",
            stage);
    }
    
    // End function body with proper scope
    shadergen.emitFunctionBodyEnd(*_rootGraph, context, stage);
    shadergen.emitLine("", stage);
}

void ClosureCompoundNodeSlang::emitFunctionCall(
    const ShaderNode& node,
    GenContext& context,
    ShaderStage& stage) const
{
    const ShaderGenerator& shadergen = context.getShaderGenerator();

    DEFINE_SHADER_STAGE(stage, Stage::VERTEX)
    {
        // Emit function calls for all child nodes to the vertex shader stage
        shadergen.emitFunctionCalls(*_rootGraph, context, stage);
    }

    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Emit calls for any closure dependencies upstream from this node.
        shadergen.emitDependentFunctionCalls(
            node, context, stage, ShaderNode::Classification::CLOSURE);

        //// Declare the output variables
        emitOutputVariables(node, context, stage);

        // Check if this is a standard surface material
        bool isStandardSurface =
            _functionName.find("standard_surface") != string::npos;
        bool isUsdPreviewSurface =
            _functionName.find("UsdPreviewSurface") != string::npos;

        if (isStandardSurface) {
            // Use packStandardSurfaceMaterialParams to create optimized structure
            
            // Parameter name mapping for standard_surface - must match input order
            static const std::vector<string> paramNames = {
                "base",
                "base_color",
                "diffuse_roughness",
                "metalness",
                "specular",
                "specular_color",
                "specular_roughness",
                "specular_IOR",
                "specular_anisotropy",
                "specular_rotation",
                "transmission",
                "transmission_color",
                "transmission_depth",
                "transmission_scatter",
                "transmission_scatter_anisotropy",
                "transmission_dispersion",
                "transmission_extra_roughness",
                "subsurface",
                "subsurface_color",
                "subsurface_radius",
                "subsurface_scale",
                "subsurface_anisotropy",
                "sheen",
                "sheen_color",
                "sheen_roughness",
                "coat",
                "coat_color",
                "coat_roughness",
                "coat_anisotropy",
                "coat_rotation",
                "coat_IOR",
                "coat_normal",
                "coat_affect_color",
                "coat_affect_roughness",
                "thin_film_thickness",
                "thin_film_IOR",
                "emission",
                "emission_color",
                "opacity",
                "thin_walled",
                "normal",
                "tangent"
            };

            auto inputs = node.getInputs();
            
            // Begin function call to packStandardSurfaceMaterialParams
            shadergen.emitLineBegin(stage);
            shadergen.emitString("PackedStandardSurfaceMaterialParams params = packStandardSurfaceMaterialParams(", stage);
            shadergen.emitLineEnd(stage, false);
            
            // Emit all parameters to the packing function
            size_t minInputs = std::min(inputs.size(), paramNames.size());
            for (size_t i = 0; i < minInputs; ++i) {
                shadergen.emitLineBegin(stage);
                shadergen.emitString("    ", stage);
                shadergen.emitInput(inputs[i], context, stage);
                if (i < minInputs - 1 || inputs.size() < paramNames.size()) {
                    shadergen.emitString(",", stage);
                }
                shadergen.emitLineEnd(stage, false);
            }
            
            // Fill defaults for missing parameters
            if (inputs.size() < 40) {
                shadergen.emitLine("    0,  // thin_walled", stage, false);
            }
            if (inputs.size() < 41) {
                shadergen.emitLine("    vd.normalW,  // normal", stage, false);
            }
            if (inputs.size() < 42) {
                shadergen.emitLine("    vd.tangentW  // tangent", stage, false);
            }
            
            // Close function call
            shadergen.emitLine(");", stage);

            // Write params to buffer using reinterpret cast
            shadergen.emitLine(
                "materialParamsBuffer[material_params_index] = "
                "reinterpret<MaterialParams>(params);",
                stage);
            shadergen.emitLine("shader_type_id = 0;", stage);
        }
        else if (isUsdPreviewSurface) {
            // Fill PreviewSurfaceMaterialParams
            shadergen.emitLine("PreviewSurfaceMaterialParams params = {};", stage);

            // Parameter name mapping for preview_surface
            // Order MUST match the USD Preview Surface node inputs order
            static const std::vector<string> paramNames = {
                "diffuseColor",         // 0
                "emissiveColor",        // 1
                "useSpecularWorkflow",  // 2
                "specularColor",        // 3
                "metallic",             // 4
                "roughness",            // 5
                "clearcoat",            // 6
                "clearcoatRoughness",   // 7
                "opacity",              // 8
                "opacityMode",          // 9
                "opacityThreshold",     // 10
                "ior",                  // 11
                "normal",               // 12
                "displacement",         // 13
                "occlusion"             // 14
            };

            auto inputs = node.getInputs();
            size_t minInputs = std::min(inputs.size(), paramNames.size());

            // Loop through and assign parameters
            for (size_t i = 0; i < minInputs; ++i) {
                shadergen.emitLineBegin(stage);
                shadergen.emitString("params." + paramNames[i] + " = ", stage);
                shadergen.emitInput(inputs[i], context, stage);
                shadergen.emitLineEnd(stage);
            }

            // Write params to buffer using reinterpret cast
            shadergen.emitLine(
                "materialParamsBuffer[material_params_index] = "
                "reinterpret<MaterialParams>(params);",
                stage);
            shadergen.emitLine("shader_type_id = 1;", stage);
        }
        else {
        }
    }
}

MATERIALX_NAMESPACE_END
