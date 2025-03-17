//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SOURCECODENODESLANG_H
#define MATERIALX_SOURCECODENODESLANG_H

#include <MaterialXFormat/File.h>
#include <MaterialXGenShader/ShaderNodeImpl.h>

#include "api.h"

MATERIALX_NAMESPACE_BEGIN

/// @class SourceCodeNodeSlang
/// Implemention for a node using data-driven static source code.
/// This is the default implementation used for all nodes that
/// do not have a custom ShaderNodeImpl class.
class HD_USTC_CG_API SourceCodeNodeSlang : public ShaderNodeImpl {
   public:
    static ShaderNodeImplPtr create();

    void initialize(const InterfaceElement& element, GenContext& context)
        override;
    void emitFunctionDefinition(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage) const override;
    void emitFunctionCall(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage) const override;

   protected:
    bool _inlined;
    string _functionName;
    string _functionSource;
    FilePath _sourceFilename;
};

MATERIALX_NAMESPACE_END

#endif
