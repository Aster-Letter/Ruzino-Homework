//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "SOurceCodeNodeSlang.h"
#include "api.h"

MATERIALX_NAMESPACE_BEGIN

/// @class ClosureSourceCodeNodeSlang
/// Implemention for a closure node using data-driven static source code.
class HD_USTC_CG_API ClosureSourceCodeNodeSlang : public SourceCodeNodeSlang {
   public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END
