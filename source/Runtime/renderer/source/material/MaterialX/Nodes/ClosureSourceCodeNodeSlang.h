//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_CLOSURESOURCECODENODE_H
#define MATERIALX_CLOSURESOURCECODENODE_H

#include "SourceCodeNodeSlang.h"
MATERIALX_NAMESPACE_BEGIN

/// @class ClosureSourceCodeNode
/// Implementation for a closure node using data-driven static source code.
class HD_RUZINO_API ClosureSourceCodeNodeSlang : public SourceCodeNodeSlang {
   public:
    static ShaderNodeImplPtr create();

    void emitFunctionCall(
        const ShaderNode& node,
        GenContext& context,
        ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
