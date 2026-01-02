//
// Copyright 2021 Pixar
//
// Licensed under the terms set forth in the LICENSE.txt file available at
// https://openusd.org/license.
//
#pragma once

#include <MaterialXCore/Library.h>

#include <memory>
#include <set>
#include <unordered_map>

#include "api.h"
#include "pxr/base/tf/token.h"
#include "pxr/imaging/hdMtlx/api.h"
#include "pxr/imaging/hdMtlx/hdMtlx.h"
#include "pxr/pxr.h"

MATERIALX_NAMESPACE_BEGIN
class FileSearchPath;
using DocumentPtr = std::shared_ptr<class Document>;
MATERIALX_NAMESPACE_END

RUZINO_NAMESPACE_OPEN_SCOPE
// Create a MaterialX Document from the given HdMaterialNetwork2
MaterialX::DocumentPtr HdMtlxCreateMtlxDocumentFromHdNetworkFast(
    pxr::HdMaterialNetwork2 const& hdNetwork,
    pxr::HdMaterialNode2 const& hdMaterialXNode,
    pxr::SdfPath const& hdMaterialXNodePath,
    pxr::SdfPath const& materialPath,
    MaterialX::DocumentPtr const& libraries,
    pxr::HdMtlxTexturePrimvarData* mxHdData);

/// Implementation that uses the material network interface.
HD_RUZINO_API
MaterialX::DocumentPtr
HdMtlxCreateMtlxDocumentFromHdMaterialNetworkInterfaceFast(
    pxr::HdMaterialNetworkInterface* netInterface,
    pxr::TfToken const& terminalNodeName,
    pxr::TfTokenVector const& terminalNodeConnectionNames,
    MaterialX::DocumentPtr const& libraries,
    pxr::HdMtlxTexturePrimvarData* mxHdData = nullptr);

RUZINO_NAMESPACE_CLOSE_SCOPE
