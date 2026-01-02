#pragma once
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/materialNetwork2Interface.h>
#include <pxr/imaging/hdMtlx/hdMtlx.h>

#include "MaterialXFormat/Util.h"
#include "material.h"
#include "pxr/imaging/hd/sceneDelegate.h"
#include "pxr/usd/sdr/registry.h"
RUZINO_NAMESPACE_OPEN_SCOPE
namespace mx = MaterialX;

TfToken _FixSingleType(TfToken const& nodeType);

bool _FindGraphAndNodeByName(
    mx::DocumentPtr const& mxDoc,
    std::string const& mxNodeGraphName,
    std::string const& mxNodeName,
    mx::NodeGraphPtr* mxNodeGraph,
    mx::NodePtr* mxNode);

// Get the Hydra equivalent for the given MaterialX input value
TfToken _GetHdWrapString(
    TfToken const& hdTextureNodeName,
    std::string const& mxInputValue);

void _GetWrapModes(
    HdMaterialNetworkInterface* netInterface,
    TfToken const& hdTextureNodeName,
    TfToken* uWrap,
    TfToken* vWrap);

void _FixOmittedConnections(
    mx::DocumentPtr const& mxDoc,
    const std::vector<mx::TypedElementPtr>& renderables);
// Returns true is the given mtlxSdrNode requires primvar support for texture
// coordinates
bool _NodeHasTextureCoordPrimvar(
    mx::DocumentPtr const& mxDoc,
    const SdrShaderNodeConstPtr mtlxSdrNode);

TfToken _GetColorSpace(
    HdMaterialNetworkInterface* netInterface,
#if PXR_VERSION >= 2402
    TfToken const& hdTextureNodeName,
    HdMaterialNetworkInterface::NodeParamData paramData)
#else
    TfToken const& hdTextureNodeName)
#endif
    ;

void _UpdateTextureNodes(
    HdMaterialNetworkInterface* netInterface,
    std::set<SdfPath> const& hdTextureNodePaths,
    mx::DocumentPtr const& mxDoc);

void _FixNodeTypes(HdMaterialNetwork2Interface* netInterface);

void _FixNodeValues(HdMaterialNetwork2Interface* netInterface);
HdMaterialNode2 const* _GetTerminalNode(
    HdMaterialNetwork2 const& network,
    TfToken const& terminalName,
    SdfPath* terminalNodePath);
RUZINO_NAMESPACE_CLOSE_SCOPE