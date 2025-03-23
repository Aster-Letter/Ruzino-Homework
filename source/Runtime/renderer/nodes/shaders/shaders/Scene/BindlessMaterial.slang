#pragma once
#include "SceneDefines.slangh"
#include "utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "utils/Math/PackedFormats.h"
using uint = uint32_t;
#endif

struct MaterialDataBlob {
    uint data[64];
};

#ifndef HOST_CODE
StructuredBuffer<MaterialDataBlob> materialBuffer;

[[vk::binding(1, 1)]]
Sampler2D t_BindlessTextures[] : register(t0, space2);

#endif