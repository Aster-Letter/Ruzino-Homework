// Instancer compute shader for GPU-based instance transform computation

import Scene.SceneTypes;

struct InstanceParams {
    float4x4 parent_transform;
    float4x4 prototype_transform;
    uint material_id;
    uint geometry_id;
    uint instance_count;
    uint base_instance_index;
    uint64_t BLAS_address;
};

cbuffer g_Params : register(b0) {
    InstanceParams params;
}

StructuredBuffer<int> g_InstanceIndices : register(t0);

#ifdef HAS_TRANSLATIONS
StructuredBuffer<float3> g_Translations : register(t1);
#endif

#ifdef HAS_ROTATIONS
StructuredBuffer<float4> g_Rotations : register(t2);
#endif

#ifdef HAS_SCALES
StructuredBuffer<float3> g_Scales : register(t3);
#endif

#ifdef HAS_INSTANCE_TRANSFORMS
StructuredBuffer<float4x4> g_InstanceTransforms : register(t4);
#endif

RWStructuredBuffer<GeometryInstanceData> g_GeometryInstances : register(u0);
RWByteAddressBuffer g_RTInstances : register(u1);

float4x4 makeTranslationMatrix(float3 t) {
    return float4x4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        t.x, t.y, t.z, 1
    );
}

float4x4 makeScaleMatrix(float3 s) {
    return float4x4(
        s.x, 0, 0, 0,
        0, s.y, 0, 0,
        0, 0, s.z, 0,
        0, 0, 0, 1
    );
}

float4x4 quaternionToMatrix(float4 q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;

    return float4x4(
        1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
        2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
        2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
        0, 0, 0, 1
    );
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint i = dispatchThreadID.x;
    if (i >= params.instance_count) {
        return;
    }

    int index = g_InstanceIndices[i];
    
    // Build instance transform
    float4x4 instanceTransform = params.parent_transform;

#ifdef HAS_TRANSLATIONS
    float3 translation = g_Translations[index];
    instanceTransform = mul(makeTranslationMatrix(translation), instanceTransform);
#endif

#ifdef HAS_ROTATIONS
    float4 rotation = g_Rotations[index];
    instanceTransform = mul(quaternionToMatrix(rotation), instanceTransform);
#endif

#ifdef HAS_SCALES
    float3 scale = g_Scales[index];
    instanceTransform = mul(makeScaleMatrix(scale), instanceTransform);
#endif

#ifdef HAS_INSTANCE_TRANSFORMS
    float4x4 customTransform = g_InstanceTransforms[index];
    instanceTransform = mul(customTransform, instanceTransform);
#endif

    // Apply prototype transform
    float4x4 finalTransform = mul(params.prototype_transform, instanceTransform);

    // Write geometry instance data
    uint geometryInstanceIndex = params.base_instance_index + i;
    g_GeometryInstances[geometryInstanceIndex].geometryID = params.geometry_id;
    g_GeometryInstances[geometryInstanceIndex].materialID = params.material_id;
    g_GeometryInstances[geometryInstanceIndex].transform = finalTransform;
    g_GeometryInstances[geometryInstanceIndex].flags = 0;

    // Write RT instance descriptor (simplified, needs proper format)
    // Each RT instance is 64 bytes according to DXR spec
    // Layout: transform (3x4 floats) + instanceID + mask + instanceContributionToHitGroupIndex + flags + accelerationStructure
    uint rtInstanceOffset = geometryInstanceIndex * 16; // 64 bytes / 4 = 16 DWORDs
    
    // Transpose for row-major storage
    float4x4 transposed = transpose(finalTransform);
    
    // Write transform (first 3 rows, 12 floats = 48 bytes)
    for (uint row = 0; row < 3; ++row) {
        for (uint col = 0; col < 4; ++col) {
            g_RTInstances.Store(rtInstanceOffset * 4 + (row * 4 + col) * 4, asuint(transposed[row][col]));
        }
    }
    
    // Write instanceID and mask (24-bit ID + 8-bit mask)
    g_RTInstances.Store(rtInstanceOffset * 4 + 48, geometryInstanceIndex | (1 << 24));
    
    // Write instanceContributionToHitGroupIndex (24 bits) + flags (8 bits)
    g_RTInstances.Store(rtInstanceOffset * 4 + 52, 0); // flags: TriangleFrontCounterclockwise would be 0x1
    
    // Write acceleration structure address (8 bytes)
    g_RTInstances.Store2(rtInstanceOffset * 4 + 56, uint2(params.BLAS_address & 0xFFFFFFFF, params.BLAS_address >> 32));
}
