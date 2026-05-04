struct PathTracingConstants {
    uint lightCount;
    uint domeLightCallableIndex;  // Only valid when USE_DOME_LIGHT_CALLABLE
    uint materialFetchCallableBaseIndex;
    uint materialOpacityCallableOffset;  // Offset to add to fetch index for
                                         // opacity callables
    uint samplesPerPixel;
    uint maxDepth;
    uint reserved0;
    uint reserved1;
};

ConstantBuffer<PathTracingConstants> ptConstants;
RaytracingAccelerationStructure SceneBVH;
