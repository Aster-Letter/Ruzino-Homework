import Scene.SceneTypes;
import utils.Spectrum;

struct FetchCallableData {
    uint materialBlobID;
    uint material_params_index; // Index into material parameters buffer, set by data fetch callable
    uint shader_type_id;
    VertexInfo vertexInfo;
};

struct EvalCallableData {
    RGBSpectrum color;
    RGBSpectrum throughput;
    float3 L;
    float3 V;
    float3 sampledDir;
    float pdf;
    uint seed;
    uint eta_flipped;
    uint material_params_index; // Index into material parameters buffer, set by data fetch callable
    VertexInfo vertexInfo;
};

struct DomeLightCallableData {
    float3 rayDirection;     // Ray direction in world space
    float3 localDirection;   // Ray direction in dome light local space
    RGBSpectrum envColor;    // Output environment color
    float3 intensity;        // Dome light intensity
    int textureIndex;        // Bindless texture index
    uint seed;               // Random seed for sampling
};
