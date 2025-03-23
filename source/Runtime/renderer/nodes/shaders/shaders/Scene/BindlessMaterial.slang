
struct MaterialDataBlob {
    uint data[32];
}

RWStructuredBuffer<MaterialDataBlob> materialBuffer;

[[vk::binding(1, 1)]]
Sampler2D t_BindlessTextures[] : register(t0, space2);
