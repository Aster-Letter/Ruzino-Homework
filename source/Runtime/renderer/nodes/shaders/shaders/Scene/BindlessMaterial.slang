struct MaterialDataBlob {
    uint data[32];
}
StructuredBuffer<MaterialDataBlob> materialBuffer;
Sampler2D t_BindlessTextures[] : register(t0, space2);

void main()
{
    MaterialDataBlob data = materialBuffer[0];

    // first 3 uints are albedo
    float3 albedo = float3(
        asfloat(data.data[0]), asfloat(data.data[1]), asfloat(data.data[2]));
}
