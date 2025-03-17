void mx_transformmatrix_vector2M3(
    float2 val,
    float3x3 transform,
    out float2 result)
{
    float3 res = mul(float3(val, 1.0), transform);
    result = res.xy;
}
