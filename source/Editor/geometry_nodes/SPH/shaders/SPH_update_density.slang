// density_hlsl
#include "utils.slangh"

ConstantBuffer<SPHConstants> sph_constants : register(b0);
StructuredBuffer<float3> positions : register(t0);
StructuredBuffer<uint2> ContactPairs : register(t1);
RWByteAddressBuffer rho : register(u0);

// pair_count
[numthreads(32, 1, 1)]
void main(int2 i_globalIdx: SV_DispatchThreadID)
{
    if (i_globalIdx.x >= sph_constants.numPairs)
        return;
    float mass = sph_constants.mV;  // mV stores particle mass
    float rho0 = sph_constants.restDensity;
    float h = sph_constants.smoothingRadius;

    uint u1 = ContactPairs[i_globalIdx.x].x;
    uint u2 = ContactPairs[i_globalIdx.x].y;
    float3 r = positions[u1] - positions[u2];

    // Density contribution: ρᵢ += m * W(rᵢⱼ, h)
    float Drho = mass * cubic_kernel(r, h);
    uint iDrho = uint(Drho * INT_SCALE);
    rho.InterlockedAdd(u1 * 4, iDrho);
    rho.InterlockedAdd(u2 * 4, iDrho);
}


