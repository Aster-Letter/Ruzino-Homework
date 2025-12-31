// Initialize density with self-contribution
#include "utils.slangh"

ConstantBuffer<SPHConstants> sph_constants : register(b0);
RWByteAddressBuffer rho : register(u0);

[numthreads(32, 1, 1)]
void main(int2 i_globalIdx: SV_DispatchThreadID)
{
    if (i_globalIdx.x >= sph_constants.numParticles)
        return;
    float mass = sph_constants.mV;  // mV stores particle mass
    float h = sph_constants.smoothingRadius;
    float rho0 = sph_constants.restDensity;
    // Self-contribution to density: ρᵢ = m * W(0, h)
    float Drho = mass * cubic_kernel(float3(0, 0, 0), h);
    uint iDrho = uint(Drho * INT_SCALE);
    rho.Store(i_globalIdx.x * 4, iDrho);  // Initialize, not add
}
