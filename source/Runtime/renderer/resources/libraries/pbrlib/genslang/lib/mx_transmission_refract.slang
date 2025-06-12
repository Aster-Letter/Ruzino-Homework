import pbrlib.genslang.lib.mx_environment_prefilter;

import pbrlib.genslang.lib.mx_microfacet_specular;

ConstantBuffer<uint> u_refractionTwoSided;

float3 mx_surface_transmission(
    float3 N,
    float3 V,
    float3 L,
    float3 X,
    float2 alpha,
    int distribution,
    FresnelData fd,
    float3 tint)
{
    // Compute proper microfacet transmission
    float3 Y = cross(N, X);

    float NdotL = clamp(abs(dot(N, L)), M_FLOAT_EPS, 1.0);
    float NdotV =
        clamp(dot(N, V), M_FLOAT_EPS, 1.0);  // Get IOR from fresnel data
    float eta = fd.ior.x;

    // For transmission, compute the appropriate half vector
    // The half vector is the normal that would refract V to L
    float3 H = normalize(V + L * eta);

    // Ensure half vector points toward the incident side
    if (dot(H, N) < 0.0) {
        H = -H;
    }

    float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);
    float LdotH = clamp(abs(dot(L, H)), M_FLOAT_EPS, 1.0);
    float3 Ht = float3(
        dot(H, X),
        dot(H, Y),
        dot(H, N));  // Compute transmission fresnel coefficient
    float3 F = mx_compute_fresnel(VdotH, fd);
    float3 T_fresnel = float3(1.0) - F;

    // Compute microfacet distribution
    float D = mx_ggx_NDF(Ht, alpha);

    // Compute masking-shadowing function
    float avgAlpha = mx_average_alpha(alpha);
    float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

    // Compute Jacobian for the transformation from half-angle to light
    // direction
    float denom = abs(VdotH) + abs(LdotH) / eta;
    float jacobian = (eta * eta * abs(LdotH)) / (denom * denom);

    // Final transmission BSDF with proper energy conservation
    float3 transmission = T_fresnel * D * G * jacobian * tint;

    return transmission / (4.0 * NdotV);
}
