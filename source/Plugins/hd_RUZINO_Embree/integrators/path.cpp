#include "path.h"

#include <algorithm>
#include <random>

#include "../surfaceInteraction.h"
RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;

VtValue PathIntegrator::Li(const GfRay& ray, std::default_random_engine& random)
{
    std::uniform_real_distribution<float> uniform_dist(
        0.0f, 1.0f - std::numeric_limits<float>::epsilon());
    std::function<float()> uniform_float = std::bind(uniform_dist, random);

    auto color = EstimateOutGoingRadiance(ray, uniform_float, 0);

    return VtValue(GfVec3f(color[0], color[1], color[2]));
}

GfVec3f PathIntegrator::EstimateOutGoingRadiance(
    const GfRay& ray,
    const std::function<float()>& uniform_float,
    int recursion_depth)
{
    if (recursion_depth >= 50) {
        return {};
    }

    SurfaceInteraction si;
    if (!Intersect(ray, si)) {
        if (recursion_depth > 0) {
            GfVec3f light_hit_pos;
            auto light_radiance = IntersectLights(ray, light_hit_pos);
            if (GfDot(light_radiance, light_radiance) > 0.0f) {
                return light_radiance;
            }
        }
        return IntersectDomeLight(ray);
    }

    // This can be customized : Do we want to see the lights? (Other than dome
    // lights?)
    if (recursion_depth == 0) {
    }

    // Flip the normal if opposite
    if (GfDot(si.shadingNormal, ray.GetDirection()) > 0) {
        si.flipNormal();
        si.PrepareTransforms();
    }

    GfVec3f color{ 0 };
    GfVec3f directLight = EstimateDirectLight(si, uniform_float);

    GfVec3f globalLight = GfVec3f{ 0.f };

    GfVec3f sampled_dir;
    float sampled_pdf = 0.0f;
    GfVec3f sampled_brdf = si.Sample(sampled_dir, sampled_pdf, uniform_float);

    float cos_theta = abs(GfDot(si.shadingNormal, sampled_dir));
    if (sampled_pdf > 1e-8f && cos_theta > 1e-8f) {
        GfVec3f path_weight = sampled_brdf * (cos_theta / sampled_pdf);
        float survive_prob = std::clamp(
            std::max(path_weight[0], std::max(path_weight[1], path_weight[2])),
            0.05f,
            0.95f);

        if (uniform_float() <= survive_prob) {
            float offset_sign = GfDot(sampled_dir, si.geometricNormal) >= 0.0f ? 1.0f : -1.0f;
            GfRay bounce_ray(
                si.position + si.geometricNormal * (0.0001f * offset_sign),
                sampled_dir);
            GfVec3f incoming_radiance = EstimateOutGoingRadiance(
                bounce_ray, uniform_float, recursion_depth + 1);
            globalLight = GfCompMult(path_weight, incoming_radiance) / survive_prob;
        }
    }

    color = directLight + globalLight;

    return color;
}

RUZINO_NAMESPACE_CLOSE_SCOPE
