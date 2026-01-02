#pragma once
#include "../api.h"
#include "../integrator.h"
#include "pxr/pxr.h"

RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;
class PathIntegrator : public SamplingIntegrator {
   public:
    PathIntegrator(
        const Hd_RUZINO_Camera* camera,
        Hd_RUZINO_RenderBuffer* render_buffer,
        HdRenderThread* render_thread)
        : SamplingIntegrator(camera, render_buffer, render_thread)
    {
    }

   protected:
    VtValue Li(const GfRay& ray, std::default_random_engine& uniform_float)
        override;

    GfVec3f EstimateOutGoingRadiance(
        const GfRay& ray,
        const std::function<float()>& uniform_float,
        int recursion_depth);
};

RUZINO_NAMESPACE_CLOSE_SCOPE
