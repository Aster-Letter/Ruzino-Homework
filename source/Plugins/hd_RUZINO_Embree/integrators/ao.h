#pragma once

#include <random>

#include "../integrator.h"
#include "../renderParam.h"
#include "../renderer.h"
#include "pxr/base/gf/ray.h"
#include "pxr/pxr.h"

RUZINO_NAMESPACE_OPEN_SCOPE
class SurfaceInteraction;
using namespace pxr;
class AOIntegrator : public SamplingIntegrator {
   public:
    AOIntegrator(
        const Hd_RUZINO_Camera* camera,
        Hd_RUZINO_RenderBuffer* render_buffer,
        HdRenderThread* render_thread)
        : SamplingIntegrator(camera, render_buffer, render_thread)
    {
    }

   protected:
    VtValue Li(const GfRay& ray, std::default_random_engine& uniform_float)
        override;
};

RUZINO_NAMESPACE_CLOSE_SCOPE
