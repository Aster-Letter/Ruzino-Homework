#ifdef __cplusplus
#include <pxr/base/gf/vec3f.h>

struct ISpectrum {
    // virtual pxr::GfVec3f toFloat3() = 0;
};

struct RGBSpectrum : ISpectrum {
    RGBSpectrum(const pxr::GfVec3f& data) : data(data)
    {
    }
    pxr::GfVec3f data;
    float padding;
};
#else
interface ISpectrum {
    // Convert the spectrum to a float3 representation
    float3 toFloat3();
}

struct RGBSpectrum : ISpectrum {
    __init(float v)
    {
        data = float3(v, v, v);
        padding = 0;
    }

    __init(float3 rgb)
    {
        data = rgb;
        padding = 0;
    }

    float3 data;
    float padding;

    float3 toFloat3()
    {
        return data;
    }
};

// Helper functions for RGBSpectrum operations
RGBSpectrum add(RGBSpectrum a, RGBSpectrum b)
{
    return RGBSpectrum(a.data + b.data);
}

RGBSpectrum sub(RGBSpectrum a, RGBSpectrum b)
{
    return RGBSpectrum(a.data - b.data);
}

RGBSpectrum mul(RGBSpectrum a, RGBSpectrum b)
{
    return RGBSpectrum(a.data * b.data);
}

RGBSpectrum mul(RGBSpectrum a, float scalar)
{
    return RGBSpectrum(a.data * scalar);
}

RGBSpectrum mul(float scalar, RGBSpectrum a)
{
    return RGBSpectrum(a.data * scalar);
}

RGBSpectrum div(RGBSpectrum a, float scalar)
{
    return RGBSpectrum(a.data / scalar);
}

struct SampledSpectrum : ISpectrum {
    __init(float samples[30])
    {
        for (int i = 0; i < 30; ++i) {
            this.samples[i] = half(samples[i]);
        }
    }

    float3 toFloat3()
    {
        // TODO: Implement proper spectrum to XYZ to RGB conversion
        // For now, return a simple average
        float sum = 0.0;
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            sum += float(samples[i]);
        }
        float avg = sum / float(NUM_SAMPLES);
        return float3(avg, avg, avg);
    }

    static const int NUM_SAMPLES = 30;

    half samples[NUM_SAMPLES];  // Example with 30 spectral samples
};
#endif 