#version 430 core

// Define a uniform struct for lights
struct Light {
    // The matrices are used for shadow mapping. You need to fill it according to how we are filling it when building the normal maps (node_render_shadow_mapping.cpp). 
    // Now, they are filled with identity matrix. You need to modify C++ code innode_render_deferred_lighting.cpp.
    // Position and color are filled.
    mat4 light_projection;
    mat4 light_view;
    vec3 position;
    float radius;
    vec3 color; // Just use the same diffuse and specular color.
    int shadow_map_id;
};

layout(std430, binding = 0) buffer lightsBuffer {
Light lights[];
};

uniform vec2 iResolution;

uniform sampler2D diffuseColorSampler;
uniform sampler2D normalMapSampler; // You should apply normal mapping in rasterize_impl.fs
uniform sampler2D metallicRoughnessSampler;
uniform sampler2DArray shadow_maps;
uniform sampler2D position;

uniform vec3 camPos;

uniform int light_count;
uniform float ambientStrength;
uniform float shadowBiasScale;
uniform float shadowPcfRadius;
uniform float shadowLightSize;
uniform bool enableAntialiasing;
uniform float aaEdgeThreshold;
uniform float aaBlendStrength;
uniform bool enableDistanceAttenuation;

layout(location = 0) out vec4 Color;

const int PCSS_SAMPLE_COUNT = 16;
const vec2 poissonDisk[PCSS_SAMPLE_COUNT] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845),
    vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507),
    vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790)
);

struct ShadingResult {
    vec3 position;
    vec3 normal;
    vec3 finalColor;
};

float computeShadowBias(vec3 worldNormal, vec3 lightDir) {
	float shadowBiasMin = max(shadowBiasScale * 0.1, 1E-4);
    return max(
        shadowBiasScale * (1.0 - max(dot(worldNormal, lightDir), 0.0)),
        shadowBiasMin);
}

float computeDistanceAttenuation(float lightDistance) {
    if (!enableDistanceAttenuation) {
        return 1.0;
    }

    return 1.0 / max(lightDistance * lightDistance, 1E-4);
}

float sampleShadowDepth(int lightIndex, vec2 uv) {
    return texture(
        shadow_maps,
        vec3(clamp(uv, vec2(0.0), vec2(1.0)), lights[lightIndex].shadow_map_id)).x;
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

mat2 buildKernelRotation(vec2 shadowUv, int lightIndex) {
    vec2 seed = shadowUv * vec2(textureSize(shadow_maps, 0).xy) +
        vec2(float(lightIndex) * 17.0, float(lightIndex) * 31.0);
    float angle = 6.28318530718 * hash12(seed);
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

vec2 getKernelOffset(int sampleIndex, mat2 rotationMatrix) {
    return rotationMatrix * poissonDisk[sampleIndex];
}

float computeLightSizeUv(int lightIndex, vec2 texelSize) {
    float texelScale = max(texelSize.x, texelSize.y);
    float lightRadius = max(lights[lightIndex].radius, 0.25);
    return shadowLightSize * lightRadius * texelScale;
}

float findAverageBlockerDepth(
    int lightIndex,
    vec2 shadowUv,
    float zReceiver,
    float bias,
    float searchRadiusUv,
    mat2 rotationMatrix)
{
    float blockerDepthSum = 0.0;
    int blockerCount = 0;

    for (int i = 0; i < PCSS_SAMPLE_COUNT; ++i) {
        vec2 sampleUv = shadowUv + getKernelOffset(i, rotationMatrix) * searchRadiusUv;
        float shadowDepth = sampleShadowDepth(lightIndex, sampleUv);
        if (shadowDepth < zReceiver - bias) {
            blockerDepthSum += shadowDepth;
            blockerCount += 1;
        }
    }

    if (blockerCount == 0) {
        return -1.0;
    }

    return blockerDepthSum / float(blockerCount);
}

float filterShadowPcf(
    int lightIndex,
    vec2 shadowUv,
    float zReceiver,
    float bias,
    float filterRadiusUv,
    mat2 rotationMatrix)
{
    float visibility = 0.0;
    for (int i = 0; i < PCSS_SAMPLE_COUNT; ++i) {
        vec2 sampleUv = shadowUv + getKernelOffset(i, rotationMatrix) * filterRadiusUv;
        float shadowDepth = sampleShadowDepth(lightIndex, sampleUv);
        visibility += zReceiver - bias > shadowDepth ? 0.0 : 1.0;
    }

    return visibility / float(PCSS_SAMPLE_COUNT);
}

float computePcssVisibility(
    int lightIndex,
    vec2 shadowUv,
    float zReceiver,
    float bias)
{
    vec2 texelSize = 1.0 / vec2(textureSize(shadow_maps, 0).xy);
    float lightSizeUv = computeLightSizeUv(lightIndex, texelSize);
    mat2 rotationMatrix = buildKernelRotation(shadowUv, lightIndex);
    float minFilterRadiusUv = shadowPcfRadius * max(texelSize.x, texelSize.y);
    float searchRadiusUv = max(
        lightSizeUv * max(zReceiver, texelSize.x),
        max(minFilterRadiusUv, texelSize.x));

    float zBlocker = findAverageBlockerDepth(
        lightIndex,
        shadowUv,
        zReceiver,
        bias,
        searchRadiusUv,
        rotationMatrix);
    if (zBlocker < 0.0) {
        return 1.0;
    }

    float penumbraRadiusUv =
        max((zReceiver - zBlocker) / max(zBlocker, 1E-4), 0.0) * lightSizeUv;

    float filterRadiusUv = max(penumbraRadiusUv, minFilterRadiusUv);

    if (filterRadiusUv <= texelSize.x && filterRadiusUv <= texelSize.y) {
        float storedDepth = sampleShadowDepth(lightIndex, shadowUv);
        return zReceiver - bias > storedDepth ? 0.0 : 1.0;
    }

    return filterShadowPcf(
        lightIndex,
        shadowUv,
        zReceiver,
        bias,
        filterRadiusUv,
        rotationMatrix);
}

float computeVisibility(int lightIndex, vec3 worldPos, vec3 worldNormal, vec3 lightDir) {
    if (lights[lightIndex].shadow_map_id < 0) {
        return 1.0;
    }

    vec4 lightClip = lights[lightIndex].light_projection *
                     lights[lightIndex].light_view * vec4(worldPos, 1.0);
    if (abs(lightClip.w) < 1E-6) {
        return 1.0;
    }

    vec3 lightNdc = lightClip.xyz / lightClip.w;
    vec2 shadowUv = lightNdc.xy * 0.5 + 0.5;
    if (shadowUv.x < 0.0 || shadowUv.x > 1.0 || shadowUv.y < 0.0 || shadowUv.y > 1.0) {
        return 1.0;
    }
    if (lightNdc.z < -1.0 || lightNdc.z > 1.0) {
        return 1.0;
    }

    float currentDepth = lightNdc.z * 0.5 + 0.5;
    float bias = computeShadowBias(worldNormal, lightDir);

    if (shadowPcfRadius <= 0.0) {
        float storedDepth = sampleShadowDepth(lightIndex, shadowUv);
        return currentDepth - bias > storedDepth ? 0.0 : 1.0;
    }

    return computePcssVisibility(lightIndex, shadowUv, currentDepth, bias);
}

ShadingResult shadeAt(vec2 uv) {
    ShadingResult result;
    vec3 pos = texture(position, uv).xyz;
    vec3 normal = normalize(texture(normalMapSampler, uv).xyz);
    vec3 albedo = texture(diffuseColorSampler, uv).xyz;

    vec2 metalnessRoughness = texture(metallicRoughnessSampler, uv).xy;
    float metal = clamp(metalnessRoughness.x, 0.0, 1.0);
    float roughness = clamp(metalnessRoughness.y, 0.04, 1.0);

    vec3 viewDir = normalize(camPos - pos);
    vec3 kd = (1.0 - metal) * albedo;
    vec3 ks = mix(vec3(0.04), albedo, metal);
    float shininess = mix(128.0, 8.0, roughness);

    vec3 ambient = ambientStrength * kd;
    vec3 directLighting = vec3(0.0);

    for (int i = 0; i < light_count; i++) {
        vec3 lightVector = lights[i].position - pos;
        float lightDistance = length(lightVector);
        if (lightDistance < 1E-6) {
            continue;
        }

        vec3 lightDir = lightVector / lightDistance;
        float nDotL = max(dot(normal, lightDir), 0.0);
        if (nDotL <= 0.0) {
            continue;
        }

        vec3 halfDir = normalize(lightDir + viewDir);
        float nDotH = max(dot(normal, halfDir), 0.0);

        vec3 diffuseTerm = kd / 3.14159265;
        vec3 specularTerm = ks * pow(nDotH, shininess);
        vec3 intensity = lights[i].color;
        float visibility = computeVisibility(i, pos, normal, lightDir);
        float attenuation = computeDistanceAttenuation(lightDistance);

        directLighting +=
            visibility * attenuation *
            (diffuseTerm + specularTerm) * intensity * nDotL;
    }

    result.position = pos;
    result.normal = normal;
    result.finalColor = ambient + directLighting;
    return result;
}

float computeEdgeMetric(vec2 uv, vec3 centerPos, vec3 centerNormal) {
    vec2 texelSize = 1.0 / iResolution;
    float maxNormalEdge = 0.0;
    float maxPositionEdge = 0.0;
    float viewDistance = max(length(camPos - centerPos), 1E-3);

    vec2 offsets[4] = vec2[](vec2(texelSize.x, 0.0),
                             vec2(-texelSize.x, 0.0),
                             vec2(0.0, texelSize.y),
                             vec2(0.0, -texelSize.y));

    for (int i = 0; i < 4; ++i) {
        vec2 sampleUv = clamp(uv + offsets[i], vec2(0.0), vec2(1.0));
        vec3 neighborNormal = normalize(texture(normalMapSampler, sampleUv).xyz);
        vec3 neighborPos = texture(position, sampleUv).xyz;

        maxNormalEdge = max(
            maxNormalEdge,
            1.0 - clamp(dot(centerNormal, neighborNormal), 0.0, 1.0));
        maxPositionEdge = max(
            maxPositionEdge,
            length(neighborPos - centerPos) / viewDistance);
    }

    return max(maxNormalEdge, maxPositionEdge);
}

vec3 applyEdgeAwareAntialiasing(vec2 uv, ShadingResult centerResult) {
    if (!enableAntialiasing) {
        return centerResult.finalColor;
    }

    float edgeMetric = computeEdgeMetric(uv, centerResult.position, centerResult.normal);
    if (edgeMetric <= aaEdgeThreshold) {
        return centerResult.finalColor;
    }

    vec2 texelSize = 1.0 / iResolution;
    vec3 neighborAverage = vec3(0.0);
    vec2 offsets[4] = vec2[](vec2(texelSize.x, 0.0),
                             vec2(-texelSize.x, 0.0),
                             vec2(0.0, texelSize.y),
                             vec2(0.0, -texelSize.y));

    for (int i = 0; i < 4; ++i) {
        vec2 sampleUv = clamp(uv + offsets[i], vec2(0.0), vec2(1.0));
        neighborAverage += shadeAt(sampleUv).finalColor;
    }
    neighborAverage *= 0.25;

    float blend = clamp(
        (edgeMetric - aaEdgeThreshold) * aaBlendStrength /
            max(1.0 - aaEdgeThreshold, 1E-3),
        0.0,
        1.0);
    return mix(centerResult.finalColor, neighborAverage, blend * 0.5);
}

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution;
    ShadingResult centerResult = shadeAt(uv);

    Color = vec4(applyEdgeAwareAntialiasing(uv, centerResult), 1.0);
}