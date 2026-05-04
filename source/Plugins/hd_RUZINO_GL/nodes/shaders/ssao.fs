#version 430 core

uniform vec2 iResolution;

uniform sampler2D colorSampler;
uniform sampler2D positionSampler;
uniform sampler2D normalSampler;

uniform mat4 view;
uniform mat4 projection;

uniform float radius;
uniform float strength;

layout(location = 0) out vec4 Color;

const int SSAO_SAMPLE_COUNT = 16;
const float PI = 3.14159265;
const float AO_MAX_DARKENING = 0.65;

float hash12(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 0.1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

vec3 buildKernelSample(int index) {
	float fi = float(index) + 1.0;
	float u1 = hash12(vec2(fi, 13.37));
	float u2 = hash12(vec2(fi, 91.17));

	float phi = 2.0 * PI * u1;
	float z = u2;
	float r = sqrt(max(0.0, 1.0 - z * z));

	vec3 sampleDir = vec3(r * cos(phi), r * sin(phi), z);
	float t = fi / float(SSAO_SAMPLE_COUNT);
	float scale = mix(0.1, 1.0, t * t);
	return sampleDir * scale;
}

vec3 worldToViewPosition(vec3 worldPos) {
	return (view * vec4(worldPos, 1.0)).xyz;
}

vec3 computeGeometricNormal(vec3 positionView) {
	vec3 dpdx = dFdx(positionView);
	vec3 dpdy = dFdy(positionView);
	vec3 normalView = normalize(cross(dpdx, dpdy));
	vec3 viewDir = normalize(-positionView);
	if (dot(normalView, viewDir) < 0.0) {
		normalView = -normalView;
	}
	return normalView;
}

void main() {
	vec2 uv = gl_FragCoord.xy / iResolution;
	vec4 baseColor = texture(colorSampler, uv);

	vec3 positionWorld = texture(positionSampler, uv).xyz;
	vec3 normalWorld = texture(normalSampler, uv).xyz;
	if (length(normalWorld) < 1E-4) {
		Color = baseColor;
		return;
	}

	vec3 positionView = worldToViewPosition(positionWorld);
	vec3 normalView = computeGeometricNormal(positionView);
	float thicknessBias = max(radius * 0.025, 0.0025);

	vec3 randomVec = normalize(vec3(
		hash12(gl_FragCoord.xy),
		hash12(gl_FragCoord.yx + 19.19),
		0.0) * 2.0 - 1.0);
	vec3 tangent = randomVec - normalView * dot(randomVec, normalView);
	if (length(tangent) < 1E-4) {
		tangent = normalize(cross(normalView, vec3(0.0, 1.0, 0.0)));
		if (length(tangent) < 1E-4) {
			tangent = normalize(cross(normalView, vec3(1.0, 0.0, 0.0)));
		}
	} else {
		tangent = normalize(tangent);
	}
	vec3 bitangent = normalize(cross(normalView, tangent));
	mat3 TBN = mat3(tangent, bitangent, normalView);

	float occlusion = 0.0;
	int validSamples = 0;

	for (int i = 0; i < SSAO_SAMPLE_COUNT; ++i) {
		vec3 sampleOffset = TBN * buildKernelSample(i);
		vec3 samplePosView = positionView + sampleOffset * radius;

		vec4 projected = projection * vec4(samplePosView, 1.0);
		if (abs(projected.w) < 1E-6) {
			continue;
		}

		vec3 sampleNdc = projected.xyz / projected.w;
		vec2 sampleUv = sampleNdc.xy * 0.5 + 0.5;
		if (sampleUv.x < 0.0 || sampleUv.x > 1.0 || sampleUv.y < 0.0 || sampleUv.y > 1.0) {
			continue;
		}

		vec3 sampleNormalWorld = texture(normalSampler, sampleUv).xyz;
		if (length(sampleNormalWorld) < 1E-4) {
			continue;
		}

		vec3 realPosView = worldToViewPosition(texture(positionSampler, sampleUv).xyz);
		vec3 sampleDelta = realPosView - positionView;

		float sampleDistance = length(sampleDelta);
		float rangeWeight = 1.0 - smoothstep(radius * 0.5, radius * 1.5, sampleDistance);
		float angleWeight = max(dot(normalView, normalize(sampleDelta)), 0.0);
		float blocked = realPosView.z >= samplePosView.z + thicknessBias ? 1.0 : 0.0;

		occlusion += blocked * rangeWeight * angleWeight;
		validSamples += 1;
	}

	float ao = 1.0;
	if (validSamples > 0) {
		ao = 1.0 - occlusion / float(validSamples);
	}
	ao = clamp(ao, 0.0, 1.0);
	float finalAo = clamp(1.0 - (1.0 - ao) * strength * AO_MAX_DARKENING, 0.0, 1.0);

	Color = vec4(baseColor.rgb * finalAo, baseColor.a);
}