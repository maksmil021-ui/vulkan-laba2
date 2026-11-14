#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 ambientColor;
    vec4 albedo;
    vec4 specular;
} ubo;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-ubo.lightDir.xyz);
    vec3 V = normalize(ubo.cameraPos.xyz - fragPos);
    vec3 H = normalize(L + V);

    vec3 ambient = ubo.ambientColor.rgb * ubo.albedo.rgb * fragColor;

    float diff = max(dot(N, L), 0.0);
    vec3 diffuse = diff * ubo.lightColor.rgb * ubo.lightColor.a * ubo.albedo.rgb * fragColor;

    float spec = pow(max(dot(N, H), 0.0), ubo.specular.a);
    vec3 specularColor = spec * ubo.lightColor.rgb * ubo.lightColor.a * ubo.specular.rgb;

    vec3 finalColor = ambient + diffuse + specularColor;
    outColor = vec4(finalColor, 1.0);
}