#version 330 core

#define MAX_LIGHTS 100

struct PointLight {
    vec3 position;
    vec3 color;
    float strength;
};

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

uniform sampler2D imageTexture;
uniform int totalLights;
uniform PointLight lights[MAX_LIGHTS];
uniform vec3 cameraPosition;

out vec4 color;

vec3 calculatePointLight(PointLight light, vec3 fragmentPosition, vec3 fragmentNormal);

void main() {
    vec3 temp = vec3(0.0);

    // Ambient lighting
    temp += texture(imageTexture, fragmentTexCoord).rgb * 0.2;

    for (int i = 0; i < totalLights; i++){
        temp += calculatePointLight(lights[i], fragmentPosition, normalize(fragmentNormal));
    }

    color = vec4(temp, texture(imageTexture, fragmentTexCoord).a);
}

vec3 calculatePointLight(PointLight light, vec3 fragmentPosition, vec3 fragmentNormal) {
    vec3 result = vec3(0.0);
    vec3 baseTexture = texture(imageTexture, fragmentTexCoord).rgb;

    // Geometric data
    vec3 fragLight = light.position - fragmentPosition;
    float distance = length(fragLight);
    fragLight = normalize(fragLight);
    vec3 fragCamera = normalize(cameraPosition - fragmentPosition);
    vec3 halfVec = normalize(fragLight + fragCamera);

    // Diffuse lighting
    result += light.color * light.strength * max(0.0, dot(fragmentNormal, fragLight)) / (distance * distance) * baseTexture;

    // Specular lighting
    result += light.color * light.strength * pow(max(0.0, dot(fragmentNormal, halfVec)), 32) / (distance * distance);

    return result;
}