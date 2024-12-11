#version 330 core

in vec2 fragmentTexCoord;
in vec3 fragmentPosition;
in vec3 fragmentNormal;

uniform sampler2D imageTexture;
uniform vec3 objectColor;

out vec4 color;

void main()
{
    vec3 temp = vec3(0.0);

    // Ambient lighting
    temp += texture(imageTexture, fragmentTexCoord).rgb * 0.3;

    color = vec4(temp, texture(imageTexture, fragmentTexCoord).a);
}