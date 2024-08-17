#include "GLUtils.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

int configDefaultShader() 
{
    const char *vertexShaderSource = R"glsl(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
        })glsl";

    const char *fragmentShaderSource = R"glsl(
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoord;

        uniform sampler2D texture1;

        void main()
        {
            FragColor = texture(texture1, TexCoord);
        })glsl";

    GLint vs_id = glCreateShader(GL_VERTEX_SHADER);
    GLint fs_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs_id, 1, &vertexShaderSource, nullptr);
    glShaderSource(fs_id, 1, &fragmentShaderSource, nullptr);
    
    GLint prog_id = glCreateProgram();
    
    glCompileShader(vs_id);
    glCompileShader(fs_id);
    
    glAttachShader(prog_id, vs_id);
    glAttachShader(prog_id, fs_id);
    
    glLinkProgram(prog_id);
    glValidateProgram(prog_id);

    glDeleteShader(vs_id);
    glDeleteShader(fs_id);

    return prog_id;
}

unsigned int generatePixelBufferObject(int width, int height, int channels)
{
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * channels, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    return pbo;
}

int generateRgbTexture(int width, int height) 
{
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}