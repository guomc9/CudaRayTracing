#ifndef _GL_UTILS_H_
#define _GL_UTILS_H_

// Return shader program id
int configDefaultShader();

// Return pixel buffer object
unsigned int generatePixelBufferObject(int width, int height, int channels);

// Return rgb texture id
int generateRgbTexture(int width, int height);

#endif