#include "Resource.cuh"
#include "GLUtils.h"
#include <glad/glad.h>
#include <cuda_gl_interop.h>
// #include <GLFW/glfw3.h>
#include "Global.h"

ImageResource::ImageResource(int width, int height)
{
    init_flag = false;

    pbo = generatePixelBufferObject(width, height, 3);
    // printf("pbo id: %d\n", pbo);
    auto err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        printf("CUDA error in registering buffer: %s\n", cudaGetErrorString(err));
    }
    pixel_size = width * height;
    // create frame_buffer in device
    // frame_buffer = new unsigned char[3 * scene->get_pixels()];
    err = cudaMalloc((void**)&device_frame_buffer, sizeof(uchar3) * pixel_size);
    if (err != cudaSuccess)
    {
        printf("CUDA error in allocating memory: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("Render.device_frame_buffer to cuda.");
    }

    init_flag = true;
}

ImageResource::~ImageResource()
{
    cudaFree(device_frame_buffer);
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
}

void ImageResource::loadImageFromDevice()
{
    // write device_frame_buffer to pbo
            
    auto err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
    {
        printf("CUDA error in mapping resources: %s\n", cudaGetErrorString(err));
    }
    size_t num_bytes = sizeof(uchar3) * pixel_size; 
    uchar3* cuda_pbo_pointer;
    err = cudaGraphicsResourceGetMappedPointer((void **)&cuda_pbo_pointer, &num_bytes, cuda_pbo_resource);
    if (err != cudaSuccess)
    {
        printf("CUDA error in getting mapped pointer: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy((void *)cuda_pbo_pointer, (void *)device_frame_buffer, sizeof(uchar3) * pixel_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess)
    {
        printf("CUDA error in memory copy: %s\n", cudaGetErrorString(err));
    }
    err = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess)
    {
         printf("CUDA error in umapping resources: %s\n", cudaGetErrorString(err));
    }
}

ImagePool::ImagePool(int width, int height)
{
    image_pool = std::vector<ImageResource*>(QUEUE_SIZE);

    for (int i = 0; i < QUEUE_SIZE; i++)
    {
        ImageResource* ir = new ImageResource(width, height);
        image_pool[i] = ir;
    }

    cur = 0;
}

ImagePool::~ImagePool()
{
    for (int i = 0; i < QUEUE_SIZE; i++)
    {
        delete image_pool[i];
    }
    image_pool.clear();
}

ImageResource* ImagePool::get(int idx) 
{
    if (image_pool.size() > 0 && idx >= 0 && idx < QUEUE_SIZE)
        return image_pool[idx];
    return nullptr;
};

int ImagePool::allocImage()
{
    int alloc = cur;
    cur = (cur + 1) % QUEUE_SIZE;
    return alloc;
}
