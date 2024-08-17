#ifndef _RESOURCE_CUH_
#define _RESOURCE_CUH_

#include <cuda_runtime.h>
#include <vector>

class ImageResource 
{
public:
    ImageResource() : pbo(0), cuda_pbo_resource(nullptr), pixel_size(0), device_frame_buffer(nullptr), init_flag(false) {};

    ImageResource(int width, int height);

    ~ImageResource();

    unsigned int getPbo() { return pbo; }

    cudaGraphicsResource* getCudaGraphicsResource() { return cuda_pbo_resource; }

    uchar3* getDeviceFrameBuffer() { return device_frame_buffer; }

    void loadImageFromDevice();

    bool getInitFlag() { return init_flag; }

private:
    unsigned int pbo;
    cudaGraphicsResource* cuda_pbo_resource;
    unsigned int pixel_size;
    uchar3* device_frame_buffer;
    bool init_flag;
};

class ImagePool 
{
public:
    ImagePool()
    {
        image_pool = std::vector<ImageResource*>(0);
        cur = 0;
    }

    ImagePool(int width, int height);

    ~ImagePool();

    ImageResource* get(int idx);

    int allocImage();

    bool getInitFlag() { return image_pool.size() > 0; }

private:
    std::vector<ImageResource*> image_pool;
    int cur;
};

#endif