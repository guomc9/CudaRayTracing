#ifndef _RENDER_CUH_
#define _RENDER_CUH_

#include "DeviceBVH.cuh"
#include "DeviceLights.cuh"
#include "DeviceStack.cuh"
#include "Global.h"
#include "Scene.h"
#include "Ray.cuh"
#include <iostream>
#include <cuda_runtime.h>

__device__ inline void sample_light(LightSamplePayload& payload, curandState* rand_state, DeviceLight* light);

__device__ inline bool blocked(Ray ray, float t_to_light, DeviceBVH* device_bvh, DeviceStack<int, BVH_STACK_SIZE>* stack);

// __device__ inline bool blocked_or_through(Ray ray, LightSamplePayload& light_sample, float t_to_light, DeviceBVH* device_bvh, DeviceStack<int, BVH_STACK_SIZE>* stack)
// {
//     HitPayload hit = device_bvh->intersect(ray.get_origin(), ray.get_dir(), ray.get_inv_dir(), stack);
//     if (t_to_light - hit.t > EPSILON)
//     {
//         if (hit.m.has_emission())
//         {
//             light_sample = LightSamplePayload(hit.pos, hit.normal, hit.m.get_ke(), hit.area);
//             return false;
//         }
//         else
//         {
//             return true;
//         }
//     }
//     return false;
// }

__device__ Eigen::Vector3f cast_ray(
    Ray ray, 
    int depth, 
    int light_sample_n, 
    float P_RR, 
    curandState* rand_state, 
    DeviceBVH* device_bvh, 
    DeviceLights* lights, 
    DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stack, 
    DeviceStack<int, BVH_STACK_SIZE>* bvh_stack
);

__device__ Eigen::Vector3f cast_ray_v2(
    Ray ray, 
    int depth, 
    int light_sample_n, 
    float P_RR, 
    curandState* rand_state, 
    DeviceBVH* device_bvh, 
    DeviceLights* lights, 
    DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stack, 
    DeviceStack<int, BVH_STACK_SIZE>* bvh_stack
);

__global__ void view_render_kernel(
    unsigned int width, 
    unsigned int height, 
    Eigen::Vector3f eye_pos, 
    Eigen::Matrix3f inv_view_mat, 
    float fovY, 
    unsigned int spp, 
    float P_RR, 
    int light_sample_n, 
    DeviceBVH* device_bvh, 
    uchar3* device_frame_buffer, 
    DeviceLights* lights, 
    DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stacks, 
    DeviceStack<int, BVH_STACK_SIZE>* bvh_stacks
);

class Render {
public:
    
};

class RTRenderImpl
{
    private:
        Scene* scene;
        unsigned int lights_size;
        unsigned int spp;
        unsigned int light_sample_n;
        float P_RR;

        // host parameters
        DeviceBVH* host_bvh;
        DeviceLights* host_lights;

        // device parameters
        DeviceBVH* device_bvh;
        // uchar3* device_frame_buffer;   // RGB
        DeviceLights* device_lights;
        DeviceStack<int, BVH_STACK_SIZE>* device_bvh_stacks;
        DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* device_bounce_stacks;
        // unsigned char* frame_buffer;  // RGB

    public:
        RTRenderImpl(Scene* _scene, unsigned int _spp=16, float _P_RR=0.8f, unsigned int _light_sample_n=1)
        {
            cudaError_t err;
            scene = _scene;
            spp = _spp;
            P_RR = _P_RR;
            light_sample_n = _light_sample_n;

            // load bvh from host to device
            try
            {
                host_bvh = new DeviceBVH(scene->get_bvh());
                err = cudaMalloc((void**)&device_bvh, sizeof(DeviceBVH));
                err = cudaMemcpy(device_bvh, host_bvh, sizeof(DeviceBVH), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_bvh to cuda.");
                }
                // load lights from host device
                host_lights = new DeviceLights(scene->get_light_objs());
                err = cudaMalloc((void**)&device_lights, sizeof(DeviceLights));
                err = cudaMemcpy(device_lights, host_lights, sizeof(DeviceLights), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_lights to cuda.");
                }
                //create stacks in device
                err = cudaMalloc((void**)&device_bvh_stacks, sizeof(DeviceStack<int, BVH_STACK_SIZE>) * scene->get_pixels());
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_bvh_stacks to cuda.");
                }
                err = cudaMalloc((void**)&device_bounce_stacks, sizeof(DeviceStack<HitPayload, BOUNCE_STACK_SIZE>) * scene->get_pixels());
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_bounce_stacks to cuda.");
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
        }

        void run_view(Eigen::Vector3f eye_pos, Eigen::Matrix3f inv_view_mat, float fovY, uchar3* device_frame_buffer)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((scene->get_width() + threadsPerBlock.x - 1) / threadsPerBlock.x, (scene->get_height() + threadsPerBlock.y - 1) / threadsPerBlock.y);
            view_render_kernel<< <numBlocks, threadsPerBlock>> >(scene->get_width(), scene->get_height(), eye_pos, inv_view_mat, fovY, spp, P_RR, light_sample_n, device_bvh, device_frame_buffer, device_lights, device_bounce_stacks, device_bvh_stacks);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }            
        }

        void free()
        {
            // delete[] frame_buffer;
            host_bvh->free();
            host_lights->free();
            cudaFree(device_bvh);
            cudaFree(device_lights);
            cudaFree(device_bounce_stacks);
            cudaFree(device_bvh_stacks);
        }

        // void save_frame_buffer(const char* save_path)
        // {
        //     printf("%s\n", save_path);
        //     stbi_write_png(save_path, scene->get_width(), scene->get_height(), 3, frame_buffer, scene->get_width() * 3);
        // }

        // unsigned char* get_frame_buffer() const
        // {
        //     return frame_buffer;
        // }

        // void set_height(const unsigned int& height)
        // {
        //     scene->set_height(height);
        // }

        // void set_width(const unsigned int& width)
        // {
        //     scene->set_width(width);
        // }

        void set_spp(const int& spp)
        {
            this->spp = spp;
        }

        void set_P_RR(const float& P_RR)
        {
            this->P_RR = P_RR;
        }

        void set_light_sample_n(const int& light_sample_n)
        {
            this->light_sample_n = light_sample_n;
        }
};

#endif