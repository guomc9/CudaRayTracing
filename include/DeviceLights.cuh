#ifndef _DEVICE_LIGHTS_CUH_
#define _DEVICE_LIGHTS_CUH_
#include "Object.h"
#include "DeviceTriangle.cuh"

class DeviceLight
{
    private:
        size_t* tn;
        DeviceTriangle* dts;
    public:
        DeviceLight(Object& obj)
        {
            cudaError_t err;
            std::vector<DeviceTriangle> vt(obj.get_triangles().begin(), obj.get_triangles().end());
            size_t t_size = vt.size();
            err = cudaMalloc((void**)&dts, sizeof(DeviceTriangle) * t_size);
            err = cudaMemcpy(dts, vt.data(), sizeof(DeviceTriangle) * t_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("DeviceLight.dts to cuda.");
            }
            err = cudaMalloc((void**)&tn, sizeof(size_t));
            err = cudaMemcpy(tn, &t_size, sizeof(size_t), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("DeviceLight.tn to cuda.");
            }
        }

        __device__ LightSamplePayload sample(curandState* rand_state)
        {
            unsigned int ti = get_cuda_random_uint(rand_state) % *tn;
            return dts[ti].sample(rand_state);
        }

        __device__ DeviceTriangle* get_triangle(uint64_t index) const
        {
            return &dts[index % *tn];
        }

        __device__ uint64_t get_triangles_size() const
        {
            return *tn;
        }

        void free()
        {
            cudaFree(tn);
            cudaFree(dts);
        }
};

class DeviceLights
{
    private:
        size_t* ln;
        DeviceLight* dls;
        std::vector<DeviceLight> vl;
    public:
        DeviceLights(std::vector<Object>& lights)
        {
            size_t l_size = lights.size();
            cudaMalloc((void**)&ln, sizeof(size_t));
            cudaMemcpy(ln, &l_size,sizeof(size_t), cudaMemcpyHostToDevice);
            auto err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("DeviceLights.ln to cuda.");
            }
            for(auto& light : lights)
            {
                DeviceLight h(light);
                vl.push_back(h);
            }
            cudaMalloc((void**)&dls, sizeof(DeviceLight) * vl.size());
            cudaMemcpy(dls, vl.data(), sizeof(DeviceLight) * vl.size(), cudaMemcpyHostToDevice);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("DeviceLights.dls to cuda.");
            }
        }

        void free()
        {
            cudaFree(ln);
            cudaFree(dls);
        }

        __device__ DeviceLight* get_lights() const
        {
            return dls;
        }

        __device__ DeviceLight* get_light(uint64_t index) const
        {
            return &dls[index % *ln];
        }

        __device__ uint64_t get_lights_size() const
        {
            return *ln;
        }
};

#endif