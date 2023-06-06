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
            std::vector<DeviceTriangle> vt(obj.get_triangles().begin(), obj.get_triangles().end());
            size_t t_size = vt.size();
            cudaMalloc((void**)&dts, sizeof(DeviceTriangle) * t_size);
            cudaMemcpy(dts, vt.data(), sizeof(DeviceTriangle) * t_size, cudaMemcpyHostToDevice);
            cudaMalloc((void**)&tn, sizeof(size_t));
            cudaMemcpy(tn, &t_size, sizeof(size_t), cudaMemcpyHostToDevice);
        }

        __device__ LightSamplePayload sample(curandState* rand_state)
        {
            // printf("tn=%d\n", *tn);
            unsigned int ti = get_cuda_random_uint(rand_state) % *tn;
            // printf("here6, ti=%d\n", ti);
            return dts[ti].sample(rand_state);
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
            
            for(auto& light : lights)
            {
                DeviceLight h(light);
                vl.push_back(h);
            }
            cudaMalloc((void**)&dls, sizeof(DeviceLight) * vl.size());
            cudaMemcpy(dls, vl.data(), sizeof(DeviceLight) * vl.size(), cudaMemcpyHostToDevice);
        }

        void free()
        {
            cudaFree(ln);
            cudaFree(dls);
        }

        __device__ DeviceLight* get_lights()
        {
            return dls;
        }

        __device__ unsigned int get_lights_size()
        {
            return *ln;
        }
};

#endif