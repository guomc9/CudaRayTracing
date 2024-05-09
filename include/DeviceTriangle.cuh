#ifndef _DEVICE_TRIANGLE_CUH_
#define _DEVICE_TRIANGLE_CUH_

#include "Eigen/Dense"
#include "DeviceMaterial.cuh"
#include "Triangle.h"
#include "Payload.cuh"
#include "Global.h"


class DeviceTriangle
{
    private:
        Eigen::Vector3f v1, v2, v3;
        Eigen::Vector3f e1, e2;
        Eigen::Vector3f normal;
        DeviceMaterial device_material;
        float area;
        float area_of_obj;

    public:
        DeviceTriangle(const Triangle t)
        {
            v1 = t.get_v1();
            v2 = t.get_v2();
            v3 = t.get_v3();
            e1 = v2 - v1;
            e2 = v3 - v1;
            device_material = DeviceMaterial(t.get_material());
            normal = t.get_normal();
            area = t.get_area();
            area_of_obj = t.get_area_of_obj();
        }

        DeviceTriangle(const DeviceTriangle& other)
        : v1(other.v1), v2(other.v2), v3(other.v3), e1(other.e1), e2(other.e2), normal(other.normal), device_material(other.device_material), area(other.area), area_of_obj(other.area_of_obj)
        {}

        __device__ HitPayload get_intersection(Eigen::Vector3f origin, Eigen::Vector3f dir) const
        {
            // Möller Trumbore Algorithm
            Eigen::Vector3f s = origin - v1;
            Eigen::Vector3f s1 = dir.cross(e2);
            Eigen::Vector3f s2 = s.cross(e1);
            float reciprocal = 1 / s1.dot(e1);
            float beta = s1.dot(s) * reciprocal;
            float gamma = s2.dot(dir) * reciprocal;
            float t = s2.dot(e2) * reciprocal;
            float alpha = 1 - beta - gamma;
            Eigen::Vector3f pos = origin + t * dir;
            if(inside(alpha, beta, gamma))
            {
                return HitPayload(origin, dir, true, pos, normal, t, area_of_obj, device_material);
            }
            return HitPayload();
        }

        __device__ bool inside(const float alpha, const float beta, const float gamma) const
        {
            if (0 < alpha && alpha < 1 && 0 < beta && beta < 1 && 0 < gamma && gamma < 1)
            {
                return true;
            }
            return false;
        }

        __device__ LightSamplePayload sample(curandState* rand_state) const
        {
            float alpha = get_cuda_random_float(rand_state);
            float beta = get_cuda_random_float(rand_state) * (1 - alpha);
            float gamma = 1 - alpha - beta;
            Eigen::Vector3f pos = alpha * v1 + beta * v2 + gamma * v3;
            return LightSamplePayload(pos, normal, device_material.get_ke(), area_of_obj, area_of_obj, area);
        }

        __device__ Eigen::Vector3f get_emission() const
        {
            return device_material.get_ke();
        }
    
};
#endif