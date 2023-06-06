#ifndef _DEVICE_MATERIAL_CUH_
#define _DEVICE_MATERIAL_CUH_
#include "Material.h"

class DeviceMaterial
{
    private:
        Eigen::Vector3f kd;
        Eigen::Vector3f ks;
        Eigen::Vector3f ka;
        Eigen::Vector3f ke;
        bool has_emit;

    public:
        __host__ __device__ DeviceMaterial()
        {
            kd = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ks = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ka = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ke = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            has_emit = false;
        }

        __host__ __device__ DeviceMaterial(Material m)
        {
            kd = m.get_kd();
            ks = m.get_ks();
            ka = m.get_ka();
            ke = m.get_ke();
            has_emit = m.has_emission();
        }

        // Getter functions
        __device__ Eigen::Vector3f get_kd() const
        {
            return kd;
        }

        __device__ Eigen::Vector3f get_ks() const
        {
            return ks;
        }

        __device__ Eigen::Vector3f get_ka() const
        {
            return ka;
        }

        __device__ Eigen::Vector3f get_ke() const
        {
            return ke;
        }

        __device__ bool has_emission()
        {   
            return has_emit;
        }
};
#endif