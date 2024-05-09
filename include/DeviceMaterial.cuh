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
        float ns;
        bool has_emit;
        Illum mode;

    public:
        __host__ __device__ DeviceMaterial()
        {
            kd = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ks = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ka = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ke = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            ns = 1.0f;
            has_emit = false;
            mode = DIFFUSE;
        }

        __host__ __device__ DeviceMaterial(Material m)
        {
            kd = m.get_kd();
            ks = m.get_ks();
            ka = m.get_ka();
            ke = m.get_ke();
            ns = m.get_ns();
            has_emit = m.has_emission();
            mode = m.get_mode();
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

        __device__ float get_ns() const
        {
            return ns;
        }

        __device__ float set_ns(float _ns)
        {
            ns = _ns;
        }

        __device__ void set_kd(Eigen::Vector3f _kd)
        {
            kd = _kd;
        }

        __device__ void set_ks(Eigen::Vector3f _ks)
        {
            ks = _ks;
        }

        __device__ void set_ka(Eigen::Vector3f _ka)
        {
            ka = _ka;
        }

        __device__ void set_ke(Eigen::Vector3f _ke)
        {
            ke = _ke;
        }

        __device__ bool has_emission()
        {   
            return has_emit;
        }

        __device__ Illum get_mode()
        {
            return mode;
        }
};
#endif