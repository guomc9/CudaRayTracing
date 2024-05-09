#ifndef _PAYLOAD_CUH_
#define _PAYLOAD_CUH_

#include "Eigen/Dense"
#include "DeviceMaterial.cuh"

#include <cfloat>
struct HitPayload
{
    Eigen::Vector3f from_pos;
    Eigen::Vector3f from_dir;
    bool happend;
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    float t;
    float area;
    DeviceMaterial m;

    __device__ HitPayload():happend(false), t(FLT_MAX), pos(Eigen::Vector3f(0.0f, 0.0f, 0.0f)), normal(Eigen::Vector3f(0.0f, 0.0f, 0.0f)), m(DeviceMaterial()){};
    __device__ HitPayload(Eigen::Vector3f _from_pos, Eigen::Vector3f _from_dir, bool _happend, Eigen::Vector3f _pos, Eigen::Vector3f _normal, float _t, float _area, DeviceMaterial _m)
    :from_pos(_from_pos), from_dir(_from_dir), happend(_happend), pos(_pos), normal(_normal), t(_t), m(_m), area(_area){};
    __device__ HitPayload& operator=(const HitPayload& other)
    {
        if (this != &other) {
            from_pos = other.from_pos;
            from_dir = other.from_dir;
            happend = other.happend;
            pos = other.pos;
            normal = other.normal;
            t = other.t;
            m = other.m;
            area = other.area;
        }
        return *this;
    }
};

struct HitAABBPayload
{
    bool happend;
    float t_enter;
    float t_exit;
    __device__ HitAABBPayload():happend(false){};
    __device__ HitAABBPayload(bool _happend, float _t_enter, float _t_exit):happend(_happend), t_enter(_t_enter), t_exit(_t_exit){};
};


struct LightSamplePayload
{
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    Eigen::Vector3f emit;
    float inv_light_area;
    float inv_triangle_area;
    float inv_pdf;
    __device__ LightSamplePayload():inv_pdf(0.0f){};
    __device__ LightSamplePayload(Eigen::Vector3f _pos, Eigen::Vector3f _normal, Eigen::Vector3f _emit, float _inv_pdf, float _inv_light_area, float _inv_triangle_area)
    :pos(_pos), normal(_normal), emit(_emit), inv_pdf(_inv_pdf), inv_triangle_area(_inv_triangle_area), inv_light_area(_inv_light_area){};
    __device__ LightSamplePayload operator=(const LightSamplePayload& other)
    {
        if (this != &other)
        {
            pos = other.pos;
            normal = other.normal;
            emit = other.emit;
            inv_light_area = other.inv_light_area;
            inv_triangle_area = other.inv_triangle_area;
            inv_pdf = other.inv_pdf;
        }
        return *this;
    }
};

#endif