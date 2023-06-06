#ifndef _PAYLOAD_CUH_
#define _PAYLOAD_CUH_

#include "Eigen/Dense"
#include "DeviceMaterial.cuh"

#include <cfloat>
struct HitPayload
{
    bool happend;
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    float t;
    DeviceMaterial m;

    __device__ HitPayload():happend(false), t(FLT_MAX), pos(Eigen::Vector3f(0.0f, 0.0f, 0.0f)), normal(Eigen::Vector3f(0.0f, 0.0f, 0.0f)), m(DeviceMaterial()){};
    __device__ HitPayload(bool _happend, Eigen::Vector3f _pos, Eigen::Vector3f _normal, float _t, DeviceMaterial _m)
    :happend(_happend), pos(_pos), normal(_normal), t(_t), m(_m){};
    __device__ HitPayload& operator=(const HitPayload& other)
    {
        if (this != &other) {
            happend = other.happend;
            pos = other.pos;
            normal = other.normal;
            t = other.t;
            m = other.m;
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
    float inv_pdf;
    __device__ LightSamplePayload():inv_pdf(0.0f){};
    __device__ LightSamplePayload(Eigen::Vector3f _pos, Eigen::Vector3f _normal, Eigen::Vector3f _emit, float _inv_pdf)
    :pos(_pos), normal(_normal), emit(_emit), inv_pdf(_inv_pdf){};
};

#endif