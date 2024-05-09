#ifndef _GLOBAL_H_
#define _GLOBAL_H_
#define _USE_MATH_DEFINES
#include <random>
#include <curand_kernel.h>
#include <math.h>
#include <cstdint>
#include <stdexcept>
#include "Eigen/Dense"

#define EPSILON         0.00001f
#define EPSILON_10       0.0001f
#define EPSILON_100       0.001f
#define EPSILON_1000       0.01f
#define MINF -std::numeric_limits<float>::max()
#define MAXF std::numeric_limits<float>::max()
#define BVH_STACK_SIZE      256
#define BOUNCE_STACK_SIZE   64
#define MAX_RESOLUTION      4096
#define DELTA_THETA         30 * M_PI / 180
#define DELTA_PHI           120 * M_PI / 180
const Eigen::Vector3f EPSILON_VECTOR3F = Eigen::Vector3f(EPSILON, EPSILON, EPSILON);

template <typename T>
struct Optional {
    bool has_val;
    T val;

    __host__ __device__ Optional() : has_val(false), val(T()) {}
    __host__ __device__ Optional(const T& v) : has_val(true), val(v) {}
    __host__ __device__ T value() const { return val; }
    __host__ __device__ operator bool() const { return has_val; }
};

__device__ inline Eigen::Vector3f to_world(const Eigen::Vector3f &a, const Eigen::Vector3f &N)
{
    Eigen::Vector3f B, C;
    if (std::fabs(N.x()) > std::fabs(N.y()))
    {
        float invLen = 1.0f / std::sqrt(N.x() * N.x() + N.z() * N.z());
        C = Eigen::Vector3f(N.z() * invLen, 0.0f, -N.x() *invLen);
    }
    else
    {
        float invLen = 1.0f / std::sqrt(N.y() * N.y() + N.z() * N.z());
        C = Eigen::Vector3f(0.0f, N.z() * invLen, -N.y() *invLen);
    }
    B = C.cross(N);
    return a.x() * B + a.y() * C + a.z() * N;
}

__device__ inline float get_cuda_random_float(curandState* rand_state)
{
    return curand_uniform(rand_state);
}

__device__ inline Eigen::Vector3f get_cuda_random_sphere_vector(const Eigen::Vector3f &N, curandState* rand_state)
{
    float x_1 = get_cuda_random_float(rand_state);
    float x_2 = get_cuda_random_float(rand_state);
    float z = std::fabs(1.0f - 2.0f * x_1);
    float r = std::sqrt(1.0f - z * z);
    float phi = static_cast<float>(2 * M_PI * x_2);
    Eigen::Vector3f local_vector(r * std::cos(phi), r * std::sin(phi), z);
    return to_world(local_vector, N);
}

__device__ inline Eigen::Vector3f get_cuda_random_specular_sphere_vector(const Eigen::Vector3f& out, const float delta_theta, const float delta_phi, curandState* rand_state)
{
    float eta_1 = 2 * get_cuda_random_float(rand_state) - 1;
    float eta_2 = 2 * get_cuda_random_float(rand_state) - 1;

    float r = out.norm();
    float theta_0 = std::acos(out.z() / r);

    float phi_0;
    if (fabs(out.x()) < 1e-5)
    {
        phi_0 = out.y() > 0.0f ? static_cast<float>(M_PI_2) : -static_cast<float>(M_PI_2);
    }
    else if (out.x() > 1e-5)
    {
        phi_0 = std::atan2f(out.y(), out.x());
    }
    else
    {
        phi_0 = std::atan2f(out.y(), out.x());
        // phi_0 = out.y() > 0.0f ? std::atan2f(out.y(), out.x()) + static_cast<float>(M_PI_2) : std::atan2f(out.y(), out.x()) - static_cast<float>(M_PI_2);
    }

    float theta = theta_0 + eta_1 * delta_theta;
    float phi = phi_0 + eta_2 * delta_phi;
    return Eigen::Vector3f(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
}

__device__ inline float get_cuda_sphere_sample_inv_pdf()
{
    return static_cast<float>(2.0f * M_PI);
}

__device__ inline float get_cuda_specular_sphere_sample_inv_pdf(const float delta_theta, const float delta_phi)
{
    return (1.0f - std::cos(delta_theta)) * delta_phi;
}

__device__ inline unsigned int get_cuda_random_uint(curandState* rand_state)
{
    return curand(rand_state);
}

__device__ inline float cuda_maxf(float x, float y)
{
    return x > y ? x : y;
}

__device__ inline float cuda_minf(float x, float y)
{
    return x < y ? x : y;
}

__device__ inline float clamp(const float &lo, const float &hi, const float &v)
{
    return cuda_maxf(lo, cuda_minf(hi, v));
}

#endif