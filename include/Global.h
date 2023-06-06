#ifndef _GLOBAL_H_
#define _GLOBAL_H_
#define  _USE_MATH_DEFINES
#include <random>
#include <curand_kernel.h>
#include <math.h>
#include "Eigen/Dense"

#define EPSILON         0.001f
#define EPSILON_10       0.01f
#define EPSILON_100       0.1f
#define EPSILON_1000      1.0f
#define MINF -std::numeric_limits<float>::max()
#define MAXF std::numeric_limits<float>::max()
#define BVH_STACK_SIZE      1024
#define BOUNCE_STACK_SIZE   32

const Eigen::Vector3f EPSILON_VECTOR3F = Eigen::Vector3f(EPSILON, EPSILON, EPSILON);

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

__device__ inline float get_cuda_sphere_sample_inv_pdf()
{
    return static_cast<float>(0.5f / M_PI);
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