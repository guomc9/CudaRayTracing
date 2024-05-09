#ifndef _DEVICE_BVH_CUH_
#define _DEVICE_BVH_CUH_
#include "Eigen/Dense"
#include "Payload.cuh"
#include "DeviceTriangle.cuh"
#include "DeviceStack.cuh"
#include "BVH.h"

struct DeviceBVHNode
{
    int lc, rc;
    unsigned int n;
    int it;
    Eigen::Vector3f AA, BB;

    __device__ __host__ DeviceBVHNode(BVHNode node)
    {
        lc = node.lc;
        rc = node.rc;
        it = node.it;
        n = node.n;
        AA = node.AA;
        BB = node.BB;
    }

    __device__ bool is_leaf()
    {
        return lc < 0 && rc < 0;
    }

    __device__ HitPayload hit(DeviceTriangle* ts, Eigen::Vector3f origin, Eigen::Vector3f dir, Eigen::Vector3f inv_dir)
    {
        HitPayload payload;
        for(int i = it; i < static_cast<int>(it + n); i++)
        {
            HitPayload tmp = ts[i].get_intersection(origin, dir);
            if(tmp.t > EPSILON && tmp.t < payload.t)
            {
                payload = tmp;
            }
        }
        return payload;
    }
};

struct DeviceBVH
{
    unsigned int* root_index;
    DeviceBVHNode* nodes;
    DeviceTriangle* triangles;

    DeviceBVH(BVH& bvh)
    {
        unsigned int host_root_index = bvh.get_root_index();
        cudaError_t err;
        err = cudaMalloc((void**)&root_index, sizeof(unsigned int));
        err = cudaMemcpy(root_index, &host_root_index, sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("DeviceBVH.root_index to cuda.");
        }

        std::vector<DeviceBVHNode> host_nodes(bvh.get_nodes().begin(), bvh.get_nodes().end());
        err = cudaMalloc((void**)&nodes, sizeof(DeviceBVHNode) * bvh.get_nodes_size());
        err = cudaMemcpy(nodes, host_nodes.data(), sizeof(DeviceBVHNode) * host_nodes.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("DeviceBVH.nodes to cuda.");
        }
        std::vector<DeviceTriangle> host_triangles(bvh.get_triangles().begin(), bvh.get_triangles().end());
        err = cudaMalloc((void**)&triangles, sizeof(DeviceTriangle) * bvh.get_triangles_size());
        err = cudaMemcpy(triangles, host_triangles.data(), sizeof(DeviceTriangle) * host_triangles.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("DeviceBVH.triangles to cuda.");
        }
    }

    __device__ unsigned int get_root_index()
    {
        return *root_index;
    }

    __device__ HitAABBPayload hit_AABB(int node_index, Eigen::Vector3f origin, Eigen::Vector3f dir, Eigen::Vector3f inv_dir)
    {
        if(node_index < 0)
        {
            return HitAABBPayload();
        }
        DeviceBVHNode node = nodes[node_index];
        Eigen::Vector3f OA = node.AA - origin;
        Eigen::Vector3f OB = node.BB - origin;
        
        Eigen::Vector3f t_min = OA.cwiseProduct(inv_dir);
        Eigen::Vector3f t_max = OB.cwiseProduct(inv_dir);

        if(dir.x() < 0)
        {
            float tmp = t_min.x();
            t_min.x() = t_max.x();
            t_max.x() = tmp;
        }
        if(dir.y() < 0)
        {
            float tmp = t_min.y();
            t_min.y() = t_max.y();
            t_max.y() = tmp;
        }
        if(dir.z() < 0)
        {
            float tmp = t_min.z();
            t_min.z() = t_max.z();
            t_max.z() = tmp;
        }
        float t_enter = cuda_maxf(cuda_maxf(t_min.x(), t_min.y()), t_min.z());
        float t_exit = cuda_minf(cuda_minf(t_max.x(), t_max.y()), t_max.z());
        
        if(t_enter <= t_exit + EPSILON && t_exit >= 0)
        {
            return HitAABBPayload(true, t_enter, t_exit);
        }
        return HitAABBPayload();
    }

    __device__ HitPayload intersect(Eigen::Vector3f origin, Eigen::Vector3f dir, Eigen::Vector3f inv_dir, DeviceStack<int, BVH_STACK_SIZE>* stack)
    {
        stack->clear();
        stack->push(*root_index);
        HitPayload closest_hit;
        while (!stack->is_empty())
        {
            int current_index = stack->pop();
            if (current_index < 0)
            {
                continue;
            }
            DeviceBVHNode node = nodes[current_index];
            if (node.is_leaf())
            {
                HitPayload hit = node.hit(triangles, origin, dir, inv_dir);
                if (hit.t < closest_hit.t)
                {
                    closest_hit = hit;
                }
            }
            else
            {
                HitAABBPayload hit_left_aabb = hit_AABB(node.lc, origin, dir, inv_dir);
                HitAABBPayload hit_right_aabb = hit_AABB(node.rc, origin, dir, inv_dir);

                if (hit_left_aabb.happend && hit_right_aabb.happend)
                {
                    stack->push(node.lc);
                    stack->push(node.rc);
                }
                else if (hit_left_aabb.happend)
                {
                    stack->push(node.lc);
                }
                else if (hit_right_aabb.happend)
                {
                    stack->push(node.rc);
                }
            }
        }
        return closest_hit;
    }

    void free()
    {
        cudaFree(root_index);
        cudaFree(nodes);
        cudaFree(triangles);
    }
};

#endif