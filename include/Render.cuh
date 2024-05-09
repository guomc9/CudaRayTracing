#ifndef _RENDER_CUH_
#define _RENDER_CUH_

#include "DeviceBVH.cuh"
#include "DeviceLights.cuh"
#include "DeviceStack.cuh"
#include "Global.h"
#include "Scene.h"
#include "Ray.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

__device__ inline void sample_light(LightSamplePayload& payload, curandState* rand_state, DeviceLight* light)
{
    payload = light->sample(rand_state);
}

__device__ inline bool blocked(Ray ray, float t_to_light, DeviceBVH* device_bvh, DeviceStack<int, BVH_STACK_SIZE>* stack)
{
    HitPayload hit = device_bvh->intersect(ray.get_origin(), ray.get_dir(), ray.get_inv_dir(), stack);
    if(t_to_light - hit.t > EPSILON)
    {
        return true;
    }
    return false;
}

// __device__ inline bool blocked_or_through(Ray ray, LightSamplePayload& light_sample, float t_to_light, DeviceBVH* device_bvh, DeviceStack<int, BVH_STACK_SIZE>* stack)
// {
//     HitPayload hit = device_bvh->intersect(ray.get_origin(), ray.get_dir(), ray.get_inv_dir(), stack);
//     if (t_to_light - hit.t > EPSILON)
//     {
//         if (hit.m.has_emission())
//         {
//             light_sample = LightSamplePayload(hit.pos, hit.normal, hit.m.get_ke(), hit.area);
//             return false;
//         }
//         else
//         {
//             return true;
//         }
//     }
//     return false;
// }

__device__ Eigen::Vector3f cast_ray(Ray ray, int depth, int light_sample_n, float P_RR, curandState* rand_state, DeviceBVH* device_bvh, DeviceLights* lights, DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stack, DeviceStack<int, BVH_STACK_SIZE>* bvh_stack)
{
    bool done = false;
    Ray tmp = ray;
    bounce_stack->clear();
    // push bounce-stack
    while(!done)
    {
        HitPayload hit = device_bvh->intersect(tmp.get_origin(), tmp.get_dir(), tmp.get_inv_dir(), bvh_stack);
        bounce_stack->push(hit);
        // if hit didn't happend, directly hit the light, or bounce-stack is filled-up, stop push
        if (!hit.happend || hit.m.has_emission() || bounce_stack->is_full())
        {
            done = true;
        }
        else
        {
            float RR = get_cuda_random_float(rand_state);
            // if the bounce ray is died, stop push
            if(RR > P_RR)
            {
                done = true;
            }
            // otherwise, continue bounce
            else
            {
                Eigen::Vector3f reflect_dir;
                if (hit.m.get_mode() == DIFFUSE)
                {
                    reflect_dir = get_cuda_random_sphere_vector(hit.normal, rand_state).normalized();
                }
                else
                {
                    Eigen::Vector3f in = tmp.get_dir();
                    Eigen::Vector3f out = in - 2.f * in.dot(hit.normal) * hit.normal;
                    reflect_dir = get_cuda_random_specular_sphere_vector(out, DELTA_THETA, DELTA_PHI, rand_state).normalized();
                }
                tmp = Ray(hit.pos, reflect_dir);
            }
        }
    }
    
    // backward bounce-stack
    HitPayload to_hit;
    HitPayload pre_hit;
    bool specular_trace = false;
    bool is_final_hit = true;
    Eigen::Vector3f L = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    while(!bounce_stack->is_empty())
    {
        to_hit = bounce_stack->pop();
        // if to_hit didn't happend
        if(!to_hit.happend)
        {
            continue;
        }
        Eigen::Vector3f tmp_L_dir = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f tmp_L_indir = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        // if to_hit hit a light
        if(to_hit.m.has_emission())
        {
            if (bounce_stack->is_empty())
                tmp_L_dir = to_hit.m.get_ke();
            else
                is_final_hit = false;
        }
        else
        {
            if (to_hit.m.get_mode() == DIFFUSE)
            {
                Eigen::Vector3f pos = to_hit.pos;
                Eigen::Vector3f f_r = to_hit.m.get_kd() / static_cast<float>(M_PI);
                Eigen::Vector3f normal = to_hit.normal;
                // figure out the contribution from lights to to_hit
                for(int i = 0; i < lights->get_lights_size(); i++)
                {
                    for(int j = 0; j < light_sample_n; j++)
                    {
                        LightSamplePayload light_sample_payload;
                        sample_light(light_sample_payload, rand_state, &lights->get_lights()[i]);
                        Eigen::Vector3f dist = light_sample_payload.pos - pos;
                        Eigen::Vector3f dir = dist.normalized();
                        Ray back(pos, dir);
                        // if the backward ray has not been blocked
                        if (!blocked(back, dist.x() / dir.x(), device_bvh, bvh_stack))
                        {
                            Eigen::Vector3f L_i = light_sample_payload.emit;
                            Eigen::Vector3f light_normal = light_sample_payload.normal;
                            float t_to_light = dist.norm();
                            float t_to_light_p2 = t_to_light * t_to_light;
                            float inv_pdf = light_sample_payload.inv_pdf;
                            float cos_theta = dir.dot(normal);
                            float cos_theta_2 = -dir.dot(light_normal);
                            cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                            cos_theta_2 = cos_theta_2 > 0.0f ? cos_theta_2 : 0.0f;
                            tmp_L_dir += L_i.cwiseProduct(f_r) * cos_theta * cos_theta_2 * inv_pdf / t_to_light_p2 / light_sample_n;
                        }
                    }
                }
                // figure out the contribution from other reflectors
                if (!is_final_hit)
                {
                    float inv_pdf = get_cuda_sphere_sample_inv_pdf();
                    float cos_theta = (pre_hit.pos - pos).normalized().dot(normal);
                    cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                    tmp_L_indir = L.cwiseProduct(f_r) * cos_theta * inv_pdf / P_RR;
                }
                else
                {
                    is_final_hit = false;
                }
            }
            else if (to_hit.m.get_mode() == SPECULAR)
            {
                if (!is_final_hit)
                {
                    float log_shininess = log10f(to_hit.m.get_ns());
                    if (log_shininess < 1.0f)
                        printf("ns: %f, log_shininess: %f\n", to_hit.m.get_ns(), log_shininess);
                    
                    log_shininess = 1.f;
                    if (pre_hit.m.has_emission())
                    {
                        Eigen::Vector3f f_r = to_hit.m.get_kd();
                        float inv_pdf = get_cuda_sphere_sample_inv_pdf() / 2;
                        float cos_theta = (pre_hit.pos - to_hit.pos).normalized().dot(to_hit.normal);
                        cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                        tmp_L_dir = log_shininess * pre_hit.m.get_ke().cwiseProduct(f_r) * cos_theta * inv_pdf / P_RR;
                    }
                    else
                    {
                        Eigen::Vector3f f_r = to_hit.m.get_kd();
                        float inv_pdf = get_cuda_sphere_sample_inv_pdf() / 2;
                        float cos_theta = (pre_hit.pos - to_hit.pos).normalized().dot(to_hit.normal);
                        cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                        tmp_L_indir = log_shininess * L.cwiseProduct(f_r) * cos_theta * inv_pdf / P_RR;
                    }
                }
                else
                {
                    is_final_hit = false;
                }
            }
        }
        pre_hit = to_hit;
        L = tmp_L_indir + tmp_L_dir;
        // if (bounce_stack->size() == 1 && specular_trace)
        //     printf("L:(%f,%f,%f), to_hit:(%f,%f,%f)\n", L.x(), L.y(), L.z(), to_hit.pos.x(), to_hit.pos.y(), to_hit.pos.z());
    }
    return L;
}

__device__ Eigen::Vector3f cast_ray_v2(Ray ray, int depth, int light_sample_n, float P_RR, curandState* rand_state, DeviceBVH* device_bvh, DeviceLights* lights, DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stack, DeviceStack<int, BVH_STACK_SIZE>* bvh_stack)
{
    bool done = false;
    Ray tmp = ray;
    bounce_stack->clear();
    // push bounce-stack
    while(!done)
    {
        HitPayload hit = device_bvh->intersect(tmp.get_origin(), tmp.get_dir(), tmp.get_inv_dir(), bvh_stack);
        bounce_stack->push(hit);
        // if hit didn't happend, directly hit the light, or bounce-stack is filled-up, stop push
        if(!hit.happend || hit.m.has_emission() || bounce_stack->is_full())
        {
            done = true;
        }
        else
        {
            float RR = get_cuda_random_float(rand_state);
            // if the bounce ray is died, stop push
            if(RR > P_RR)
            {
                done = true;
            }
            // otherwise, continue bounce
            else
            {
                Eigen::Vector3f reflect_dir;
                reflect_dir = get_cuda_random_sphere_vector(hit.normal, rand_state).normalized();
                tmp = Ray(hit.pos, reflect_dir);
            }
        }
    }
    
    // backward bounce-stack
    HitPayload to_hit;
    HitPayload pre_hit;
    bool specular_trace = false;
    bool is_final_hit = true;
    Eigen::Vector3f L = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    while(!bounce_stack->is_empty())
    {
        to_hit = bounce_stack->pop();
        // if to_hit didn't happend
        if(!to_hit.happend)
        {
            continue;
        }
        Eigen::Vector3f tmp_L_dir = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        Eigen::Vector3f tmp_L_indir = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        // if to_hit hit a light
        if(to_hit.m.has_emission())
        {
            if (bounce_stack->is_empty())
                tmp_L_dir = to_hit.m.get_ke();
            else
                is_final_hit = false;
        }
        else
        {
            Eigen::Vector3f pos = to_hit.pos;
            Eigen::Vector3f f_r = to_hit.m.get_kd() / static_cast<float>(M_PI);
            Eigen::Vector3f normal = to_hit.normal;
            // figure out the contribution from lights to to_hit
            for(int i = 0; i < lights->get_lights_size(); i++)
            {
                for(int j = 0; j < light_sample_n; j++)
                {
                    LightSamplePayload light_sample_payload;
                    sample_light(light_sample_payload, rand_state, &lights->get_lights()[i]);
                    Eigen::Vector3f dist = light_sample_payload.pos - pos;
                    Eigen::Vector3f dir = dist.normalized();
                    Ray back(pos, dir);
                    // if the backward ray has not been blocked
                    if (!blocked(back, dist.x() / dir.x(), device_bvh, bvh_stack))
                    {
                        Eigen::Vector3f L_i = light_sample_payload.emit;
                        Eigen::Vector3f light_normal = light_sample_payload.normal;
                        float t_to_light = dist.norm();
                        float t_to_light_p2 = t_to_light * t_to_light;
                        float inv_pdf = light_sample_payload.inv_pdf;
                        float cos_theta = dir.dot(normal);
                        float cos_theta_2 = -dir.dot(light_normal);
                        cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                        cos_theta_2 = cos_theta_2 > 0.0f ? cos_theta_2 : 0.0f;
                        tmp_L_dir += L_i.cwiseProduct(f_r) * cos_theta * cos_theta_2 * inv_pdf / t_to_light_p2 / light_sample_n;
                    }
                }
            }
            // figure out the contribution from other reflectors
            if (!is_final_hit)
            {
                float inv_pdf = get_cuda_sphere_sample_inv_pdf();
                float cos_theta = (pre_hit.pos - pos).normalized().dot(normal);
                cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                tmp_L_indir = L.cwiseProduct(f_r) * cos_theta * inv_pdf / P_RR;
                if (to_hit.m.get_mode() == SPECULAR)
                {
                    float ns = to_hit.m.get_ns();
                    float delta_coeff = (exp(25 / ns) - 1) / (M_E - 1);
                    Eigen::Vector3f in = to_hit.from_dir.normalized();
                    Eigen::Vector3f out = in - 2.f * in.dot(to_hit.normal) * to_hit.normal;
                    auto ref = get_cuda_random_specular_sphere_vector(out, delta_coeff * DELTA_THETA, delta_coeff * DELTA_PHI, rand_state).normalized();
                    
                    auto probe = Ray(to_hit.pos, ref);
                    HitPayload hit = device_bvh->intersect(probe.get_origin(), probe.get_dir(), probe.get_inv_dir(), bvh_stack);
                    if (hit.m.has_emission())
                    {
                        float log_shininess = log10f(to_hit.m.get_ns());
                        float shininess_coeff = (log_shininess * 0.5 + 1);
                        float inv_pdf = get_cuda_sphere_sample_inv_pdf() / 8.f;
                        float cos_theta = (hit.pos - to_hit.pos).normalized().dot(normal);
                        cos_theta = cos_theta > 0.0f ? cos_theta : 0.0f;
                        auto temp = shininess_coeff * hit.m.get_ke().cwiseProduct(to_hit.m.get_kd()) * cos_theta * inv_pdf;
                        tmp_L_dir += temp;
                    }
                }
            }
            else
            {
                is_final_hit = false;
            }

        }
        pre_hit = to_hit;
        L = tmp_L_indir + tmp_L_dir;
        // if (bounce_stack->size() == 1 && specular_trace)
        //     printf("L:(%f,%f,%f), to_hit:(%f,%f,%f)\n", L.x(), L.y(), L.z(), to_hit.pos.x(), to_hit.pos.y(), to_hit.pos.z());
    }
    return L;
}

__global__ void view_render_kernel(unsigned int width, unsigned int height, Eigen::Vector3f eye_pos, Eigen::Matrix3f inv_view_mat, float fovY, unsigned int spp, float P_RR, int light_sample_n, DeviceBVH* device_bvh, uchar3* device_frame_buffer, DeviceLights* lights, DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stacks, DeviceStack<int, BVH_STACK_SIZE>* bvh_stacks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height)
    {
        int pixel_index = j * width + i;
        Eigen::Vector3f temp_color = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        float scale = tanf(fovY / 2);
        float ar = static_cast<float>(width) / height;
        curandState rand_state;
        curand_init(clock() + pixel_index, 0, 0, &rand_state);
        for(unsigned int k = 0; k < spp ; k++)
        {
            float x = (2 * (i + get_cuda_random_float(&rand_state)) / width - 1) * scale * ar;
            float y = (1 - 2 * (j + get_cuda_random_float(&rand_state)) / height) * scale;
            Eigen::Vector3f dir = inv_view_mat * Eigen::Vector3f(-x, y, 1).normalized();
            Ray ray(eye_pos, dir);
            temp_color += cast_ray_v2(ray, 0, light_sample_n, P_RR, &rand_state, device_bvh, lights, &bounce_stacks[pixel_index], &bvh_stacks[pixel_index]) / spp;
        }
        device_frame_buffer[pixel_index] = make_uchar3(255 * std::pow(clamp(0, 1, temp_color.x()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.y()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.z()), 0.6f));
        // device_frame_buffer[pixel_index] = make_uchar3(255 * std::pow(clamp(0, 1, temp_color.x()), 0.5f), 255 * std::pow(clamp(0, 1, temp_color.y()), 0.5f), 255 * std::pow(clamp(0, 1, temp_color.z()), 0.5f));
        // device_frame_buffer[pixel_index] = make_uchar3(255 * std::pow(clamp(0, 1, temp_color.x()), 0.4f), 255 * std::pow(clamp(0, 1, temp_color.y()), 0.4f), 255 * std::pow(clamp(0, 1, temp_color.z()), 0.4f));
    }
}


class Render
{
    private:
        Scene* scene;
        unsigned int lights_size;
        unsigned int spp;
        unsigned int light_sample_n;
        float P_RR;

        // host parameters
        DeviceBVH* host_bvh;
        DeviceLights* host_lights;

        // device parameters
        DeviceBVH* device_bvh;
        uchar3* device_frame_buffer;   // RGB
        DeviceLights* device_lights;
        DeviceStack<int, BVH_STACK_SIZE>* device_bvh_stacks;
        DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* device_bounce_stacks;
        unsigned char* frame_buffer;  // RGB

    public:
        Render(Scene* _scene, unsigned int _spp=16, float _P_RR=0.8f, unsigned int _light_sample_n=1)
        {
            cudaError_t err;
            scene = _scene;
            spp = _spp;
            P_RR = _P_RR;
            light_sample_n = _light_sample_n;

            // load bvh from host to device
            try
            {
                host_bvh = new DeviceBVH(scene->get_bvh());
                err = cudaMalloc((void**)&device_bvh, sizeof(DeviceBVH));
                err = cudaMemcpy(device_bvh, host_bvh, sizeof(DeviceBVH), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_bvh to cuda.");
                }
                // create frame_buffer in device
                frame_buffer = new unsigned char[3 * scene->get_pixels()];
                err = cudaMalloc((void**)&device_frame_buffer, sizeof(uchar3) * scene->get_pixels());
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_frame_buffer to cuda.");
                }
                // load lights from host device
                host_lights = new DeviceLights(scene->get_light_objs());
                err = cudaMalloc((void**)&device_lights, sizeof(DeviceLights));
                err = cudaMemcpy(device_lights, host_lights, sizeof(DeviceLights), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_lights to cuda.");
                }
                //create stacks in device
                err = cudaMalloc((void**)&device_bvh_stacks, sizeof(DeviceStack<int, BVH_STACK_SIZE>) * scene->get_pixels());
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_bvh_stacks to cuda.");
                }
                err = cudaMalloc((void**)&device_bounce_stacks, sizeof(DeviceStack<HitPayload, BOUNCE_STACK_SIZE>) * scene->get_pixels());
                if (err != cudaSuccess)
                {
                    printf("CUDA error: %s\n", cudaGetErrorString(err));
                    throw std::runtime_error("Render.device_bounce_stacks to cuda.");
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
            }
        }

        void run_view(Eigen::Vector3f eye_pos, Eigen::Matrix3f inv_view_mat, float fovY, cudaGraphicsResource* cuda_pbo_resource)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((scene->get_width() + threadsPerBlock.x - 1) / threadsPerBlock.x, (scene->get_height() + threadsPerBlock.y - 1) / threadsPerBlock.y);
            view_render_kernel<<<numBlocks, threadsPerBlock>>>(scene->get_width(), scene->get_height(), eye_pos, inv_view_mat, fovY, spp, P_RR, light_sample_n, device_bvh, device_frame_buffer, device_lights, device_bounce_stacks, device_bvh_stacks);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
            
            // write device_frame_buffer to pbo
            err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
            size_t num_bytes = sizeof(uchar3) * scene->get_pixels(); 
            uchar3* cuda_pbo_pointer;
            err = cudaGraphicsResourceGetMappedPointer((void **)&cuda_pbo_pointer, &num_bytes, cuda_pbo_resource);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
            err = cudaMemcpy((void *)cuda_pbo_pointer, (void *)device_frame_buffer, sizeof(uchar3) * scene->get_pixels(), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
            err = cudaMemcpy((void *)frame_buffer, (void *)device_frame_buffer, sizeof(uchar3) * scene->get_pixels(), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
            err = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
            }
            
        }

        void free()
        {
            delete[] frame_buffer;
            host_bvh->free();
            host_lights->free();
            cudaFree(device_bvh);
            cudaFree(device_lights);
            cudaFree(device_bounce_stacks);
            cudaFree(device_bvh_stacks);
            cudaFree(device_frame_buffer);
        }

        void save_frame_buffer(const char* save_path)
        {
            printf("%s\n", save_path);
            stbi_write_png(save_path, scene->get_width(), scene->get_height(), 3, frame_buffer, scene->get_width() * 3);
        }

        unsigned char* get_frame_buffer() const
        {
            return frame_buffer;
        }

        void set_height(const unsigned int& height)
        {
            scene->set_height(height);
            delete[] frame_buffer;
            cudaError_t err = cudaFree(device_frame_buffer);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                return ;
            }
            frame_buffer = NULL;
            device_frame_buffer = NULL;
            frame_buffer = new unsigned char[3 * scene->get_pixels()];
            err = cudaMalloc((void**)&device_frame_buffer, sizeof(uchar3) * scene->get_pixels());
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                return ;
            }
        }

        void set_width(const unsigned int& width)
        {
            scene->set_width(width);
            delete[] frame_buffer;
            cudaError_t err = cudaFree(device_frame_buffer);
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                return ;
            }
            frame_buffer = NULL;
            device_frame_buffer = NULL;
            frame_buffer = new unsigned char[3 * scene->get_pixels()];
            err = cudaMalloc((void**)&device_frame_buffer, sizeof(uchar3) * scene->get_pixels());
            
            if (err != cudaSuccess)
            {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                return ;
            }
        }

        void set_spp(const int& spp)
        {
            this->spp = spp;
        }

        void set_P_RR(const float& P_RR)
        {
            this->P_RR = P_RR;
        }

        void set_light_sample_n(const int& light_sample_n)
        {
            this->light_sample_n = light_sample_n;
        }
};

#endif