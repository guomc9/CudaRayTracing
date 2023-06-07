#ifndef _RENDER_CUH_
#define _RENDER_CUH_

#include "DeviceBVH.cuh"
#include "DeviceLights.cuh"
#include "DeviceStack.cuh"
#include "Global.h"
#include "Scene.h"
#include "Ray.cuh"
#include <cuda_runtime.h>

__device__ inline void sample_light(LightSamplePayload& payload, curandState* rand_state, DeviceLight* light)
{
    payload = light->sample(rand_state);
}

__device__ inline bool blocked(Ray ray, float t_to_light, DeviceBVH* device_bvh, DeviceStack<int, BVH_STACK_SIZE>* stack)
{
    HitPayload hit = device_bvh->intersect(ray.get_origin(), ray.get_dir(), ray.get_inv_dir(), stack);
    if(t_to_light - hit.t > EPSILON_1000)
    {
        return true;
    }
    return false;
}

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
                Eigen::Vector3f reflect_dir = get_cuda_random_sphere_vector(hit.normal, rand_state).normalized();
                tmp = Ray(hit.pos, reflect_dir);
            }
        }
    }
    
    // backward bounce-stack
    HitPayload to_hit;
    HitPayload pre_hit;
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
            tmp_L_dir = to_hit.m.get_ke();
        }
        else
        {
            Eigen::Vector3f pos = to_hit.pos;
            Eigen::Vector3f f_r = to_hit.m.get_kd();
            Eigen::Vector3f normal = to_hit.normal;
            // figure out the contribution from lights to to_hit
            for(int i=0;i<lights->get_lights_size();i++)
            {
                for(int j=0;j<light_sample_n;j++)
                {
                    LightSamplePayload light_sample_payload;
                    sample_light(light_sample_payload, rand_state, &lights->get_lights()[i]);
                    Eigen::Vector3f dist = light_sample_payload.pos - pos;
                    Eigen::Vector3f dir = dist.normalized();
                    Ray back(pos, dir);
                    // if the backward ray has not been blocked
                    if(!blocked(back, dist.x() / dir.x(), device_bvh, bvh_stack))
                    {
                        Eigen::Vector3f L_i = light_sample_payload.emit;
                        Eigen::Vector3f light_normal = light_sample_payload.normal;
                        float t_to_light = dist.norm();
                        float t_to_light_p2 = t_to_light * t_to_light;
                        float inv_pdf = light_sample_payload.inv_pdf;
                        float cos_theta = fabsf(dir.dot(normal));
                        float cos_theta_2 =  fabsf(dir.dot(light_normal));
                        tmp_L_dir += L_i.cwiseProduct(f_r) * cos_theta * cos_theta_2 * inv_pdf / t_to_light_p2 / light_sample_n;
                    }
                }
            }

            // figure out the contribution from other reflectors
            float inv_pdf = get_cuda_sphere_sample_inv_pdf();
            if(!is_final_hit)
            {
                float cos_theta = fabsf((pre_hit.pos - pos).normalized().dot(normal));
                tmp_L_indir = L.cwiseProduct(f_r) * cos_theta * inv_pdf / P_RR;
            }
            else
            {
                is_final_hit = false;
            }
        }
        pre_hit = to_hit;
        L = tmp_L_indir + tmp_L_dir;
    }
    return L;
}

__global__ void video_render_kernel(unsigned int width, unsigned int height, Eigen::Vector3f eye_pos, Eigen::Matrix4f view, float fovY, unsigned int spp, float P_RR, int light_sample_n, DeviceBVH* device_bvh, uchar3* device_frame_buffer, DeviceLights* lights, DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stacks, DeviceStack<int, BVH_STACK_SIZE>* bvh_stacks)
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
            Eigen::Vector4f tmp_dir(-x, y, 1, 1);
            tmp_dir = view * tmp_dir;
            Eigen::Vector3f dir = Eigen::Vector3f(tmp_dir.x() / tmp_dir.w(), tmp_dir.y() / tmp_dir.w(), tmp_dir.z() / tmp_dir.w()).normalized();
            Ray ray(eye_pos, dir);
            temp_color += cast_ray(ray, 0, light_sample_n, P_RR, &rand_state, device_bvh, lights, &bounce_stacks[pixel_index], &bvh_stacks[pixel_index]) / spp;
        }
        // OpenCV BGR
        device_frame_buffer[pixel_index] = make_uchar3(255 * std::pow(clamp(0, 1, temp_color.z()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.y()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.x()), 0.6f));
    }
}

__global__ void rotation_render_kernel(unsigned int width, unsigned int height, Eigen::Vector3f eye_pos, Eigen::Matrix4f view, float fovY, unsigned int spp, float P_RR, int light_sample_n, DeviceBVH* device_bvh, uchar3* device_frame_buffer, DeviceLights* lights, DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stacks, DeviceStack<int, BVH_STACK_SIZE>* bvh_stacks)
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
            Eigen::Vector4f tmp_dir(-x, y, 1, 1);
            tmp_dir = view * tmp_dir;
            Eigen::Vector3f dir = Eigen::Vector3f(tmp_dir.x() / tmp_dir.w(), tmp_dir.y() / tmp_dir.w(), tmp_dir.z() / tmp_dir.w()).normalized();
            Ray ray(eye_pos, dir);
            temp_color += cast_ray(ray, 0, light_sample_n, P_RR, &rand_state, device_bvh, lights, &bounce_stacks[pixel_index], &bvh_stacks[pixel_index]) / spp;
        }
        device_frame_buffer[pixel_index] = make_uchar3(255 * std::pow(clamp(0, 1, temp_color.x()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.y()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.z()), 0.6f));
    }
}

__global__ void image_render_kernel(unsigned int width, unsigned int height, Eigen::Vector3f eye_pos, float fovY, unsigned int spp, float P_RR, int light_sample_n, DeviceBVH* device_bvh, uchar3* device_frame_buffer, DeviceLights* lights, DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* bounce_stacks, DeviceStack<int, BVH_STACK_SIZE>* bvh_stacks)
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
            Eigen::Vector3f dir = Eigen::Vector3f(-x, y, 1).normalized();
            Ray ray(eye_pos, dir);
            temp_color += cast_ray(ray, 0, light_sample_n, P_RR, &rand_state, device_bvh, lights, &bounce_stacks[pixel_index], &bvh_stacks[pixel_index]) / spp;
        }
        device_frame_buffer[pixel_index] = make_uchar3(255 * std::pow(clamp(0, 1, temp_color.x()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.y()), 0.6f), 255 * std::pow(clamp(0, 1, temp_color.z()), 0.6f));
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

        // to device parameters
        DeviceBVH* device_bvh;
        size_t pitch;
        uchar3* device_frame_buffer;   // RGB
        DeviceLights* device_lights;
        DeviceStack<int, BVH_STACK_SIZE>* device_bvh_stacks;
        DeviceStack<HitPayload, BOUNCE_STACK_SIZE>* device_bounce_stacks;
        unsigned char* frame_buffer;  // RGB

    public:
        Render(Scene* _scene, unsigned int _spp=16, float _P_RR=0.8f, unsigned int _light_sample_n=1)
        {
            scene = _scene;
            spp = _spp;
            P_RR = _P_RR;
            light_sample_n = _light_sample_n;

            // load bvh from host to device
            host_bvh = new DeviceBVH(scene->get_bvh());
            cudaMalloc((void**)&device_bvh, sizeof(DeviceBVH));
            cudaMemcpy(device_bvh, host_bvh, sizeof(DeviceBVH), cudaMemcpyHostToDevice);

            // create frame_buffer in device
            frame_buffer = new unsigned char[3 * scene->get_pixels()];
            cudaMalloc((void**)&device_frame_buffer, sizeof(uchar3) * scene->get_pixels());

            // load lights from host device
            host_lights = new DeviceLights(scene->get_light_objs());
            cudaMalloc((void**)&device_lights, sizeof(DeviceLights));
            cudaMemcpy(device_lights, host_lights, sizeof(DeviceLights), cudaMemcpyHostToDevice);

            //create stacks in device
            cudaMalloc((void**)&device_bvh_stacks, sizeof(DeviceStack<int, BVH_STACK_SIZE>) * scene->get_pixels());
            cudaMalloc((void**)&device_bounce_stacks, sizeof(DeviceStack<HitPayload, BOUNCE_STACK_SIZE>) * scene->get_pixels());
        }

        void run_video(Eigen::Vector3f eye_pos, float fovY, Eigen::Matrix4f view)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((scene->get_width() + threadsPerBlock.x - 1) / threadsPerBlock.x, (scene->get_height() + threadsPerBlock.y - 1) / threadsPerBlock.y);
            video_render_kernel<<<numBlocks, threadsPerBlock>>>(scene->get_width(), scene->get_height(), eye_pos, view, fovY, spp, P_RR, light_sample_n, device_bvh, device_frame_buffer, device_lights, device_bounce_stacks, device_bvh_stacks);
            cudaDeviceSynchronize();
            cudaMemcpy(frame_buffer, device_frame_buffer, sizeof(uchar3) * scene->get_pixels(), cudaMemcpyDeviceToHost);
        }

        void run_rotation_data(Eigen::Vector3f eye_pos, float fovY, Eigen::Matrix4f view)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((scene->get_width() + threadsPerBlock.x - 1) / threadsPerBlock.x, (scene->get_height() + threadsPerBlock.y - 1) / threadsPerBlock.y);
            rotation_render_kernel<<<numBlocks, threadsPerBlock>>>(scene->get_width(), scene->get_height(), eye_pos, view, fovY, spp, P_RR, light_sample_n, device_bvh, device_frame_buffer, device_lights, device_bounce_stacks, device_bvh_stacks);
            cudaDeviceSynchronize();
            cudaMemcpy(frame_buffer, device_frame_buffer, sizeof(uchar3) * scene->get_pixels(), cudaMemcpyDeviceToHost);
        }

        void run_image(Eigen::Vector3f eye_pos, float fovY)
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((scene->get_width() + threadsPerBlock.x - 1) / threadsPerBlock.x, (scene->get_height() + threadsPerBlock.y - 1) / threadsPerBlock.y);
            image_render_kernel<<<numBlocks, threadsPerBlock>>>(scene->get_width(), scene->get_height(), eye_pos, fovY, spp, P_RR, light_sample_n, device_bvh, device_frame_buffer, device_lights, device_bounce_stacks, device_bvh_stacks);
            cudaDeviceSynchronize();
            cudaMemcpy(frame_buffer, device_frame_buffer, sizeof(uchar3) * scene->get_pixels(), cudaMemcpyDeviceToHost);
        }

        void free()
        {
            host_bvh->free();
            host_lights->free();
            cudaFree(device_bvh);
            cudaFree(device_lights);
        }

        void save_frame_buffer(const char* save_path)
        {
            stbi_write_png(save_path, scene->get_width(), scene->get_height(), 3, frame_buffer, scene->get_width() * 3);
        }

        unsigned char* get_frame_buffer() const
        {
            return frame_buffer;
        }
};

#endif