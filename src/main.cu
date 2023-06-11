#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Loader.h"
#include "Object.h"
#include "Scene.h"
#include "Render.cuh"
#include "Eigen/Dense"
#include "Eigen/Core"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <json.hpp>
#include <opencv2/opencv.hpp>


// ======================================================================================
// A binary tree contains nodes with degree = 0 or 2, satisfies:
// 2 * y = x + y + 1,  ( x, y denote the number of nodes with degree = 0, degree = 2 )
// then, y = x + 1.
// Meanwhile, x = ceil(triangles_n / bvh_thresh_n), 
// thus, bvh_nodes_n = x + y = 2 * x + 1.
// Consequently, bvh_nodes_n = 2 * ceil(triangles_n / bvh_thresh_n) + 1.
// If we set the bvh_nodes_n, it can be worked out that:
// bvh_thresh_n = triangles_n / [(bvh_nodes_n - 1) / 2 - 1]
// ======================================================================================


using json = nlohmann::json;
struct Task {
    int task_id;
    std::string task_type;
    std::vector<std::pair<std::string, std::string>> OBJ_paths;
    Eigen::Vector3f eye_pos;
    float fovY;
    float far;
    float near;
    int view_z_dir;
    unsigned int width;
    unsigned int height;
    unsigned int bvh_thresh_n;
    unsigned int light_sample_n;
    float P_RR;
    unsigned int spp;
    std::string image_save_path;
    Eigen::Vector3f rot_axis;
    float rot_angle;
    float rot_speed;
    std::string video_save_path;
    std::string image_save_dir;
    std::string transform_save_path;
    unsigned int fps;
};
std::vector<Task> tasks;
#ifdef _MSC_VER
std::string config_path("../../config.json");
#else
std::string config_path("../config.json");
#endif

bool config_tasks()
{
    std::ifstream json_file(config_path);
    std::stringstream json_data;
    json_data << json_file.rdbuf();
    json j = json::parse(json_data);

    for (const auto& task_json : j["tasks"])
    {
        Task task;
        task.task_id = task_json["task_id"];
        task.task_type = task_json["task_type"];

        for (const auto& path_json : task_json["OBJ_paths"])
        {
            task.OBJ_paths.push_back({path_json["OBJ_path"], path_json["MTL_dir"]});
        }
        task.view_z_dir = (int)task_json["view_z_dir"] > 0? 1:-1;
        task.eye_pos = Eigen::Vector3f(task_json["eye_pos"]["x"], task_json["eye_pos"]["y"], task_json["eye_pos"]["z"]);
        task.fovY = task_json["fovY"];
        task.width = task_json["width"];
        task.height = task_json["height"];
        task.bvh_thresh_n = task_json["bvh_thresh_n"];
        task.P_RR = task_json["P_RR"];
        task.spp = task_json["spp"];
        task.light_sample_n = task_json["light_sample_n"];
        
        if(task.task_type == "V")
        {
            task.rot_axis = Eigen::Vector3f(task_json["rot_axis"]["x"], task_json["rot_axis"]["y"], task_json["rot_axis"]["z"]);
            task.rot_speed = task_json["rot_speed"];
            task.rot_angle = task_json["rot_angle"];
            task.video_save_path = task_json["video_save_path"];
            task.fps = task_json["fps"];
        }
        else if(task.task_type == "I")
        {
            task.image_save_path = task_json["image_save_path"];
        }
        else if(task.task_type == "RD")
        {
            task.near = task_json["near"];
            task.far = task_json["far"];
            task.rot_axis = Eigen::Vector3f(task_json["rot_axis"]["x"], task_json["rot_axis"]["y"], task_json["rot_axis"]["z"]);
            task.rot_speed = task_json["rot_speed"];
            task.rot_angle = task_json["rot_angle"];
            task.image_save_dir = task_json["image_save_dir"];
            task.transform_save_path = task_json["transform_save_path"];
        }
        tasks.push_back(task);
    }
    return true;
}


bool config_CUDA()
{
    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount > 0)
    {
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaSetDevice(0);
        return true;
    }
    printf("Failed to find CUDA device\n");
    return false;
   
}

Eigen::Matrix4f get_translation_matrix(Eigen::Vector3f move) 
{
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 3) = move.x();
    transform(1, 3) = move.y();
    transform(2, 3) = move.z();
    return transform;
}

Eigen::Matrix4f get_rotation_matrix(Eigen::Vector3f axis, float angle)
{
    Eigen::AngleAxisf rot(angle * (float)M_PI / 180.0f, axis.normalized());
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rot.toRotationMatrix();
    return transform.matrix();
}

void render_image(Task task)
{
    OBJLoader loader;
    Scene scene(task.width, task.height);

    for(const auto& OBJ_path : task.OBJ_paths)
    {
        if(loader.read_OBJ(OBJ_path.first.c_str(), OBJ_path.second.c_str()))
        {
            std::cout << OBJ_path.first.c_str() << " loaded" << std::endl;
        }
        else
        {
            std::cout << OBJ_path.first.c_str() << " failed to load" << std::endl;
        }
        while(!loader.is_empty())
        {
            loader.load_object();
            if(loader.get_triangles().size()>0)
            {
                Object object(loader.get_triangles());
                scene.add_normal_obj(object);
            }
            if(loader.get_light_triangles().size()>0)
            {
                Object light(loader.get_light_triangles());
                scene.add_light_obj(light);
            }
        }
    }

    scene.set_BVH(task.bvh_thresh_n);
    printf("BVH built.\n");
    float fovY = task.fovY * (float)M_PI / 180;
    Render render(&scene, task.spp, task.P_RR, task.light_sample_n);
    render.run_image(task.eye_pos, task.view_z_dir, fovY);
    render.save_frame_buffer(task.image_save_path.c_str());
    render.free();
    scene.free();
}

void show_progress_bar(float progress, float total, int bar_width = 50)
{
    std::cout << "[";

    int completed = static_cast<int>(bar_width * progress / total);
    for (int i = 0; i < bar_width; ++i)
    {
        if (i < completed) {
            std::cout << "#";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << progress << " / " << total << " \r";
    std::cout.flush();
}

void render_video(Task task)
{
    OBJLoader loader;
    Scene scene(task.width, task.height);

    for(const auto& OBJ_path : task.OBJ_paths)
    {
        if(loader.read_OBJ(OBJ_path.first.c_str(), OBJ_path.second.c_str()))
        {
            std::cout << OBJ_path.first.c_str() << " loaded" << std::endl;
        }
        else
        {
            std::cout << OBJ_path.first.c_str() << " failed to load" << std::endl;
        }
        while(!loader.is_empty())
        {
            loader.load_object();
            if(loader.get_triangles().size()>0)
            {
                Object object(loader.get_triangles());
                scene.add_normal_obj(object);
            }
            if(loader.get_light_triangles().size()>0)
            {
                Object light(loader.get_light_triangles());
                scene.add_light_obj(light);
            }
        }
    }

    scene.set_BVH(task.bvh_thresh_n);
    printf("BVH built.\n");
    Render render(&scene, task.spp, task.P_RR, task.light_sample_n);

    cv::VideoWriter writer;

    writer.open(task.video_save_path, cv::VideoWriter::fourcc('M','J','P','G'), task.fps, cv::Size(task.width, task.height));
    
    float cur_rot_angle = 0.0f;
    float fovY = task.fovY * (float)M_PI / 180;
    while(cur_rot_angle < task.rot_angle)
    {
        Eigen::Matrix4f view = get_rotation_matrix(task.rot_axis, cur_rot_angle);
        render.run_video(task.eye_pos, task.view_z_dir, fovY, view);
        cv::Mat frame(task.height, task.width, CV_8UC3);
        memcpy(frame.data, render.get_frame_buffer(), sizeof(unsigned char) * 3 * task.width * task.height);
        writer.write(frame);
        cur_rot_angle += task.rot_speed;
        show_progress_bar(cur_rot_angle, task.rot_angle);
    }
    writer.release();
    render.free();
    scene.free();
}

struct Frame
{
    Eigen::Matrix4f transform;
    std::string file_path;
};

struct Transform
{
    float fovX;
    float near;
    float far;
    int view_z_dir;
    std::vector<Frame> frames;
};


void write_transform(Transform t, const std::string& save_path)
{
    json data;
    data["camera_angle_x"] = t.fovX;
    data["near"] = t.near;
    data["far"] = t.far;
    data["view_z_dir"] = t.view_z_dir;
    std::vector<json> frames_data;
    for(const auto & frame : t.frames)
    {
        json frame_data;
        frame_data["file_path"] = frame.file_path.c_str();
        frame_data["transform_matrix"] = {
            {frame.transform(0, 0), frame.transform(0, 1), frame.transform(0, 2), frame.transform(0, 3)},
            {frame.transform(1, 0), frame.transform(1, 1), frame.transform(1, 2), frame.transform(1, 3)},
            {frame.transform(2, 0), frame.transform(2, 1), frame.transform(2, 2), frame.transform(2, 3)},
            {frame.transform(3, 0), frame.transform(3, 1), frame.transform(3, 2), frame.transform(3, 3)}
        };
        frames_data.push_back(frame_data);
    }
    data["frames"] = frames_data;
    std::ofstream json_file(save_path.c_str());
    if (json_file.is_open())
    {
        json_file << data.dump(4);
        json_file.close();
    }
    else
    {
        std::cout << "Unable to open " << save_path.c_str() << std::endl;
    }
}

void render_rotation_data(Task task)
{
    OBJLoader loader;
    Scene scene(task.width, task.height);

    for(const auto& OBJ_path : task.OBJ_paths)
    {
        if(loader.read_OBJ(OBJ_path.first.c_str(), OBJ_path.second.c_str()))
        {
            std::cout << OBJ_path.first.c_str() << " loaded" << std::endl;
        }
        else
        {
            std::cout << OBJ_path.first.c_str() << " failed to load" << std::endl;
        }
        while(!loader.is_empty())
        {
            loader.load_object();
            if(loader.get_triangles().size()>0)
            {
                Object object(loader.get_triangles());
                scene.add_normal_obj(object);
            }
            if(loader.get_light_triangles().size()>0)
            {
                Object light(loader.get_light_triangles());
                scene.add_light_obj(light);
            }
        }
    }
    
    scene.set_BVH(task.bvh_thresh_n);
    printf("BVH built.\n");
    Render render(&scene, task.spp, task.P_RR, task.light_sample_n);

    int cnt = 0;
    float cur_rot_angle = 0.0f;
    Transform t;
    t.far = task.far;
    t.near = task.near;
    t.view_z_dir = task.view_z_dir;
    float fovY = task.fovY * (float)M_PI / 180;
    t.fovX = 2 * atan(tan(fovY / 2) * task.width / task.height);
    while(cur_rot_angle < task.rot_angle)
    {
        Eigen::Matrix4f view = get_rotation_matrix(task.rot_axis, cur_rot_angle);
        Eigen::Vector4f view_pos(task.eye_pos.x(), task.eye_pos.y(), task.eye_pos.z(), 1.0f);
        view_pos = view * view_pos;
        render.run_rotation_data(view_pos.head<3>(), task.view_z_dir, fovY, view);
        cur_rot_angle += task.rot_speed;
        show_progress_bar(cur_rot_angle, task.rot_angle);
        std::string dir = task.image_save_dir;
        Frame frame;
        frame.file_path = dir.append(std::to_string(cnt).append(".png"));
        view(0, 3) += view_pos.x();
        view(1, 3) += view_pos.y();
        view(2, 3) += view_pos.z();
        frame.transform = view;
        render.save_frame_buffer(frame.file_path.c_str());
        t.frames.push_back(frame);
        cnt++;
    }
    write_transform(t, task.transform_save_path);
    render.free();
    scene.free();
}

int main(void)
{
    if(!config_CUDA())
    {
        printf("Failed to configure CUDA\n");
        return -1;
    }
    if(!config_tasks())
    {
        return -1;
    }
    for(const auto& task:tasks)
    {
        if(task.task_type == "I")
        {
            render_image(task);
        }
        else if(task.task_type == "V")
        {
            render_video(task);
        }
        else if(task.task_type == "RD")
        {
            render_rotation_data(task);
        }
        else
        {
            return -1;
        }
    }
    return 0;
}