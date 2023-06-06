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

// const unsigned int width = 784;
// const unsigned int height = 784;
// unsigned int width = 1024;
// unsigned int height = 512;
// unsigned int spp = 128;
// float P_RR = 0.99f;
// unsigned int light_sample_n = 32;
// ======================================================================================
// A binary tree contains nodes with degree = 0 or 2, satisfies:
// 2 * y = x + y + 1,  ( x, y denote the number of nodes with degree = 0, degree = 2 )
// then, y = x + 1.
// Meanwhile, x = ceil(triangles_n / thresh_n), 
// thus, nodes_n = x + y = 2 * x + 1.
// Consequently, nodes_n = 2 * ceil(triangles_n / thresh_n) + 1.
// If we set the nodes_n, it can be worked out that:
// thresh_n = triangles_n / [(nodes_n - 1) / 2 - 1]
// ======================================================================================

// const unsigned int bvh_thresh_n = 16;
// // Eigen::Vector3f eye_pos = Eigen::Vector3f(278, 273, -1000);
// Eigen::Vector3f eye_pos = Eigen::Vector3f(0, 50, -350);
// float fovY = 45.0f;
// Eigen::Vector3f rot_axis(0, 1, 0);
// float rot_angle = 0.003f;
// float rot_speed = 0.0001f;
// Eigen::Vector3f move(278, 0, 1270);

// std::vector<std::string> objs_path = {
//     "../../models/CornellBox/floor.obj",
//     "../../models/CornellBox/left.obj",
//     "../../models/CornellBox/right.obj",
//     "../../models/CornellBox/shortbox.obj",
//     "../../models/CornellBox/tallbox.obj"
// };
// std::string mtl_dir = "../../models/CornellBox/";

// std::vector<std::string> lights_path = {
//     "../../models/CornellBox/light.obj"
// };

// std::string image_save_path = "../../models/CornellBox/scene.png";
// std::string video_save_path = "../../models/CornellBox/Render/scene01.avi";

// // Scene-01
// std::vector<std::string> objs_path = {
//     "../../models/Scene-01/scene.obj"
// };

// std::vector<std::string> lights_path = {
// };

// std::string mtl_dir = "../../models/Scene-01/";
// std::string image_save_path = "../../models/Scene-01/scene.png";

// // Scene-02
// std::vector<std::string> objs_path = {
//     "../../models/Scene-02/scene.obj"
// };

// std::vector<std::string> lights_path = {
// };

// std::string mtl_dir = "../../models/Scene-02/";
// std::string image_save_path = "../../models/Scene-02/scene.png";

// // Scene-03
// std::vector<std::string> objs_path = {
//     "../../models/Scene-03/scene.obj"
// };

// std::vector<std::string> lights_path = {
// };

// std::string mtl_dir = "../../models/Scene-03/";
// std::string image_save_path = "../../models/Scene-03/scene.png";

using json = nlohmann::json;
struct Task {
    int task_id;
    std::string task_type;
    std::vector<std::pair<std::string, std::string>> OBJ_paths;
    Eigen::Vector3f eye_pos;
    float fovY;
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
    unsigned int fps;
};
std::vector<Task> tasks;
const char config_path[] = "../../config.json";

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
        else
        {
            return false;
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
    Render render(&scene, task.spp, task.P_RR, task.light_sample_n);
    render.run_image(task.eye_pos, task.fovY);
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
    while(cur_rot_angle < task.rot_angle)
    {
        Eigen::Matrix4f view = get_rotation_matrix(task.rot_axis, cur_rot_angle);
        render.run_video(task.eye_pos, task.fovY, view);
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
    }
    // // create video writer
    // cv::VideoWriter writer;

    // writer.open(video_save_path, cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(width, height));
    
    // // create Render
    // Render render(&scene, spp, P_RR);
    // float cur_rot_angle = 0.0f;
    // Eigen::Matrix4f move_back = get_translation_matrix(-move);
    // Eigen::Matrix4f move_forward = get_translation_matrix(move);
    // // int cnt = 0;
    // while(cur_rot_angle < rot_angle)
    // {
    //     Eigen::Matrix4f rot = get_rotation_matrix(rot_axis, cur_rot_angle);
    //     Eigen::Matrix4f view = move_forward * rot * move_back;
    //     render.run(eye_pos, fovY, view);
    //     cur_rot_angle += rot_speed;
    //     cv::Mat frame(height, width, CV_8UC3);
    //     memcpy(frame.data, render.get_frame_buffer(), sizeof(unsigned char) * 3 * width * height);
    //     writer.write(frame);
    // }
    // // free
    // writer.release();

    // ======================================================

    // ========================================================
    return 0;
}