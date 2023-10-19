#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Loader.h"
#include "Object.h"
#include "Scene.h"
#include <glad/glad.h>
#include "Render.cuh"
#include "Eigen/Dense"
#include "Eigen/Core"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <GLFW/glfw3.h>
#include <chrono>
#include <json.hpp>
#include <cuda_gl_interop.h>

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
    Eigen::Vector3f rot_axis_coords;
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
        task.view_z_dir = (int)task_json["view_z_dir"] > 0? 1 : -1;
        task.eye_pos = Eigen::Vector3f(task_json["eye_pos"]["x"], task_json["eye_pos"]["y"], task_json["eye_pos"]["z"]);
        task.fovY = task_json["fovY"];
        task.width = task_json["width"];
        task.height = task_json["height"];
        task.bvh_thresh_n = task_json["bvh_thresh_n"];
        task.P_RR = task_json["P_RR"];
        task.spp = task_json["spp"];
        task.light_sample_n = task_json["light_sample_n"];
        
        if(task.task_type == "I")
        {
            task.image_save_path = task_json["image_save_path"];
        }
        else if(task.task_type == "RD")
        {
            task.near = task_json["near"];
            task.far = task_json["far"];
            if(task_json["rot_axis"]["axis"] == "y")
            {
                task.rot_axis = Eigen::Vector3f(0.f, 1.f, 0.f);
                task.rot_axis_coords = Eigen::Vector3f(task_json["rot_axis"]["coord_1"], 0.f, task_json["rot_axis"]["coord_2"]);
            }
            else if(task_json["rot_axis"]["axis"] == "x")
            {
                task.rot_axis = Eigen::Vector3f(1.f, 0.f, 0.f);
                task.rot_axis_coords = Eigen::Vector3f(0.f, task_json["rot_axis"]["coord_1"], task_json["rot_axis"]["coord_2"]);
            }
            else if(task_json["rot_axis"]["axis"] == "z")
            {
                task.rot_axis = Eigen::Vector3f(0.f, 0.f, 1.f);
                task.rot_axis_coords = Eigen::Vector3f(task_json["rot_axis"]["coord_1"], task_json["rot_axis"]["coord_2"], 0.f);
            }
            else
            {
                return false;
            }
            task.rot_speed = task_json["rot_speed"];
            task.rot_angle = task_json["rot_angle"];
            task.image_save_dir = task_json["image_save_dir"];
            task.transform_save_path = task_json["transform_save_path"];
        }
        else if(task.task_type == "V")
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
    Eigen::Matrix4f forward = get_translation_matrix(-task.rot_axis_coords);
    Eigen::Matrix4f back = get_translation_matrix(task.rot_axis_coords);
    while(cur_rot_angle < task.rot_angle)
    {
        Eigen::Matrix4f view = get_rotation_matrix(task.rot_axis, cur_rot_angle);
        Eigen::Vector4f view_pos(task.eye_pos.x(), task.eye_pos.y(), task.eye_pos.z(), 1.0f);
        view_pos = back * view * forward * view_pos;
        render.run_rotation_data(view_pos.head<3>(), task.view_z_dir, fovY, view);
        cur_rot_angle += task.rot_speed;
        show_progress_bar(cur_rot_angle, task.rot_angle);
        std::string dir = task.image_save_dir;
        Frame frame;
        frame.file_path = dir.append(std::to_string(cnt).append(".png"));
        view(0, 3) = view_pos.x();
        view(1, 3) = view_pos.y();
        view(2, 3) = view_pos.z();
        frame.transform = view;
        render.save_frame_buffer(frame.file_path.c_str());
        t.frames.push_back(frame);
        cnt++;
    }
    write_transform(t, task.transform_save_path);
    render.free();
    scene.free();
}


void render_view(Task task)
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

    // configure opengl
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Triangle", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return ;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return ;
    }

    glViewport(0, 0, 800, 600);

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, task.width * task.height * 3, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    // 设置纹理参数
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 分配纹理内存
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, task.width, task.height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);


    float vertices[] = {
        // positions        // texture coords
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right

        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom left
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f  // top left
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    const char *vertexShaderSource = R"glsl(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main()
        {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
        })glsl";

    const char *fragmentShaderSource = R"glsl(
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoord;

        uniform sampler2D texture1;

        void main()
        {
            FragColor = texture(texture1, TexCoord);
        })glsl";

    GLint vs_id = glCreateShader(GL_VERTEX_SHADER);
    GLint fs_id = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vs_id, 1, &vertexShaderSource, nullptr);
    glShaderSource(fs_id, 1, &fragmentShaderSource, nullptr);
    
    GLint prog_id = glCreateProgram();
    
    glCompileShader(vs_id);
    glCompileShader(fs_id);
    
    glAttachShader(prog_id, vs_id);
    glAttachShader(prog_id, fs_id);
    
    glLinkProgram(prog_id);
    glValidateProgram(prog_id);

    glDeleteShader(vs_id);
    glDeleteShader(fs_id);

    glUseProgram(prog_id);

    cudaGraphicsResource* cuda_pbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

    scene.set_BVH(task.bvh_thresh_n);
    printf("BVH built.\n");
    float fovY = task.fovY * (float)M_PI / 180;
    Render render(&scene, task.spp, task.P_RR, task.light_sample_n);
    glUseProgram(prog_id);
    auto begin = std::chrono::high_resolution_clock::now();
    while(!glfwWindowShouldClose(window))
    {   
        render.run_view(task.eye_pos, task.view_z_dir, fovY, cuda_pbo_resource);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - begin;
        printf("fps: %lf frames / s\n", 1. / elapsed.count());
        begin = end;
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, task.width, task.height, GL_RGB, GL_UNSIGNED_BYTE, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        
        // Bind the PBO and texture
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

        // Upload the texture data
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, task.width, task.height, GL_RGB, GL_UNSIGNED_BYTE, 0);

        // Unbind the PBO and texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        glBindVertexArray(VAO);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    render.free();
    scene.free();
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &textureID);

    glfwTerminate();
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
        else if(task.task_type == "RD")
        {
            render_rotation_data(task);
        }
        else if(task.task_type == "V")
        {
            render_view(task);
        }
        else
        {
            return -1;
        }
    }
    return 0;
}