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
#include <thread>
#include <json.hpp>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


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
};
Task task;
#ifdef _MSC_VER
std::string config_path("../../config.json");
#else
std::string config_path("../config.json");
#endif

bool config_task()
{
    std::ifstream json_file(config_path);
    std::stringstream json_data;
    json_data << json_file.rdbuf();
    json task_json = json::parse(json_data);

    for (const auto& path_json : task_json["OBJ_paths"])
    {
        task.OBJ_paths.push_back({path_json["OBJ_path"], path_json["MTL_dir"]});
    }
    task.view_z_dir = (int)task_json["view_z_dir"] > 0 ? 1 : -1;
    task.eye_pos = Eigen::Vector3f(task_json["eye_pos"]["x"], task_json["eye_pos"]["y"], task_json["eye_pos"]["z"]);
    task.fovY = task_json["fovY"];
    task.width = task_json["width"];
    task.height = task_json["height"];
    task.bvh_thresh_n = task_json["bvh_thresh_n"];
    task.P_RR = task_json["P_RR"];
    task.spp = task_json["spp"];
    task.light_sample_n = task_json["light_sample_n"];

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

void render_view()
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
            if(loader.get_triangles().size() > 0)
            {
                Object object(loader.get_triangles());
                scene.add_normal_obj(object);
            }
            if(loader.get_light_triangles().size() > 0)
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

    GLFWwindow* window = glfwCreateWindow(800, 600, "CudaRayTracing", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return ;
    }
    glfwMakeContextCurrent(window);
    
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return ;
    }

    glViewport(0, 300, 800, 600);

    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, task.width * task.height * 3, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

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
    int spp = task.spp;
    float P_RR = task.P_RR;
    int light_sample_n = task.light_sample_n;
    int fps_limit = 30;

    Render render(&scene, spp, P_RR, light_sample_n);
    glUseProgram(prog_id);
    auto begin = std::chrono::high_resolution_clock::now();
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        render.run_view(task.eye_pos, task.view_z_dir, fovY, cuda_pbo_resource);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - begin;
        double frame_time = 1.0 / fps_limit;
        if (elapsed.count() < frame_time)
        {
            auto sleep_time = std::chrono::duration<double>(frame_time - elapsed.count());
            std::this_thread::sleep_for(sleep_time);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - begin;
        }
        double fps = 1. / elapsed.count();
        printf("fps: %lf frames / second\n", 1. / elapsed.count());

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
        

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("CudaRayTracing");
        ImGui::Text("Frames per Second(fps): %.3f", fps);

        if (ImGui::SliderInt("Samples per Pixel", &spp, 1, 32))
        {
            std::cout << "Update spp value: " << spp << std::endl;
            render.set_spp(spp);
        }

        if (ImGui::SliderFloat("Russian Roulette Probability (P_RR)", &P_RR, 0.0f, 1.0f))
        {
            std::cout << "Update P_RR value: " << P_RR << std::endl;
            render.set_P_RR(P_RR);
        }

        if (ImGui::SliderInt("Light Samples (light_sample_n)", &light_sample_n, 1, 64))
        {
            std::cout << "Update light_sample_n value: " << light_sample_n << std::endl;
            render.set_light_sample_n(light_sample_n);
        }

        if (ImGui::SliderInt("Frames per Second Limit", &fps_limit, 5, 60))
        {
            std::cout << "Update frames per second limit: " << fps_limit << std::endl;
        }
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    
    render.free();
    scene.free();
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &textureID);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}


int main(void)
{
    if(!config_CUDA())
    {
        printf("Failed to configure CUDA\n");
        return -1;
    }
    if(!config_task())
    {
        return -1;
    }
    
    render_view();

    return 0;
}