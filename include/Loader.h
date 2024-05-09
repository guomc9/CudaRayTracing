#ifndef _LOADER_H_
#define _LOADER_H_
#define STB_IMAGE_IMPLEMENTATION
#include "Global.h"
#include "Eigen/Dense"
#include "OBJLoader.h"
#include "Triangle.h"
#include "Material.h"
#include <vector>
#include <string>
#include <iostream>

class Loader
{
    private:
        std::vector<Shape> shapes;
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3f> normals;
        std::vector<Eigen::Vector2f> textures;

    public:
        Loader()
        {
            vertices.clear();
            normals.clear();
            textures.clear();
            shapes.clear();
        }

        void read_OBJ(const char* obj_path, const char* mtl_dir)
        {
            OBJLoader obj_loader(obj_path, mtl_dir);
            obj_loader.parse();
            shapes = obj_loader.get_shapes();
            vertices = obj_loader.get_vertices();
            normals = obj_loader.get_normals();
            textures = obj_loader.get_textures();
        }

        void load_object(uint64_t index, std::vector<Triangle>& triangles, std::vector<Triangle>& light_triangles)
        {
            int height, width, channel;
            float area_of_obj = 0.0f;
            uint8_t* kd_tex = nullptr;
            auto ka = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            auto kd = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            auto ks = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            auto ke = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            float ns = 1;
            std::string map_kd;
            triangles.clear();
            light_triangles.clear();
            const Shape s = shapes[index];
            
            if (s._has_map_kd)
            {
                map_kd = s._map_kd;
                kd_tex = stbi_load(map_kd.c_str(), &height, &width, &channel, 0);        
            }
            for (uint64_t i = 0; i < s._vs.size(); i++)
            {
                uint64_t idx1 = s._vs[i][0];
                uint64_t idx2 = s._vs[i][1];
                uint64_t idx3 = s._vs[i][2];
                
                Eigen::Vector3f v1 = vertices[idx1];
                Eigen::Vector3f v2 = vertices[idx2];
                Eigen::Vector3f v3 = vertices[idx3];
                
                Eigen::Vector3f vn1 = normals[idx1];
                Eigen::Vector3f vn2 = normals[idx2];
                Eigen::Vector3f vn3 = normals[idx3];
                
                auto normal = ((vn1 + vn2 + vn3) / 3.0f).normalized();
                kd = Eigen::Vector3f(s._kd[0], s._kd[1], s._kd[2]);
                ke = Eigen::Vector3f(s._ke[0], s._ke[1], s._ke[2]);
                ns = s._ns;
                if (kd_tex != nullptr)
                {
                    float intpart;
                    Eigen::Vector2f vt1 = textures[idx1];
                    Eigen::Vector2f vt2 = textures[idx2];
                    Eigen::Vector2f vt3 = textures[idx3];
                    
                    // std::cout << "vt1: " << vt1 << std::endl;
                    int u1 = static_cast<int>(std::modff(std::modff(vt1[0], &intpart) + 1, &intpart) * (width-1));
                    int v1 = static_cast<int>(std::modff(std::modff(vt1[1], &intpart) + 1, &intpart) * (height-1));
                    int offset = (v1 * width + u1) * channel;
                    auto kd_1 = Eigen::Vector3f(kd_tex[offset], kd_tex[offset+1], kd_tex[offset+2]) / 255.;
                    
                    // std::cout << "vt2: " << vt2 << std::endl;
                    int u2 = static_cast<int>(std::modff(std::modff(vt2[0], &intpart) + 1, &intpart) * (width-1));
                    int v2 = static_cast<int>(std::modff(std::modff(vt2[1], &intpart) + 1, &intpart) * (height-1));
                    offset = (v2 * width + u2) * channel;
                    auto kd_2 = Eigen::Vector3f(kd_tex[offset], kd_tex[offset+1], kd_tex[offset+2]) / 255.;
                    
                    // std::cout << "vt3: " << vt3 << std::endl;
                    int u3 = static_cast<int>(std::modff(std::modff(vt3[0], &intpart) + 1, &intpart) * (width-1));
                    int v3 = static_cast<int>(std::modff(std::modff(vt3[1], &intpart) + 1, &intpart) * (height-1));
                    offset = (v3 * width + u3) * channel;
                    auto kd_3 = Eigen::Vector3f(kd_tex[offset], kd_tex[offset+1], kd_tex[offset+2]) / 255.;
                    
                    kd = (kd_1 + kd_2 + kd_3) / 3;
                    
                }
                
                Material m(kd, ks, ka, ke, ns, ns > 1 ? SPECULAR : DIFFUSE);
                if (v1.y() < 0.1f && v2.y() < 0.1f && v3.y() < 0.1f)
                {
                    normal = Eigen::Vector3f(0, 1, 0);
                }
                Triangle t(v1, v2, v3, normal, m);
                // if (i == s._vs.size() - 1 && s._vs.size() == 2)
                // {
                //     printf("vertex: (%f, %f, %f)\n", v1.x(), v1.y(), v1.z());
                //     printf("normal: (%f, %f, %f)\n", normal.x(), normal.y(), normal.z());
                //     printf("kd: (%f, %f, %f)\n", kd.x(), kd.y(), kd.z());
                // }
                if (m.has_emission())
                    light_triangles.push_back(t);
                else
                    triangles.push_back(t);
            }
        }

        uint64_t size() const
        {
            return shapes.size();
        }

};

#endif