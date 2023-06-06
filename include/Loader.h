#ifndef _LOADER_H_
#define _LOADER_H_
#include "Eigen/Dense"
#include "tiny_obj_loader.h"
#include "Triangle.h"
#include "Material.h"
#include <vector>
#include <string>
#include <iostream>

class OBJLoader
{
    private:
        std::vector<Triangle> triangles;
        std::vector<Triangle> light_triangles;
        Eigen::Vector3f ka;
        Eigen::Vector3f kd;
        Eigen::Vector3f ks;
        Eigen::Vector3f ke;
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

    public:
        OBJLoader()
        {
            ka = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            kd = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ks = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ke = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            shapes.clear();
            materials.clear();
        }

        bool read_OBJ(const char* obj_path, const char* mtl_dir)
        {
            std::string err;
            std::string warn;
            shapes.clear();
            materials.clear();
            bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_path, mtl_dir);
            if (!err.empty()) {
                std::cerr << "Error loading OBJ file: " << err << std::endl;
                return 0;
            }
            if (!warn.empty()) {
                std::cerr << "Warn loading OBJ file: " << warn << std::endl;
                return 0;
            }
            if (!success) {
                std::cerr << "Failed to load OBJ file: " << obj_path << std::endl;
                return 0;
            }
            return true;
        }

        bool is_empty()
        {
            return shapes.empty();
        }

        void load_object()
        {
            ka = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            kd = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ks = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
            ke = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
            triangles.clear();
            light_triangles.clear();
            const auto shape = shapes.back();
            const std::vector<tinyobj::index_t>& indices = shape.mesh.indices;
            for (size_t i = 0; i < indices.size(); i += 3)
            {
                unsigned int idx1 = indices[i].vertex_index;
                unsigned int idx2 = indices[i + 1].vertex_index;
                unsigned int idx3 = indices[i + 2].vertex_index;    
                float x1 = attrib.vertices[3 * idx1];
                float y1 = attrib.vertices[3 * idx1 + 1];
                float z1 = attrib.vertices[3 * idx1 + 2];
                Eigen::Vector3f v1 = Eigen::Vector3f(x1, y1, z1);

                float x2 = attrib.vertices[3 * idx2];
                float y2 = attrib.vertices[3 * idx2 + 1];
                float z2 = attrib.vertices[3 * idx2 + 2];
                Eigen::Vector3f v2 = Eigen::Vector3f(x2, y2, z2);

                float x3 = attrib.vertices[3 * idx3];
                float y3 = attrib.vertices[3 * idx3 + 1];
                float z3 = attrib.vertices[3 * idx3 + 2];
                Eigen::Vector3f v3 = Eigen::Vector3f(x3, y3, z3);
                int material_id = shape.mesh.material_ids[i / 3];
                if (material_id >= 0 && material_id < materials.size())
                {
                    const tinyobj::material_t& material = materials[material_id];
                    ka = Eigen::Vector3f(material.ambient);
                    kd = Eigen::Vector3f(material.diffuse);
                    ks = Eigen::Vector3f(material.specular);
                    ke = Eigen::Vector3f(material.emission);
                }
                Material m(kd, ks, ka, ke);
                Triangle t(v1, v2, v3, m);
                if(m.has_emission())
                    light_triangles.push_back(t);
                else
                    triangles.push_back(t);
            }
            shapes.pop_back();
        }

        std::vector<Triangle>& get_triangles()
        {
            return triangles;
        }

        std::vector<Triangle>& get_light_triangles()
        {
            return light_triangles;
        }

};

#endif