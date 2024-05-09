#ifndef _OBJLOADER_H_
#define _OBJLOADER_H_
#include "Global.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "stb_image.h"

class Shape
{
    public:
        std::string _material_id;
        std::vector<std::vector<uint64_t> > _vs;
        std::vector<std::vector<uint64_t> > _vns;
        std::vector<std::vector<uint64_t> > _vts;
        std::vector<float> _kd;
        std::vector<float> _ke;
        std::vector<float> _ks;
        bool _has_map_kd;
        float _ns;
        std::string _map_kd;
        Shape(const std::string& material_id)
        {
            _material_id = material_id;
            _has_map_kd = false;
            _kd.resize(3);
            _ke.resize(3);
            _ks.resize(3);
        };
        void add_face(const std::vector<uint64_t>& v, const std::vector<uint64_t>& vn, const std::vector<uint64_t>& vt)
        {
            _vs.push_back(v);
            _vns.push_back(vn);
            _vts.push_back(vt);
        };
};

class OBJLoader
{
    private:
        std::string _obj_path;
        std::string _mtl_dir;
        std::string _mtl_path;
        std::vector<std::string> _tex_path_list;
        std::vector<Eigen::Vector3f> _vertices;
        std::vector<Eigen::Vector3f> _normals;
        std::vector<Eigen::Vector2f> _textures;
        std::vector<Shape> _shapes;
        std::map<std::string, std::vector<uint64_t> > _mts;

    public:
        OBJLoader(const char* obj_path, const char* mtl_dir)
        {
            _obj_path = std::string(obj_path);
            _mtl_dir = std::string(mtl_dir);
        };

        void parse()
        {
            std::ifstream objFile(_obj_path);
            if (!objFile.is_open())
            {
                std::cerr << "Unable to open OBJ file: " << _obj_path << std::endl;
                return;
            }

            std::string line;
            while (std::getline(objFile, line))
            {
                std::istringstream lineStream(line);
                std::string prefix;
                lineStream >> prefix;
                if (prefix == "v")
                {
                    Eigen::Vector3f vertex;
                    lineStream >> vertex[0] >> vertex[1] >> vertex[2];
                    _vertices.push_back(vertex);
                }
                else if (prefix == "vn")
                {
                    Eigen::Vector3f normal;
                    lineStream >> normal[0] >> normal[1] >> normal[2];
                    _normals.push_back(normal);
                }
                else if (prefix == "vt")
                {
                    Eigen::Vector2f texCoord;
                    lineStream >> texCoord[0] >> texCoord[1];
                    _textures.push_back(texCoord);
                }
                else if (prefix == "f")
                {
                    std::vector<uint64_t> vertexIndices, uvIndices, normalIndices;
                    std::string vertexStr;
                    while (lineStream >> vertexStr)
                    {
                        std::stringstream vertexStream(vertexStr);
                        std::string indexStr;
                        uint64_t vIndex, vtIndex, vnIndex;
                        
                        std::getline(vertexStream, indexStr, '/');
                        vIndex = std::stoull(indexStr);
                        vertexIndices.push_back(vIndex > 0 ? vIndex - 1 : _vertices.size() + vIndex);

                        if (std::getline(vertexStream, indexStr, '/'))
                        {
                            vtIndex = std::stoull(indexStr);
                            uvIndices.push_back(vtIndex > 0 ? vtIndex - 1 : _textures.size() + vtIndex);
                        }

                        if (std::getline(vertexStream, indexStr, '/'))
                        {
                            vnIndex = std::stoull(indexStr);
                            normalIndices.push_back(vnIndex > 0 ? vnIndex - 1 : _normals.size() + vnIndex);
                        }
                    }
                    if (_shapes.size() > 0)
                    {
                        _shapes.back().add_face(vertexIndices, normalIndices, uvIndices);
                    }
                }
                else if (prefix == "mtllib")
                {
                    std::string _mtl_name;
                    lineStream >> _mtl_name;
                    _mtl_path = _mtl_dir + "/" + _mtl_name;
                }
                else if (prefix == "usemtl")
                {
                    std::string _mtl_id;
                    lineStream >> _mtl_id;
                    _mts[_mtl_id].push_back(_shapes.size());
                    _shapes.push_back(Shape(_mtl_id));
                }
            }
            objFile.close();

            std::ifstream mtlFile(_mtl_path);
            if (!mtlFile.is_open())
            {
                std::cerr << "Unable to open MTL file: " << _mtl_path << std::endl;
                return;
            }

            std::string _mtl_id;
            std::vector<uint64_t> _s_ids;
            std::vector<float> ke(3);
            std::vector<float> kd(3);
            std::vector<float> ks(3);
            std::string map_Kd;
            while (getline(mtlFile, line))
            {
                std::istringstream lineStream(line);
                std::string prefix;
                lineStream >> prefix;
                if (prefix == "newmtl")
                {
                    lineStream >> _mtl_id;
                    _s_ids = _mts[_mtl_id];
                }
                else if (prefix == "Kd")
                {
                    lineStream >> kd[0] >> kd[1] >> kd[2];
                    for (auto _s_id : _s_ids)
                        _shapes[_s_id]._kd = kd;
                }
                else if (prefix == "Ks")
                {
                    std::vector<float> ks(3);
                    lineStream >> ks[0] >> ks[1] >> ks[2];
                    for (auto _s_id : _s_ids)
                        _shapes[_s_id]._ks = ks;
                }
                else if (prefix == "Ke")
                {
                    std::vector<float> ke(3);
                    lineStream >> ke[0] >> ke[1] >> ke[2];
                    for (auto _s_id : _s_ids)
                        _shapes[_s_id]._ke = ke;
                }
                else if (prefix == "map_Kd")
                {
                    std::string map_kd;
                    lineStream >> map_kd;
                    for (auto _s_id : _s_ids)
                    {
                        _shapes[_s_id]._has_map_kd = true;
                        _shapes[_s_id]._map_kd = _mtl_dir + "/" + map_kd;
                    }
                }
                else if (prefix == "Ns")
                {
                    float ns;
                    lineStream >> ns;
                    for (auto _s_id : _s_ids)
                        _shapes[_s_id]._ns = ns;
                }
            }
            mtlFile.close();
        };

        std::vector<Shape> get_shapes() const
        {
            return _shapes;
        }

        std::vector<Eigen::Vector3f> get_vertices() const
        {
            return _vertices;
        }

        std::vector<Eigen::Vector3f> get_normals() const
        {
            return _normals;
        }

        std::vector<Eigen::Vector2f> get_textures() const
        {
            return _textures;
        }

        uint64_t size() const
        {
            return _shapes.size();
        }
};
#endif