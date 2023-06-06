#ifndef _SCENE_CUH_
#define _SCENE_CUH_

#include "Triangle.h"
#include "BVH.h"
#include "Object.h"
#include "Global.h"
#define __STDC_LIB_EXT1__
#include "stb_image_write.h"
#define  _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <iostream>


class Scene
{
    private:
        BVH* bvh;
        unsigned int width;
        unsigned int height;
        unsigned int pixels;
        std::vector<Object> normal_objs;
        std::vector<Object> light_objs;
        std::vector<Triangle> triangles;

    public:
        Scene(unsigned int _width, unsigned int _height)
        :width(_width), height(_height), pixels(_width * _height){};
        
    private:
        void add_triangles(std::vector<Triangle>& ts)
        {
            triangles.insert(triangles.end(), ts.begin(), ts.end());
        }

    public:
        void add_light_obj(Object& obj)
        {
            add_triangles(obj.get_triangles());
            light_objs.push_back(obj);
        }

        void add_normal_obj(Object& obj)
        {
            add_triangles(obj.get_triangles());
            normal_objs.push_back(obj);
        }

        void set_BVH(unsigned int thresh_n)
        {
            bvh = new BVH(thresh_n, triangles);
        }

        void free()
        {
            delete bvh;
        }

        unsigned int get_height()
        {
            return height;
        }

        unsigned int get_width()
        {
            return width;
        }

        unsigned int get_pixels()
        {
            return pixels;
        }

        BVH& get_bvh()
        {
            return *bvh;
        }

        std::vector<Object>& get_light_objs()
        {
            return light_objs;
        }

        std::vector<Triangle>& get_triangles()
        {
            return triangles;
        }
};

#endif