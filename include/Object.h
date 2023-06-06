#ifndef _OBJECT_H_
#define _OBJECT_H_
#include "Triangle.h"
#include "Material.h"
#include <vector>

class Object
{
    private:
        std::vector<Triangle> triangles;
    public:
        Object(std::vector<Triangle> ts):triangles(ts){};
        std::vector<Triangle>& get_triangles()
        {
            return triangles;
        }
};

#endif