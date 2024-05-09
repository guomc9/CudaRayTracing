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
        Object(std::vector<Triangle> ts)
        {
            triangles = ts;
            float area_of_obj = 0.0f;
            for (uint64_t i = 0; i < triangles.size(); i++)
            {
                area_of_obj += triangles[i].get_area();
            }            
            for (uint64_t i = 0; i < triangles.size(); i++)
            {
                triangles[i].set_area_of_obj(area_of_obj);
            }
            // printf("OBJ ke: (%f, ...)\n", triangles[0].get_material().get_ke().x());
            printf("OBJ area: %f\n", area_of_obj);
        };
        std::vector<Triangle>& get_triangles()
        {
            return triangles;
        }
};

#endif