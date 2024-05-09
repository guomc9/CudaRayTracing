#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_
#include <algorithm>
#include "Global.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Material.h"

class Triangle
{
    private:
        Eigen::Vector3f v1, v2, v3;
        Eigen::Vector3f center;
        Eigen::Vector3f normal;
        Material material;
        float area;
        float area_of_obj;
        float max_x, min_x;
        float max_y, min_y;
        float max_z, min_z;

    public:
        Triangle(Eigen::Vector3f _v1, Eigen::Vector3f _v2, Eigen::Vector3f _v3, Eigen::Vector3f _normal, Material _m)
        :v1(_v1), v2(_v2), v3(_v3), material(_m)
        {
            center = (v1 + v2 + v3) / 3;
            normal = (v2 - v1).cross(v3 - v1).normalized();
            // normal = _normal;

            max_x = std::max(std::max(v1.x(), v2.x()), v3.x());
            min_x = std::min(std::min(v1.x(), v2.x()), v3.x());
            
            max_y = std::max(std::max(v1.y(), v2.y()), v3.y());
            min_y = std::min(std::min(v1.y(), v2.y()), v3.y());

            max_z = std::max(std::max(v1.z(), v2.z()), v3.z());
            min_z = std::min(std::min(v1.z(), v2.z()), v3.z());

            area = (v2 - v1).cross(v3 - v1).norm() * 0.5f;
            area_of_obj = 0.0f;
        }

        static bool cmp_x(Triangle &t1, Triangle &t2)
        {
            return t1.center.x() < t2.center.x();
        }

        static bool cmp_y(Triangle &t1, Triangle &t2)
        {
            return t1.center.y() < t2.center.y();
        }

        static bool cmp_z(Triangle &t1, Triangle &t2)
        {
            return t1.center.z() < t2.center.z();
        }

        inline Eigen::Vector3f get_v1() const
        {
            return v1;
        }

        inline Eigen::Vector3f get_v2() const
        {
            return v2;
        }

        inline Eigen::Vector3f get_v3() const
        {
            return v3;
        }

        inline Eigen::Vector3f get_normal() const
        {
            return normal;
        }

        inline Eigen::Vector3f get_center() const
        {
            return center;
        }

        inline float get_area() const
        {
            return area;
        }

        inline float get_area_of_obj() const
        {
            return area_of_obj;
        }

        inline void set_area_of_obj(const float _area_of_obj)
        {
            area_of_obj = _area_of_obj;
        }

        inline float get_max_x() const
        {
            return max_x;
        }

        inline float get_max_y() const
        {
            return max_y;
        }

        inline float get_max_z() const
        {
            return max_z;
        }

        inline float get_min_x() const
        {
            return min_x;
        }

        inline float get_min_y() const
        {
            return min_y;
        }

        inline float get_min_z() const
        {
            return min_z;
        }

        inline Material get_material() const
        {
            return material;
        }
};
#endif