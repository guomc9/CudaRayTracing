#ifndef _RAY_H_
#define _RAY_H_
#include "Eigen/Dense"

class Ray
{
    private:
        Eigen::Vector3f origin;
        Eigen::Vector3f dir;
        Eigen::Vector3f inv_dir;
    public:
        __device__ Ray(Eigen::Vector3f o, Eigen::Vector3f d):origin(o){
            dir = d.normalized();
            inv_dir = Eigen::Vector3f(1 / dir.x(), 1 / dir.y(), 1 / dir.z());
        };
        
        __device__ Eigen::Vector3f get_origin() const
        {
            return origin;
        }

        __device__ Eigen::Vector3f get_dir() const
        {
            return dir;
        }

        __device__ Eigen::Vector3f get_inv_dir() const
        {
            return inv_dir;
        }
};
#endif