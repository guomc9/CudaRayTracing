#ifndef _CAMERA_H_
#define _CAMERA_H_
#define  _USE_MATH_DEFINES
#include "Eigen/Dense"
#include <math.h>
#include <algorithm>


Eigen::Matrix3f get_inverse_view_matrix(Eigen::Vector3f eye_pos, Eigen::Vector3f lookat, Eigen::Vector3f up)
{
    // Step 1: Create the forward vector from eye_pos to lookat
    Eigen::Vector3f f = (lookat - eye_pos).normalized();

    // Step 2: Create the right vector
    Eigen::Vector3f r = up.cross(f).normalized();

    // Step 3: Define the up vector (re-calculate to ensure orthogonality)
    Eigen::Vector3f u = f.cross(r).normalized();

    // Step 4: Construct the inverse view matrix
    Eigen::Matrix3f inverse_view = Eigen::Matrix3f::Identity();

    // Set the rotation part, remember that looking direction is negative Z
    // In the inverse view matrix, the rotation part is the transpose of the original
    inverse_view(0, 0) = r.x();
    inverse_view(1, 0) = r.y();
    inverse_view(2, 0) = r.z();
    inverse_view(0, 1) = u.x();
    inverse_view(1, 1) = u.y();
    inverse_view(2, 1) = u.z();
    inverse_view(0, 2) = f.x();
    inverse_view(1, 2) = f.y();
    inverse_view(2, 2) = f.z();
    
    return inverse_view;
}

#endif