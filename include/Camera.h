#ifndef _CAMERA_H_
#define _CAMERA_H_

class Camera {
    private:
        Eigen::Vector3f eye_pos;
        float fov_y;

    public:
        Camera(Eigen::Vector3f eye_pos, float fov_y) 
            : eye_pos(eye_pos), fov_y(fov_y) {}

        Eigen::Vector3f get_eye_pos() const
        {
            return eye_pos;
        }

        float get_fov_y() const
        {
            return fov_y;
        }
};

#endif