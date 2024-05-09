#ifndef __GUI_H__
#define __GUI_H__
#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
class Gui
{
    private:
        int _height;
        int _width;
        float _eye_pos[3];
        float _lookat[3];
        float _up[3];
        float _P_RR;
        int _spp;
        int _light_sample_n;
    
    public:
        Gui(int height, int width, float* eye_pos, float* lookat, float* up, float P_RR, int spp, int light_sample_n)
        :_height(height), _width(width), _P_RR(P_RR), _spp(spp), _light_sample_n(light_sample_n)
        {
            memcpy(_eye_pos, eye_pos, 3 * sizeof(float));
            memcpy(_lookat, lookat, 3 * sizeof(float));
            memcpy(_up, up, 3 * sizeof(float));
        };

        int* height()
        {
            return &_height;
        }

        int* width()
        {
            return &_width;
        }

        float* P_RR()
        {
            return &_P_RR;
        }

        int* spp()
        {
            return &_spp;
        }

        int* light_sample_n()
        {
            return &_light_sample_n;
        }

        float* eye_pos()
        {
            return _eye_pos;
        }

        float* lookat()
        {
            return _lookat;
        }

        float* up()
        {
            return _up;
        }

        inline Eigen::Vector3f get_eye_pos_vec() const
        {
            return Eigen::Vector3f(_eye_pos[0], _eye_pos[1], _eye_pos[2]);
        }

        inline Eigen::Vector3f get_lookat_vec() const
        {
            return Eigen::Vector3f(_lookat[0], _lookat[1], _lookat[2]);
        }

        inline Eigen::Vector3f get_up_vec() const
        {
            return Eigen::Vector3f(_up[0], _up[1], _up[2]);
        }

        inline int get_height() const
        {
            return _height;
        }

        inline int get_width() const
        {
            return _width;
        }

        inline float get_P_RR() const
        {
            return _P_RR;
        };

        inline int get_spp() const
        {
            return _spp;
        }

        inline int get_light_sample_n() const
        {
            return _light_sample_n;
        }

};
#endif