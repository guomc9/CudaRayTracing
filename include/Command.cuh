#ifndef _COMMAND_CUH_
#define _COMMAND_CUH_

#include "concurrent_queue.h"
#include "Render.cuh"
#include "Resource.cuh"
#include "Scene.h"

struct RenderCommand 
{
    RTRenderImpl* render;
    Eigen::Vector3f eye_pos;
    Eigen::Matrix3f inv_view_mat;
    float fovY;
};

struct RenderResult 
{
    unsigned int img_res;
    double time_cost;
};

class AsyncQueueManager 
{
public:
    AsyncQueueManager();

    ~AsyncQueueManager()
    {
        if (cmd_q != nullptr) delete cmd_q;
        if (rslt_q != nullptr) delete rslt_q;
    }

    bool submitRenderCommand(const RenderCommand cmd);

    bool receiveRenderCommand(RenderCommand &cmd);

    bool submitRenderResult(const RenderResult rslt);

    bool receiveRenderResult(RenderResult &rslt);

private:
    concurrency::concurrent_queue<RenderCommand>* cmd_q;
    concurrency::concurrent_queue<RenderResult>* rslt_q;
};
#endif