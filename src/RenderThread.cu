#include "RenderThread.cuh"

#include <chrono>

void renderFunc(AsyncQueueManager *q_manager, ImagePool *i_pool, std::mutex* release_req, std::mutex* release_rsp)
{
    release_rsp->lock();
    while (release_req->try_lock())
    {
        RenderCommand cmd;
        if (q_manager->receiveRenderCommand(cmd))
        {
            int img_res = i_pool->allocImage();
            auto begin = std::chrono::high_resolution_clock::now();
            cmd.render->run_view(cmd.eye_pos, cmd.inv_view_mat, cmd.fovY, i_pool->get(img_res)->getDeviceFrameBuffer());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - begin;

            RenderResult rslt;
            rslt.img_res = img_res;
            rslt.time_cost = elapsed.count();
            q_manager->submitRenderResult(rslt);
        }
        release_req->unlock();
    }
    printf("try unlock release response\n");
    release_rsp->unlock();
    printf("Render thread return.\n");
    return;
}
