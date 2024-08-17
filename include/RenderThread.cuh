#ifndef _RENDER_THREAD_CUH_
#define _RENDER_THREAD_CUH_

#include "Command.cuh"
#include "Resource.cuh"

#include <mutex>

void renderFunc(AsyncQueueManager* q_manager, ImagePool* i_pool, std::mutex* release_req, std::mutex* release_rsp);

#endif