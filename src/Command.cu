#include "Command.cuh"

#include "Global.h"

AsyncQueueManager::AsyncQueueManager() 
{
    cmd_q = new concurrency::concurrent_queue<RenderCommand>();
    rslt_q = new concurrency::concurrent_queue<RenderResult>();
}

bool AsyncQueueManager::submitRenderCommand(const RenderCommand cmd)
{
    cmd_q->push(cmd);
    return true;
}

bool AsyncQueueManager::receiveRenderCommand(RenderCommand &cmd)
{
    return cmd_q->try_pop(cmd);
}

bool AsyncQueueManager::submitRenderResult(const RenderResult rslt)
{
    while(rslt_q->unsafe_size() >= QUEUE_SIZE) {}
    rslt_q->push(rslt);
    return true;
}

bool AsyncQueueManager::receiveRenderResult(RenderResult &rslt)
{
    return rslt_q->try_pop(rslt);
}
