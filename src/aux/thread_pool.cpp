#include "thread_pool.h"

void ThreadPool::thread_entry()
{
    while (1)
    {
        unique_lock<mutex> l(mut);
        while (!is_shutdown && job_queue.empty())
            signal_wakeup.wait(l, [this](){ return is_shutdown || !job_queue.empty(); });

        if (!job_queue.empty())
        {
            ++num_busy;
            auto job = job_queue.front();
            job_queue.pop();
            std::promise<double> cost;
            auto f = cost.get_future();
            l.unlock();

            job(move(cost));

            l.lock();
            total_cost += f.get();
            --num_busy;
            signal_finished.notify_one();
        }
        else if (is_shutdown)
            break;
    }
}


