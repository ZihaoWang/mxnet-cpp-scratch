#ifndef ZIHAO_THREAD_POOL
#define ZIHAO_THREAD_POOL

#include "common.h"
#include <queue>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>

using std::thread;
using std::mutex;
using std::lock_guard;
using std::unique_lock;
using std::condition_variable;

class ThreadPool
{
    public:
    ThreadPool(const size_t num_thread):
        total_cost(0.0),
        num_busy(0),
        is_shutdown(false)
    {
        thread_pool.reserve(num_thread);
        for (size_t i = 0; i < num_thread; ++i)
            thread_pool.emplace_back(std::move([this](){ thread_entry(); }));
    }

    ~ThreadPool()
    {
        {
            // Unblock any threads and tell them to stop
            unique_lock<mutex> l(mut);
            is_shutdown = true;
            signal_wakeup.notify_all();
        }

        // Wait for all threads to stop
        cout << "ThreadPool is destroyed and remain threads are joining" << endl;
        for (auto &thread : thread_pool)
            thread.join();
    }

    void do_job(std::function<void(std::promise<double>)> fn)
    {
        // Place a job on the queue and unblock a thread
        unique_lock<mutex> l(mut);
        job_queue.emplace(move(fn));
        signal_wakeup.notify_one();
    }

    double wait_finished()
    {
        unique_lock<mutex> l(mut);
        signal_finished.wait(l, [this](){ return job_queue.empty() && (num_busy == 0); });
        
        double tmp = total_cost;
        total_cost = 0.0;
        return tmp;
    }

    protected:

    void thread_entry();

    std::queue<std::function <void(std::promise<double>)>> job_queue;
    vector<thread> thread_pool;
    double total_cost;

    mutex mut;
    condition_variable signal_wakeup;
    condition_variable signal_finished;
    size_t num_busy;
    bool is_shutdown;
};

#endif
