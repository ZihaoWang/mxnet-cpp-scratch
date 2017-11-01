#include "common.h"
#include "utils.h"

using namespace zh;

auto def_core()
{
    auto s_x = Symbol("x");
    auto s_w1 = Symbol("w1");
    auto s_b1 = Symbol("b1");

    // some symbolic operators can't be broadcasted defaultly, instead they have their broadcastable version
    auto s_h1 = tanh(broadcast_add(dot(s_x, s_w1), s_b1));
    return s_h1;
}

auto def_exe()
{
    auto ctx_dev = Context::gpu(0);

    auto raw_x = make_unique<std::array<float, 4 * 3>>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            (*raw_x)[3 * i + j] = static_cast<float>(i * j);

    NDArray x(Shape(4, 3), ctx_dev, false);
    x.SyncCopyFromCPU(raw_x->data(), 4 * 3);
    x.WaitToRead();

    NDArray w1(Shape(3, 3), ctx_dev, false);
    w1 = 0.5f;
    NDArray b1(Shape(3), ctx_dev, false);
    b1 = 0.4f;
    NDArray w1_g(Shape(3, 3), ctx_dev, false);
    NDArray b1_g(Shape(3), ctx_dev, false);

    // the sequence of args follows topological order of variables in the computational graph
    vector<NDArray> in_args = {x, w1, b1};
    vector<NDArray> grads = {NDArray(), w1_g, b1_g};
    vector<OpReqType> grad_types = {kNullOp, kWriteTo, kWriteTo};
    vector<NDArray> aux_states;
    return make_unique<Executor>(def_core(), ctx_dev, in_args, grads, grad_types, aux_states);
}

void run()
{
    auto exe = def_exe();

    exe->Forward(true);
    vector<NDArray>& tmp = exe->outputs;
    auto out = make_unique<vector<float>>(4 * 3, 0.0);
    tmp[0].SyncCopyToCPU(out->data(), 4 * 3);
    NDArray::WaitAll();

    cout << *out << endl;
}

int main(int argc, char** argv)
{
    run();
    MXNotifyShutdown();

    return 0;
}
