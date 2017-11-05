#include "common.h"
#include "hyp_container.h"
#include "utils.h"
#include "logger.h"

using namespace zh;
const string PROJ_ROOT("/misc/projdata12/info_fil/zhwang/workspace/mxnet_learn/");
const string MLP_ROOT(PROJ_ROOT + "src/mlp/");

auto def_core(HypContainer &hyp)
{
    auto s_x = Symbol("x");
    auto s_w1 = Symbol("w1");
    auto s_b1 = Symbol("b1");
    auto s_w2 = Symbol("w2");
    auto s_b2 = Symbol("b2");
    auto s_y = Symbol("y");

    auto s_h1 = sigmoid(broadcast_add(dot(s_x, s_w1), s_b1));
    auto s_h2 = sigmoid(broadcast_add(dot(s_h1, s_w2), s_b2));
    auto output = SoftmaxOutput(s_h2, s_y);
    return output;
}

auto def_data_iter(HypContainer &hyp)
{
    const string &data_root = hyp.sget("data_root");
    auto train_iter = MXDataIter("MNISTIter")
        .SetParam("image", data_root + "train-images-idx3-ubyte")
        .SetParam("label", data_root + "train-labels-idx1-ubyte")
        .SetParam("batch_size", hyp.iget("batch_size"))
        .SetParam("flat", 1)
        .CreateDataIter();
    auto test_iter = MXDataIter("MNISTIter")
        .SetParam("image", data_root + "t10k-images-idx3-ubyte")
        .SetParam("label", data_root + "t10k-labels-idx1-ubyte")
        .SetParam("batch_size", hyp.iget("batch_size"))
        .SetParam("flat", 1)
        .CreateDataIter();

    return make_pair(train_iter, test_iter);
}

void init_args(map<string, NDArray> &args,
        map<string, NDArray> &grads,
        map<string, OpReqType> &grad_types,
        map<string, NDArray> &aux_states,
        const Context &ctx,
        HypContainer &hyp)
{
    const int dim_x = hyp.iget("img_row") * hyp.iget("img_col");
    const int batch_size = hyp.iget("batch_size");
    const auto &dim_hidden = hyp.viget("dim_hidden");

    args["x"] = NDArray(Shape(batch_size, dim_x), ctx);
    args["w1"] = NDArray(Shape(dim_x, dim_hidden[0]), ctx);
    args["b1"] = NDArray(Shape(dim_hidden[0]), ctx);
    args["w2"] = NDArray(Shape(dim_hidden[0], dim_hidden[1]), ctx);
    args["b2"] = NDArray(Shape(dim_hidden[1]), ctx);
    args["y"] = NDArray(Shape(batch_size), ctx);

    grads["x"] = NDArray();
    grads["w1"] = NDArray(Shape(dim_x, dim_hidden[0]), ctx);
    grads["b1"] = NDArray(Shape(dim_hidden[0]), ctx);
    grads["w2"] = NDArray(Shape(dim_hidden[0], dim_hidden[1]), ctx);
    grads["b2"] = NDArray(Shape(dim_hidden[1]), ctx);
    grads["y"] = NDArray();

    grad_types["x"] = kNullOp;
    grad_types["w1"] = kWriteTo;
    grad_types["b1"] = kWriteTo;
    grad_types["w2"] = kWriteTo;
    grad_types["b2"] = kWriteTo;
    grad_types["y"] = kNullOp;

    auto init = Xavier();
    for (auto &duo : args)
        init(duo.first, &duo.second);

    for (auto &duo : grads)
        if (grad_types[duo.first] == kWriteTo)
            duo.second = 0.0f;
}

void get_batch_data(const unique_ptr<Executor> &exec, DataIter *iter)
{
    const auto batch = iter->GetDataBatch();
    auto arg_dict = exec->arg_dict();

    arg_dict["x"].SyncCopyFromCPU(batch.data.GetData(), batch.data.Size());
    arg_dict["y"].SyncCopyFromCPU(batch.label.GetData(), batch.label.Size());
}

void run(Logger &logger, HypContainer &hyp)
{
    auto duo = def_data_iter(hyp);
    auto train_iter = std::get<0>(duo);
    auto test_iter = std::get<1>(duo);

    auto ctx = hyp.bget("using_gpu") ? Context::gpu(hyp.iget("idx_gpu")) : Context::cpu();

    map<string, NDArray> args, grads, aux_states;
    map<string, OpReqType> grad_types;
    init_args(args, grads, grad_types, aux_states, ctx, hyp);

    auto core = def_core(hyp);
    unique_ptr<Executor> exec(core.SimpleBind(ctx, args, grads, grad_types, aux_states));
    if (hyp.bget("load_existing_model"))
    {
        const string existing_model_path(PROJ_ROOT + "model/" +
                hyp.sget("model_prefix") +
                "_epoch" + to_string(hyp.iget("existing_epoch")));
        logger.make_log("load model: " + existing_model_path);
        load_model(exec.get(), existing_model_path);
    }

    unique_ptr<Optimizer> opt(OptimizerRegistry::Find(hyp.sget("optimizer")));
    opt->SetParam("rescale_grad", 1.0 / hyp.iget("batch_size"));

    auto time_start = system_clock::now();
    for (int idx_epoch = hyp.iget("existing_epoch") + 1; idx_epoch <= hyp.iget("max_epoch"); ++idx_epoch)
    {
        logger.make_log("epoch " + to_string(idx_epoch));
        train_iter.Reset();
        for (size_t idx_batch = 0; train_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &train_iter);
            exec->Forward(true);
            exec->Backward();
            exec->UpdateAll(opt.get(), hyp.fget("learning_rate"), hyp.fget("weight_decay"));
        }

        if (idx_epoch % hyp.iget("save_freq") == 0)
        {
            const string saving_model_path(PROJ_ROOT + "model/" +
                    hyp.sget("model_prefix") +
                    "_epoch" + to_string(idx_epoch));
            logger.make_log("save model: " + saving_model_path);
            save_model(*exec, saving_model_path, {"x", "y"});
        }

        Accuracy test_acc;
        test_iter.Reset();
        for (size_t idx_batch = 0; test_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &test_iter);
            unique_ptr<Executor> exec(core.SimpleBind(ctx, args, grads, grad_types, aux_states));
            exec->Forward(false);
            test_acc.Update(args["y"], exec->outputs[0]);
        }

        logger.make_log("test acc = " + to_string(test_acc.Get()) +
                ", time cost = " + to_string(get_time_interval(time_start)));
    }
}

int main(int argc, char** argv)
{
    auto logger = make_unique<Logger>(cout, "", "mlp");
    //auto logger = make_unique<Logger>(cout, PROJ_ROOT + "result/", "mlp");
    auto hyp = make_unique<HypContainer>(MLP_ROOT + "mlp.hyp");
    logger->make_log(*hyp);

    run(*logger, *hyp);
    MXNotifyShutdown();

    return 0;
}
