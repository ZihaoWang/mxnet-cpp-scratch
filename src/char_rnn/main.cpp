#include "common.h"
#include "hyp_container.h"
#include "utils.h"
#include "logger.h"

using namespace zh;
const string PROJ_ROOT("/misc/projdata12/info_fil/zhwang/workspace/mxnet_learn/");
const string CHAR_RNN_ROOT(PROJ_ROOT + "src/char_rnn/");

auto def_core(vector<pair<string, Shape>> &io_shapes,
        vector<pair<string, Shape>> &arg_shapes,
        HypContainer &hyp)
{
}

auto def_data_iter(HypContainer &hyp)
{
    const string &data_root = hyp.sget("data_root");

    return make_pair(train_iter, test_iter);
}

void init_args(map<string, NDArray> &args,
        map<string, NDArray> &grads,
        map<string, OpReqType> &grad_types,
        map<string, NDArray> &aux_states,
        const Context &ctx,
        vector<pair<string, Shape>> &io_shapes,
        vector<pair<string, Shape>> &arg_shapes,
        HypContainer &hyp)
{
    for (auto &duo : io_shapes)
    {
        args.insert({duo.first, NDArray(duo.second, ctx)});
        grads.insert({duo.first, NDArray(duo.second, ctx)});
        grad_types.insert({duo.first, kNullOp});
    }

    for (auto &duo : arg_shapes)
    {
        args.insert({duo.first, NDArray(duo.second, ctx)});
        grads.insert({duo.first, NDArray(duo.second, ctx)});
        grad_types.insert({duo.first, kWriteTo});
    }

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

    vector<pair<string, Shape>> io_shapes;
    vector<pair<string, Shape>> arg_shapes;
    auto core = def_core(io_shapes, arg_shapes, hyp);

    map<string, NDArray> args, grads, aux_states;
    map<string, OpReqType> grad_types;
    init_args(args, grads, grad_types, aux_states, ctx, io_shapes, arg_shapes, hyp);

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
    opt->SetParam("clip_gradient", 10.0);

    auto time_start = system_clock::now();
    for (int idx_epoch = hyp.iget("existing_epoch") + 1; idx_epoch <= hyp.iget("max_epoch"); ++idx_epoch)
    {
        logger.make_log("epoch " + to_string(idx_epoch));
        train_iter.Reset();
        float train_loss = 0.0;
        Accuracy train_acc;

        for (size_t idx_batch = 1; train_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &train_iter);
            exec->Forward(true);
            exec->Backward();
            exec->UpdateAll(opt.get(), hyp.fget("learning_rate"), hyp.fget("weight_decay"));

            train_acc.Update(args["y"], exec->outputs[1]);
            if (idx_batch % hyp.iget("print_freq") == 0)
            {
                vector<mx_float> tmp(hyp.iget("batch_size") * 10, 0.0);
                exec->outputs[0].SyncCopyToCPU(&tmp);
                for (auto e : tmp)
                    train_loss += e;
                logger.make_log("idx_batch = " + to_string(idx_batch) +
                        ", train_loss = " + to_string(train_loss / idx_batch) +
                        ", train acc = " + to_string(train_acc.Get()) +
                        ", time cost = " + to_string(get_time_interval(time_start)));
            }
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
        for (size_t idx_batch = 1; test_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &test_iter);
            unique_ptr<Executor> exec(core.SimpleBind(ctx, args, grads, grad_types, aux_states));
            exec->Forward(false);
            test_acc.Update(args["y"], exec->outputs[1]);
        }
        logger.make_log("test acc = " + to_string(test_acc.Get()) +
                ", time cost = " + to_string(get_time_interval(time_start)));
    }
}

int main(int argc, char** argv)
{
    //auto logger = make_unique<Logger>(cout, "", "char_rnn");
    auto logger = make_unique<Logger>(cout, PROJ_ROOT + "result/", "char_rnn");
    auto hyp = make_unique<HypContainer>(CHAR_RNN_ROOT + "char_rnn.hyp");
    logger->make_log(*hyp);

    run(*logger, *hyp);
    MXNotifyShutdown();

    return 0;
}
