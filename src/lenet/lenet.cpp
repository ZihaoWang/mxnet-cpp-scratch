#include "common.h"
#include "hyp_container.h"
#include "utils.h"
#include "logger.h"

using namespace zh;
const string LENET_ROOT(PROJ_ROOT + "src/lenet/");

auto def_core(HypContainer &hyp)
{
    const auto &dim_conv_rker = hyp.viget("dim_conv_rker");
    const auto &dim_conv_cker = hyp.viget("dim_conv_cker");
    const auto &num_filter = hyp.viget("num_filter");
    const auto &dim_pool_rker = hyp.viget("dim_pool_rker");
    const auto &dim_pool_cker = hyp.viget("dim_pool_cker");
    const auto &dim_pool_rstrd = hyp.viget("dim_pool_rstrd");
    const auto &dim_pool_cstrd = hyp.viget("dim_pool_cstrd");

    auto x = make_sym("x");
    vector<Symbol> w_convs = {make_sym("w_conv1"), make_sym("w_conv2"), make_sym("w_conv3")};
    vector<Symbol> b_convs = {make_sym("b_conv1"), make_sym("b_conv2"), make_sym("b_conv3")};
    auto y = make_sym("y");

    vector<Symbol> layers;
    const Symbol *last = &x;
    for (size_t i = 0; i < dim_conv_rker.size(); ++i)
    {
        layers.push_back(Convolution(*last, w_convs[i], b_convs[i], Shape(dim_conv_rker[i], dim_conv_cker[i]), num_filter[i])); 
        layers.push_back(Activation(layers.back(), ActivationActType::kTanh));
        layers.push_back(Pooling(layers.back(), Shape(dim_pool_rker[i], dim_pool_cker[i]), PoolingPoolType::kMax, false, false, PoolingPoolingConvention::kValid, Shape(dim_pool_rstrd[i], dim_pool_cstrd[i])));
        last = &layers.back();
    }

    auto output = broadcast_mul(*last, sum(y));
    //auto output = SoftmaxOutput(h2, y);
    return output;

    /*
    vector<vector<string>> tmp;
    tmp.push_back(h1.ListArguments());
    tmp.push_back(h1.ListArguments());
    cout << tmp << endl;
    */
}

auto def_data_iter(HypContainer &hyp)
{
    const string &data_root = hyp.sget("data_root");
    auto train_iter = MXDataIter("MNISTIter")
        .SetParam("image", data_root + "train-images-idx3-ubyte")
        .SetParam("label", data_root + "train-labels-idx1-ubyte")
        .SetParam("batch_size", hyp.iget("batch_size"))
        .SetParam("flat", 0)
        .CreateDataIter();
    auto test_iter = MXDataIter("MNISTIter")
        .SetParam("image", data_root + "t10k-images-idx3-ubyte")
        .SetParam("label", data_root + "t10k-labels-idx1-ubyte")
        .SetParam("batch_size", hyp.iget("batch_size"))
        .SetParam("flat", 0)
        .CreateDataIter();

    return make_pair(train_iter, test_iter);
}

void init_args(map<string, NDArray> &args, map<string, NDArray> &grads, map<string, OpReqType> &grad_types, map<string, NDArray> &aux_states, const Context &ctx, HypContainer &hyp)
{
    const int batch_size = hyp.iget("batch_size");
    const auto &num_filter = hyp.viget("num_filter");
    const auto &dim_conv_rker = hyp.viget("dim_conv_rker");
    const auto &dim_conv_cker = hyp.viget("dim_conv_cker");

    const vector<pair<string, Shape>> io_shapes = {
        {"x", Shape(batch_size, hyp.iget("num_channel"), hyp.iget("img_row"), hyp.iget("img_col"))},
        {"y", Shape(batch_size)}
    };
    for (auto &duo : io_shapes)
    {
        args.insert({duo.first, NDArray(duo.second, ctx)});
        grads.insert({duo.first, NDArray()});
        grad_types.insert({duo.first, kNullOp});
    }

    const vector<pair<string, Shape>> arg_shapes = {
        {"w_conv1", Shape(num_filter[0], hyp.iget("num_channel"), dim_conv_rker[0], dim_conv_cker[0])},
        {"b_conv1", Shape(num_filter[0])},
        {"w_conv2", Shape(num_filter[1], num_filter[0], dim_conv_rker[1], dim_conv_cker[1])},
        {"b_conv2", Shape(num_filter[1])},
        {"w_conv3", Shape(num_filter[2], num_filter[1], dim_conv_rker[2], dim_conv_cker[2])},
        {"b_conv3", Shape(num_filter[2])}
    };
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

    arg_dict["x"].SyncCopyFromCPU(batch.data.Reshape(Shape(0, 1, 0, 0)).GetData(), batch.data.Size());
    arg_dict["x"].WaitToRead();
    arg_dict["y"].SyncCopyFromCPU(batch.label.GetData(), batch.label.Size());
    arg_dict["y"].WaitToRead();
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

    unique_ptr<Optimizer> opt(OptimizerRegistry::Find("adadelta"));
    opt->SetParam("rescale_grad", 1.0 / hyp.iget("batch_size"));

    int idx_epoch = hyp.iget("existing_epoch") + 1;
    float acc = 0.0;
    double time_cost = 0.0;
    logger.add_var("idx_epoch", &idx_epoch)
        .add_var("acc", &acc)
        .add_var("time_cost", &time_cost);

    auto tic = system_clock::now();
    for (; idx_epoch <= hyp.iget("max_epoch"); ++idx_epoch)
    {
        train_iter.Reset();
        for (size_t idx_batch = 0; train_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &train_iter);
            exec->Forward(true);
            for (auto e : exec->outputs)
                cout << e.GetShape() << endl;
            return;
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

        Accuracy acc_metric;
        test_iter.Reset();
        for (size_t idx_batch = 0; test_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &test_iter);
            unique_ptr<Executor> exec(core.SimpleBind(ctx, args, grads, grad_types, aux_states));
            exec->Forward(false);
            acc_metric.Update(args["y"], exec->outputs[0]);
        }

        auto toc = system_clock::now();
        time_cost = duration_cast<milliseconds>(toc - tic).count() / 1000.0;
        acc = acc_metric.Get();
        logger.log_watching_var();
    }
}

int main(int argc, char** argv)
{
    auto logger = make_unique<Logger>(cout, PROJ_ROOT + "result/", "lenet");
    auto hyp = make_unique<HypContainer>(LENET_ROOT + "lenet.hyp");
    logger->make_log("Hyperparameters:\n");
    logger->make_log(*hyp);

    run(*logger, *hyp);
    MXNotifyShutdown();

    return 0;
}
