#include "common.h"
#include "hyp_container.h"
#include "utils.h"
#include "logger.h"

using namespace zh;
const string LENET_ROOT(PROJ_ROOT + "src/lenet/");

auto def_core(vector<pair<string, Shape>> &io_shapes, vector<pair<string, Shape>> &arg_shapes, HypContainer &hyp)
{
    const unsigned int &batch_size = hyp.iget("batch_size");
    const auto &dim_conv_rker = hyp.viget("dim_conv_rker");
    const auto &dim_conv_cker = hyp.viget("dim_conv_cker");
    const auto &num_filter = hyp.viget("num_filter");
    const auto &dim_pool_rker = hyp.viget("dim_pool_rker");
    const auto &dim_pool_cker = hyp.viget("dim_pool_cker");
    const auto &dim_pool_rstrd = hyp.viget("dim_pool_rstrd");
    const auto &dim_pool_cstrd = hyp.viget("dim_pool_cstrd");
    const auto &dim_fc = hyp.viget("dim_fc");

    vector<Symbol> layers;

    const string input_name("x");
    vector<mx_uint> input_shape = {
        batch_size,
        static_cast<mx_uint>(hyp.iget("num_channel")),
        static_cast<mx_uint>(hyp.iget("img_row")),
        static_cast<mx_uint>(hyp.iget("img_col"))
    };
    map<string, vector<mx_uint>> input_info = {{"x", input_shape}};
    layers.push_back(Symbol(input_name));
    io_shapes.push_back({input_name, Shape(input_shape)});

    // convolution-tanh-pooling layers
    for (size_t i = 0; i < dim_conv_rker.size(); ++i)
    {
        const string w_name("w_conv" + to_string(i + 1));
        const string b_name("b_conv" + to_string(i + 1));
        const int num_filter_last = i == 0 ? hyp.iget("num_channel") : num_filter[i - 1];

        layers.push_back(Convolution(layers.back(),
                    Symbol(w_name),
                    Symbol(b_name),
                    Shape(dim_conv_rker[i], dim_conv_cker[i]),
                    num_filter[i])); 
        arg_shapes.push_back({w_name, Shape(num_filter[i], num_filter_last, dim_conv_rker[i], dim_conv_cker[i])});
        arg_shapes.push_back({b_name, Shape(num_filter[i])});

        layers.push_back(Activation(layers.back(), ActivationActType::kTanh));

        layers.push_back(Pooling(layers.back(),
                    Shape(dim_pool_rker[i], dim_pool_cker[i]),
                    PoolingPoolType::kMax,
                    false,
                    false,
                    PoolingPoolingConvention::kValid,
                    Shape(dim_pool_rstrd[i], dim_pool_cstrd[i])));
    }

    layers.push_back(Flatten(layers.back()));
    const auto &out_shapes = infer_output_shape(layers.back(), input_info);
    int dim_flatten = out_shapes[0][1];

    // fully-connected layers
    layers.push_back(FullyConnected(layers.back(),
                Symbol("w_fc1"),
                Symbol("b_fc1"),
                dim_fc[0]));
    arg_shapes.push_back({"w_fc1", Shape(dim_fc[0], dim_flatten)});
    arg_shapes.push_back({"b_fc1", Shape(dim_fc[0])});
    layers.push_back(Activation(layers.back(), ActivationActType::kTanh));

    layers.push_back(FullyConnected(layers.back(),
                Symbol("w_fc2"),
                Symbol("b_fc2"),
                dim_fc[1]));
    arg_shapes.push_back({"w_fc2", Shape(dim_fc[1], dim_fc[0])});
    arg_shapes.push_back({"b_fc2", Shape(dim_fc[1])});

    // softmax loss
    layers.push_back(SoftmaxOutput(layers.back(), Symbol("y")));
    io_shapes.push_back({"y", Shape(batch_size)});

    return layers.back();
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
        grads.insert({duo.first, NDArray()});
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

    unique_ptr<Optimizer> opt(OptimizerRegistry::Find("adadelta"));
    opt->SetParam("rescale_grad", 1.0 / hyp.iget("batch_size"));

    int idx_epoch = hyp.iget("existing_epoch") + 1;
    float train_loss = 0.0;
    float acc = 0.0;
    double time_cost = 0.0;
    logger.add_var("idx_epoch", &idx_epoch)
        .add_var("train_loss", &train_loss)
        .add_var("acc", &acc)
        .add_var("time_cost", &time_cost);

    auto ce_loss = make_unique<LogLoss>();
    auto tic = system_clock::now();
    for (; idx_epoch <= hyp.iget("max_epoch"); ++idx_epoch)
    {
        train_iter.Reset();
        ce_loss->Reset();
        for (size_t idx_batch = 0; train_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &train_iter);
            exec->Forward(true);
            exec->Backward();
            exec->UpdateAll(opt.get(), hyp.fget("learning_rate"), hyp.fget("weight_decay"));
            ce_loss->Update(args["y"], exec->outputs[0]);
        }
        train_loss = ce_loss->Get();

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
    auto logger = make_unique<Logger>(cout, "", "lenet");
    //auto logger = make_unique<Logger>(cout, PROJ_ROOT + "result/", "lenet");
    auto hyp = make_unique<HypContainer>(LENET_ROOT + "lenet.hyp");
    logger->make_log("Hyperparameters:\n");
    logger->make_log(*hyp);

    run(*logger, *hyp);
    MXNotifyShutdown();

    return 0;
}
