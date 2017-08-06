#include "common.h"
#include "utils.h"
#include "logger.h"

const string data_root("/misc/projdata12/info_fil/zhwang/data/image/mnist/mnist_data/");
const string model_root("./model/");
const string model_prefix("mlp");
const size_t img_row = 28;
const size_t img_col = 28;
const size_t batch_size = 100;
const size_t max_epoch = 100;
const vector<size_t> dim_hidden = {128, 10};
const float learning_rate = 1.0;
const float weight_decay = 0.0;
const size_t save_freq = 10;

const bool load_existing_model = false;
const size_t existing_epoch = 0;
string existing_model_path(model_root + model_prefix + "_epoch" + to_string(existing_epoch));

using namespace zh;

auto def_core()
{
    auto s_x = make_sym("x");
    auto s_w1 = make_sym("w1");
    auto s_b1 = make_sym("b1");
    auto s_w2 = make_sym("w2");
    auto s_b2 = make_sym("b2");
    auto s_y = make_sym("y");

    auto s_h1 = sigmoid(broadcast_add(dot(s_x, s_w1), s_b1));
    auto s_h2 = sigmoid(broadcast_add(dot(s_h1, s_w2), s_b2));
    /*
    vector<vector<string>> tmp;
    tmp.push_back(s_h1.ListArguments());
    tmp.push_back(s_h1.ListArguments());
    cout << tmp << endl;
    */
    auto output = SoftmaxOutput(s_h2, s_y);
    return output;
}

auto def_data_iter()
{
    auto train_iter = MXDataIter("MNISTIter")
        .SetParam("image", data_root + "train-images-idx3-ubyte")
        .SetParam("label", data_root + "train-labels-idx1-ubyte")
        .SetParam("batch_size", batch_size)
        .SetParam("flat", 1)
        .CreateDataIter();
    auto test_iter = MXDataIter("MNISTIter")
        .SetParam("image", data_root + "t10k-images-idx3-ubyte")
        .SetParam("label", data_root + "t10k-labels-idx1-ubyte")
        .SetParam("batch_size", batch_size)
        .SetParam("flat", 1)
        .CreateDataIter();

    return make_tuple(train_iter, test_iter);
}

void init_args(map<string, NDArray> &args, map<string, NDArray> &grads, map<string, OpReqType> &grad_types, map<string, NDArray> &aux_states, const Context &ctx)
{
    size_t dim_x = img_row * img_col;
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
    exec->arg_dict()["x"].SyncCopyFromCPU(batch.data.GetData(), batch.data.Size());
    exec->arg_dict()["x"].WaitToRead();
    exec->arg_dict()["y"].SyncCopyFromCPU(batch.label.GetData(), batch.label.Size());
    exec->arg_dict()["y"].WaitToRead();
}

void run()
{
    Logger logger(cout, "result/", "mlp_gpu");

    auto duo = def_data_iter();
    auto train_iter = std::get<0>(duo);
    auto test_iter = std::get<1>(duo);

    auto ctx = Context::gpu(0);
    map<string, NDArray> args, grads, aux_states;
    map<string, OpReqType> grad_types;
    init_args(args, grads, grad_types, aux_states, ctx);

    auto core = def_core();
    unique_ptr<Executor> exec(core.SimpleBind(ctx, args, grads, grad_types, aux_states));
    if (load_existing_model)
    {
        cout << "load model: " << existing_model_path << endl;
        load_model(exec.get(), existing_model_path);
    }

    unique_ptr<Optimizer> opt(OptimizerRegistry::Find("adadelta"));
    opt->SetParam("rescale_grad", 1.0 / batch_size);

    size_t idx_epoch = existing_epoch + 1;
    float acc = 0.0;
    double time_cost = 0.0;
    logger.add_var("idx_epoch", &idx_epoch)
        .add_var("acc", &acc)
        .add_var("time_cost", &time_cost);

    auto tic = system_clock::now();
    for (; idx_epoch <= max_epoch; ++idx_epoch)
    {
        train_iter.Reset();
        for (size_t idx_batch = 0; train_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &train_iter);
            exec->Forward(true);
            exec->Backward();
            exec->UpdateAll(opt.get(), learning_rate, weight_decay);
        }
        if (idx_epoch % save_freq == 0)
        {
            string saving_model_path(model_root + model_prefix + "_epoch" + to_string(idx_epoch));
            cout << "save model: " << saving_model_path << endl;
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
    Logger logger(cout, PROJ_ROOT + "result/", "mlp_gpu");
    unique_ptr<unordered_map<string, hyp_t>> hyp = std::move(load_hyp(PROJ_ROOT + "hyperparameter.txt"));
    //vector<int> dim_hidden = boost::get<vector<int>>(hyp->at("dim_hidden"));
    logger.make_log("dim_hidden", hyp->at("dim_hidden"));
    exit(0);
    run();
    MXNotifyShutdown();

    return 0;
}