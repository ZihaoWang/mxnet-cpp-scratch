#include "common.h"
#include "hyp_container.h"
#include "utils.h"
#include "logger.h"
#include "capsule.h"

using namespace zh;
const string CAPSULE_ROOT(PROJ_ROOT + "src/capsule/");

auto def_core(vector<pair<string, Shape>> &io_shapes,
        vector<pair<string, Shape>> &arg_shapes,
        vector<pair<string, Shape>> &aux_arg_shapes,
        HypContainer &hyp)
{
    const int &batch_size = hyp.iget("batch_size");
    CapsuleConv cap(hyp);
    vector<Symbol> layers;

    const string input_name("x");
    vector<mx_uint> input_shape = {
        static_cast<mx_uint>(batch_size),
        static_cast<mx_uint>(hyp.iget("num_channel")),
        static_cast<mx_uint>(hyp.iget("img_row")),
        static_cast<mx_uint>(hyp.iget("img_col"))
    };
    map<string, vector<mx_uint>> input_info = {{"x", input_shape}};
    Symbol x(input_name);
    layers.push_back(x);
    io_shapes.push_back({input_name, Shape(input_shape)});

    layers.push_back(cap.conv1_layer(layers.back(), arg_shapes));

    layers.push_back(cap.primary_caps_layer(layers.back(), arg_shapes));

    const string b_ij_name("b_ij");
    Symbol b_ij(b_ij_name); // zeros(Shape(1152, 10));
    const auto &out_shapes = infer_output_shape(layers.back(), input_info);
    int num_capsule = out_shapes[0][1];
    aux_arg_shapes.push_back({b_ij_name, Shape(num_capsule, hyp.iget("dim_y"))});

    auto duo = cap.digit_caps_layer(layers.back(), b_ij, arg_shapes);
    auto pred = sqrt(sum(square(duo.first), Shape(1)));
    layers.push_back(pred);

    const string label_name("y");
    Symbol y(label_name);
    io_shapes.push_back({label_name, Shape(batch_size)});

    Symbol loss1 = cap.margin_loss(pred, y);
    //Symbol loss = MakeLoss(sum(cap.margin_loss(layers.back(), y)) +
    //Symbol loss2 = hyp.fget("reconstruct_loss_weight") * cap.reconstruct_loss(layers.back(), x, y, arg_shapes);
    return Symbol::Group({loss1, BlockGrad(pred), BlockGrad(duo.second)});
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
        vector<pair<string, Shape>> &aux_arg_shapes,
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

    for (auto &duo : aux_arg_shapes)
    {
        args.insert({duo.first, NDArray(duo.second, ctx)});
        grads.insert({duo.first, NDArray(duo.second, ctx)});
        grad_types.insert({duo.first, kNullOp});
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
    vector<pair<string, Shape>> aux_arg_shapes;
    auto core = def_core(io_shapes, arg_shapes, aux_arg_shapes, hyp);

    map<string, NDArray> args, grads, aux_states;
    map<string, OpReqType> grad_types;
    init_args(args, grads, grad_types, aux_states, ctx, io_shapes, arg_shapes, aux_arg_shapes, hyp);

    unique_ptr<Executor> exec(core.SimpleBind(ctx, args, grads, grad_types, aux_states));
    if (hyp.bget("load_existing_model"))
    {
        const string existing_model_path(PROJ_ROOT + "model/" +
                hyp.sget("model_prefix") +
                "_epoch" + to_string(hyp.iget("existing_epoch")));
        logger.make_log("load model: " + existing_model_path);
        load_model(exec.get(), existing_model_path);
    }

    unique_ptr<Optimizer> opt(OptimizerRegistry::Find("adam"));
    opt->SetParam("rescale_grad", 1.0 / hyp.iget("batch_size"));
    opt->SetParam("clip_gradient", 10.0);

    int idx_epoch = hyp.iget("existing_epoch") + 1;
    auto tic = system_clock::now();
    for (; idx_epoch <= hyp.iget("max_epoch"); ++idx_epoch)
    {
        logger.make_log("epoch " + to_string(idx_epoch));
        train_iter.Reset();
        float train_loss = 0.0;
        Accuracy train_acc;

        for (size_t idx_batch = 1; train_iter.Next(); ++idx_batch)
        {
            get_batch_data(exec, &train_iter);
            exec->Forward(true);
            //cout << "output shape: " << exec->outputs[1].GetShape() << endl;
            //return;
            exec->Backward();
            /*
            for (auto e : exec->arg_arrays)
                cout << e.GetShape() << endl;
            return;
            */
            exec->UpdateAll(opt.get(), hyp.fget("learning_rate"), hyp.fget("weight_decay"));

            exec->outputs[2].CopyTo(&exec->arg_dict()["b_ij"]);
            //cout << exec->arg_dict()["w_ij"].Slice(0, 1) << endl;

            train_acc.Update(args["y"], exec->outputs[1]);
            if (idx_batch % hyp.iget("print_freq") == 0)
            {
                vector<mx_float> tmp(hyp.iget("batch_size") * 10, 0.0);
                exec->outputs[0].SyncCopyToCPU(&tmp);
                for (auto e : tmp)
                    train_loss += e;
                logger.make_log("idx_batch = " + to_string(idx_batch) +
                        ", train_loss = " + to_string(train_loss / idx_batch) +
                        ", train acc = " + to_string(train_acc.Get()));
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

        auto toc = system_clock::now();
        auto time_cost = duration_cast<milliseconds>(toc - tic).count() / 1000.0;
        logger.make_log("total training loss = " + to_string(train_loss / idx_batch) +
                ", test acc = " + to_string(test_acc.Get()) +
                ", time cost = " + to_string(time_cost));
    }
}

int main(int argc, char** argv)
{
    auto logger = make_unique<Logger>(cout, "", "capsule");
    //auto logger = make_unique<Logger>(cout, PROJ_ROOT + "result/", "capsule");
    auto hyp = make_unique<HypContainer>(CAPSULE_ROOT + "capsule.hyp");
    logger->make_log("Hyperparameters:\n");
    logger->make_log(*hyp);

    run(*logger, *hyp);
    MXNotifyShutdown();

    return 0;
}
