#include "common.h"
#include "hyp_container.h"
#include "utils.h"
#include "logger.h"
#include "capsule.h"

using namespace zh;
const string PROJ_ROOT("/misc/projdata12/info_fil/zhwang/workspace/mxnet_learn/");
const string CAPSULE_ROOT(PROJ_ROOT + "src/capsule/");

/*
 * the layer and variable names are the same as what in Hinton's paper "Dynamic Routing Between Capsules"
 */

auto def_core(vector<pair<string, Shape>> &io_shapes,
        vector<pair<string, Shape>> &arg_shapes,
        vector<pair<string, Shape>> &state_shapes,
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
    map<string, vector<mx_uint>> input_info = {{input_name, input_shape}};
    Symbol x(input_name);
    layers.push_back(x);
    io_shapes.push_back({input_name, Shape(input_shape)});

    layers.push_back(cap.conv1_layer(layers.back(), arg_shapes));

    layers.push_back(cap.primary_caps_layer(layers.back(), arg_shapes));

    // b_ij is an individual state which can't be updated like other arguments by gradient
    // so we treat is as an output and manually update it after Executor::Forward()
    const string b_ij_name("b_ij");
    Symbol b_ij(b_ij_name); // zeros(Shape(1152, 10));
    const auto &out_shapes = infer_output_shape(layers.back(), input_info);
    int num_capsule = out_shapes[0][1];
    state_shapes.push_back({b_ij_name, Shape(num_capsule, hyp.iget("dim_y"))});

    auto tup = cap.digit_caps_layer(layers.back(), b_ij, arg_shapes);
    auto pred = sqrt(sum(square(std::get<0>(tup)), Shape(1)));
    auto updated_b_ij = std::get<1>(tup);
    layers.push_back(pred);

    const string label_name("y");
    Symbol y(label_name);
    io_shapes.push_back({label_name, Shape(batch_size)});

    auto loss1 = sum(cap.margin_loss(pred, y), Shape(1));
    auto loss2 = sum(cap.reconstruct_loss(std::get<0>(tup), x, y, arg_shapes), Shape(1));
    auto final_loss = loss1 + hyp.fget("reconstruct_loss_weight") * loss2;
    final_loss = MakeLoss(final_loss);

    // we use Symbol::Group() to output multiple values
    // pred and updated_b_ij are not loss, so we use BlockGrad() here
    return Symbol::Group({final_loss,
            BlockGrad(pred),
            BlockGrad(updated_b_ij)
            }); 
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
        vector<pair<string, Shape>> &state_shapes,
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

    for (auto &duo : state_shapes)
    {
        args.insert({duo.first, NDArray(duo.second, ctx)});
        grads.insert({duo.first, NDArray(duo.second, ctx)});
        // we do not update states with gradients
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
    vector<pair<string, Shape>> state_shapes; // save individual states
    auto core = def_core(io_shapes, arg_shapes, state_shapes, hyp);
    auto arg_names = core.ListArguments();

    map<string, NDArray> args, grads, aux_states;
    map<string, OpReqType> grad_types;
    init_args(args, grads, grad_types, aux_states, ctx, io_shapes, arg_shapes, state_shapes, hyp);

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
    opt->SetParam("lr", hyp.fget("learning_rate"));
    opt->SetParam("wd", hyp.fget("weight_decay"));
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
            for (size_t i = 0; i < arg_names.size(); ++i)
            {
                const string &name = arg_names[i];
                if (name == "x" || name == "y")
                    continue;
                else if (name == "b_ij")
                    // unlike other parameters, we manually update b_ij by directly copy the updated_b_ij to original b_ij
                    exec->outputs[2].CopyTo(&exec->arg_arrays[i]);
                else
                    opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
            }

            train_acc.Update(args["y"], exec->outputs[1]);
            if (idx_batch % hyp.iget("print_freq") == 0)
            {
                vector<mx_float> tmp(hyp.iget("batch_size"), 0.0);
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
            exec->Forward(false);
            test_acc.Update(args["y"], exec->outputs[1]);
        }
        logger.make_log("test acc = " + to_string(test_acc.Get()) +
                ", time cost = " + to_string(get_time_interval(time_start)));
    }
}

int main(int argc, char** argv)
{
    auto logger = make_unique<Logger>(cout, "", "capsule");
    //auto logger = make_unique<Logger>(cout, PROJ_ROOT + "result/", "capsule");
    auto hyp = make_unique<HypContainer>(CAPSULE_ROOT + "capsule.hyp");
    logger->make_log(*hyp);

    run(*logger, *hyp);
    MXNotifyShutdown();

    return 0;
}
