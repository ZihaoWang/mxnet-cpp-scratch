#include "./utils.h"

namespace zh
{

vector<mx_uint> shape2vec(Shape shape)
{
    vector<mx_uint> vec(shape.ndim(), 0);

    for (size_t i = 0; i < shape.ndim(); ++i)
        vec[i] = static_cast<mx_uint>(shape[i]);
    return vec;
}

void save_model(const Executor &exec, const string &path, const unordered_set<string> except_args)
{
    map<string, NDArray> args;
    for (const auto &duo : const_cast<Executor &>(exec).arg_dict())
        if (except_args.find(duo.first) == except_args.end())
            args.insert({duo.first, duo.second});

    map<string, NDArray> auxs;
    for (const auto &duo : const_cast<Executor &>(exec).aux_dict())
        auxs.insert({duo.first, duo.second});

    NDArray::Save(path + "::args", args);
    NDArray::Save(path + "::auxs", auxs);
}

void load_model(Executor *exec, const string &path)
{
    auto args = NDArray::LoadToMap(path + "::args");
    auto arg_dst = exec->arg_dict(); 
    if (arg_dst.size() < args.size())
    {
        cout << "loaded args: " << endl;
        for (const auto iter : args)
            cout << iter.first << " ";
        cout << endl;
        cout << "model args: " << endl;
        for (const auto iter : arg_dst)
            cout << iter.first << " ";
        cout << endl;
        CRY("an invalid file is loaded: " + path);
    }
    for (const auto &duo : args)
    {
        const string &name = duo.first;
        if (arg_dst.find(name) != arg_dst.end())
            duo.second.CopyTo(&arg_dst[name]);
    }

    auto auxs = NDArray::LoadToMap(path + "::auxs");
    auto aux_dst = exec->aux_dict();
    if (aux_dst.size() < auxs.size())
    {
        cout << "loaded auxs: " << endl;
        for (const auto iter : auxs)
            cout << iter.first << " ";
        cout << endl;
        cout << "model auxs: " << endl;
        for (const auto iter : aux_dst)
            cout << iter.first << " ";
        cout << endl;
        CRY("an invalid file is loaded: " + path);
    }
    for (const auto &duo : auxs)
    {
        const string &name = duo.first;
        if (aux_dst.find(name) != aux_dst.end())
            duo.second.CopyTo(&aux_dst[name]);
    }
}

void print_sym_info(const Symbol &sym, const map<string, vector<mx_uint>> &input_info)
{
    static vector<vector<mx_uint>> arg_shapes, aux_shapes, out_shapes;

    arg_shapes.clear();
    aux_shapes.clear();
    out_shapes.clear();
    sym.InferShape(input_info, &arg_shapes, &aux_shapes, &out_shapes);
    auto args = sym.ListArguments();
    auto auxs = sym.ListAuxiliaryStates();
    auto outputs = sym.ListOutputs();
    
    cout << "argument information:" << (arg_shapes.empty() ? " null" : "") << endl;
    for (size_t i = 0; i < arg_shapes.size(); ++i)
        cout << "    arg" << i + 1 << ": " << args[i] << ", shape = (" << arg_shapes[i] << ")" << endl;
    cout << "auxiliary state information:" << (aux_shapes.empty() ? " null" : "") << endl;
    for (size_t i = 0; i < aux_shapes.size(); ++i)
        cout << "    aux" << i + 1 << ": " << auxs[i] << ", shape = (" << aux_shapes[i] << ")" << endl;
    cout << "output information:" << (out_shapes.empty() ? " null" : "") << endl;
    for (size_t i = 0; i < out_shapes.size(); ++i)
        cout << "    out" << i + 1 << ": " << outputs[i] << ", shape = (" << out_shapes[i] << ")" << endl;
}

const vector<vector<mx_uint>> &infer_output_shape(const Symbol &sym, const map<string, vector<mx_uint>> &input_info)
{
    static vector<vector<mx_uint>> arg_shapes, aux_shapes, out_shapes;

    arg_shapes.clear();
    aux_shapes.clear();
    out_shapes.clear();
    sym.InferShape(input_info, &arg_shapes, &aux_shapes, &out_shapes);

    return out_shapes;
}

} // namespace zh
