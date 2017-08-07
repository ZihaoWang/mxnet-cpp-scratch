#include "./utils.h"

namespace zh
{

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


} // namespace zh
