#include "./utils.h"

namespace zh
{

unique_ptr<unordered_map<string, hyp_t>> load_hyp(const string &path)
{
    auto hyp = make_unique<unordered_map<string, hyp_t>>();
    const unordered_set<string> valid_types{"int", "double", "string", "bool"};

    ifstream ifs(path);
    if (!ifs)
        CRY("hyperparameter.txt does not exist: " + path);

    string line;
    string val_type;
    vector<string> splits;
    while (getline(ifs, line))
    {
        boost::trim(line);
        if (line.empty())
            continue;
        if (line[0] == '/' && line[1] == '/') // the line of comment
            continue;

        if (line[0] == '[') // parsing the line of type specification
        {
            if (line[line.size() - 1] != ']')
                CRY("the format of type specification is not correct: " + line + "\ncorrect example:\n[int]\n");
            val_type = string(line.begin() + 1, line.end() - 1);
            if (valid_types.find(val_type) == valid_types.end())
                CRY("the type is not valid: " + val_type + "\nvalid types: int, double, string, bool\n");
        }
        else // parsing the line of hyperparameters
        {
            boost::split(splits, line, boost::is_any_of("\t "));
            if (splits.size() == 1) // no space
                CRY("hyperparameter format is not correct (no space): " + line + "\ncorrect example:\nlr 1.0\n");

            const string &hyp_name = splits[0];
            if (splits.size() == 2) // this hyperparameter has only one value
            {
                const string &hyp_val = splits[1];
                switch (val_type[0])
                {
                    case 'i': // int
                        hyp->insert({hyp_name, hyp_t(std::stoi(hyp_val))});
                        break;
                    case 'd': // double
                        hyp->insert({hyp_name, hyp_t(std::stod(hyp_val))});
                        break;
                    case 's': // string
                        hyp->insert({hyp_name, hyp_t(hyp_val)});
                        break;
                    default: // bool
                        if (hyp_val == "true")
                            hyp->insert({hyp_name, hyp_t(true)});
                        else if (hyp_val == "false")
                            hyp->insert({hyp_name, hyp_t(false)});
                        else
                            CRY("the format of bool hyperparameter " + hyp_name + " is not valid: " + hyp_val +
                                    "\n correct type: true, false\n");
                        break;

                }
            }
            else // this hyperparameter has multiple values
            {
                using std::transform;
                switch (val_type[0])
                {
                    case 'i': // int
                    {
                        vector<int> hyp_val(splits.size() - 1, 0);
                        transform(splits.begin() + 1, splits.end(), hyp_val.begin(), [](const string &s){ return std::stoi(s); });
                        hyp->insert({hyp_name, hyp_t(hyp_val)});
                        break;
                    }
                    case 'd': // double
                    {
                        vector<double> hyp_val(splits.size() - 1, 0.0);
                        transform(splits.begin() + 1, splits.end(), hyp_val.begin(), [](const string &s){ return std::stod(s); });
                        hyp->insert({hyp_name, hyp_t(hyp_val)});
                        break;
                    }
                    case 's': // string
                        hyp->insert({hyp_name, hyp_t(vector<string>(splits.begin() + 1, splits.end()))});
                        break;
                    default: // bool
                    {
                        CRY("bool hyperparameter " + hyp_name + " can't have a list of values\n");
                        break;
                    }
                }
            }
            splits.clear();
        }
    }

    return hyp;
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


} // namespace zh
