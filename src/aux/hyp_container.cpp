#include "hyp_container.h"

namespace zh
{

HypContainer::HypContainer(const string &hyp_path)
{
    const unordered_set<string> valid_types{"int", "float", "string", "bool"};

    ifstream ifs(hyp_path);
    if (!ifs)
        CRY("hyperparameter.txt does not exist: " + hyp_path);

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
                CRY("the type is not valid: " + val_type + "\nvalid types: int, float, string, bool\n");
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
                        hyp.insert({hyp_name, HypVal(std::stoi(hyp_val))});
                        break;
                    case 'f': // float
                        hyp.insert({hyp_name, HypVal(std::stof(hyp_val))});
                        break;
                    case 's': // string
                        hyp.insert({hyp_name, HypVal(hyp_val)});
                        break;
                    default: // bool
                        if (hyp_val == "true")
                            hyp.insert({hyp_name, HypVal(true)});
                        else if (hyp_val == "false")
                            hyp.insert({hyp_name, HypVal(false)});
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
                        hyp.insert({hyp_name, HypVal(hyp_val)});
                        break;
                    }
                    case 'f': // float
                    {
                        vector<float> hyp_val(splits.size() - 1, 0.0);
                        transform(splits.begin() + 1, splits.end(), hyp_val.begin(), [](const string &s){ return std::stof(s); });
                        hyp.insert({hyp_name, HypVal(hyp_val)});
                        break;
                    }
                    case 's': // string
                        hyp.insert({hyp_name, HypVal(vector<string>(splits.begin() + 1, splits.end()))});
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
}

} // namespace zh
