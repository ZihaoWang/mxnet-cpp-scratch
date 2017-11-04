#ifndef ZIHAO_HYP_CONTAINER
#define ZIHAO_HYP_CONTAINER

#include "common.h"
#include <boost/algorithm/string.hpp>

namespace zh
{

// Such types can be used as hyperparameters.
// Using Logger::make_log() to print and log these hyperparameters.
typedef variant<int, float, string, bool,
       vector<int>, vector<float>,
       vector<string>> HypVal;

// convient wrappers
class HypContainer
{
    public:
        /*
         * load hyperparameters from file
         *
         * the format of contents in the file can be viewed from *.hyp under hyp/
         */
        HypContainer(const string &hyp_path);

        /*
         * convenient wrappers for boost::get<T>()
         */
        int &iget(const string &name)
        {
            check_name(name);
            return boost::get<int>(hyp.at(name));
        }

        float &fget(const string &name)
        {
            check_name(name);
            return boost::get<float>(hyp.at(name));
        }

        string &sget(const string &name)
        {
            check_name(name);
            return boost::get<string>(hyp.at(name));
        }

        bool &bget(const string &name)
        {
            check_name(name);
            return boost::get<bool>(hyp.at(name));
        }

        vector<int> &viget(const string &name)
        {
            check_name(name);
            return boost::get<vector<int>>(hyp.at(name));
        }

        vector<float> &vfget(const string &name)
        {
            check_name(name);
            return boost::get<vector<float>>(hyp.at(name));
        }

        vector<string> &vsget(const string &name)
        {
            check_name(name);
            return boost::get<vector<string>>(hyp.at(name));
        }

        unordered_map<string, HypVal> hyp;

    private:
        void check_name(const string &name)
        {
            if (hyp.find(name) != hyp.end())
                return;
            else
                CRY("This hyperparameter does not in the container: " + name);
        }
        
};


} // namespace zh

#endif
