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
struct HypContainer
{
    /*
     * load hyperparameters from file
     *
     * the format of contents in the file can be viewed from *.hyp under hyp/
     */
    HypContainer(const string &hyp_path);

    /*
     * convenient wrappers for boost::get<T>()
     */
    int &iget(const string &name);

    float &fget(const string &name);

    string &sget(const string &name);

    bool &bget(const string &name);

    vector<int> &viget(const string &name);

    vector<float> &vfget(const string &name);

    vector<string> &vsget(const string &name);

    unordered_map<string, HypVal> hyp;
};


} // namespace zh

#endif
