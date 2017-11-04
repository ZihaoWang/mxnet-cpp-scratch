#ifndef ZIHAO_COMMON
#define ZIHAO_COMMON

#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <set>
#include <iterator>
#include <memory>
#include <algorithm>

#include <cmath>
#include <random>
#include <cstdlib>
#include <utility>

#include <limits>
#include <cctype>

#include <chrono>
#include <ctime>

#include <execinfo.h> // for stacktrace

//#include <boost/functional/hash.hpp>

#include <boost/variant.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>

//#include <armadillo> // if <mlpack/core.hpp> is includeded, this line should be commented
//#include <mlpack/core.hpp>

#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
//#include "mxnet-cpp/op.h"

using namespace mxnet::cpp;
using std::size_t;
using std::ifstream;
using std::ofstream;
using std::istream;
using std::ostream;
using std::cin;
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::map;
using std::set;
using std::unordered_map;
using std::unordered_set;
using std::pair;
using std::tuple;
using std::shared_ptr;
using std::unique_ptr;
using std::numeric_limits;
using std::make_shared;
using std::make_pair;
using std::make_tuple;
using std::tie;
using std::to_string;
//using std::make_unique;
using std::exit;
using std::chrono::system_clock;

const string PROJ_ROOT("/misc/projdata12/info_fil/zhwang/workspace/mxnet_learn/");

inline void CRY(const string &msg)
{
    cerr << "\nAn exception occurs!\n" << endl;

    cerr << "with message: " << msg << "\n" << endl;

    const int MAX_STACKTRACE_SIZE = 10;
    void *stacktraces[MAX_STACKTRACE_SIZE];
    int num_stack = backtrace(stacktraces, MAX_STACKTRACE_SIZE);
    char **stack_info = backtrace_symbols(stacktraces, num_stack);

    cerr << "stack trace returned " << num_stack - 1 << " entries" << endl;
    if (stack_info)
        for (int i = 1; i < num_stack; ++i)
        {
            cerr << "[bt] (" << i - 1 << ") " << stack_info[i] << endl;
            int j = 0;
            while(stack_info[i][j] != '(' && stack_info[i][j] != ' ' && stack_info[i][j] != 0)
                ++j;
        
            char command[256];
            sprintf(command, "addr2line %p -e %.*s", stacktraces[i], j, stack_info[i]); //last parameter is the file name of the symbol
            FILE *fp = popen(command, "r");
            if (!fp)
            {
                cerr << "fail to run command to print line number of stacktraces" << endl;
                continue;
            }
            char result[256];
            while (fgets(result, sizeof(result) - 1, fp))
                cerr << string(result) << endl;
            pclose(fp);
        }
    abort();
}

template <typename T, typename ... Args>
unique_ptr<T> make_unique(Args &&... args){ return unique_ptr<T>(new T(std::forward<Args>(args)...)); }

/*
 * functions for printing std::vector
 */

const size_t CONTAINER_MAX_PRINT_TIME = 100;

template <typename T>
ostream &operator<<(ostream &os, const vector<vector<T>> &val)
{
    if (val.size() <= CONTAINER_MAX_PRINT_TIME)
    {
        for (size_t i = 0; i < val.size() - 1; ++i)
            os << val[i] << endl;
        os << val.back();
    }
    else
    {
        for (size_t i = 0; i < CONTAINER_MAX_PRINT_TIME; ++i)
            os << val[i] << endl;
        os << "...";
    }

    return os;
}

template <typename T>
ostream &operator<<(ostream &os, const vector<T> &val)
{
    if (val.size() <= CONTAINER_MAX_PRINT_TIME)
    {
        for (size_t i = 0; i < val.size() - 1; ++i)
            os << val[i] << ", ";
        os << val.back();
    }
    else
    {
        for (size_t i = 0; i < CONTAINER_MAX_PRINT_TIME; ++i)
            os << val[i] << ", ";
        os << "...";
    }

    return os;
}

/*
 * time interval wrapper
 */
auto get_time_interval(std::chrono::time_point<system_clock> &time_start)
{
    auto time_end = system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() / 1000.0;
}

// boost::variant is adopted to construct heterogeneous containers.
// In order to output variables without explicitly casting to a certain type, we don't use boost::any.
using boost::variant;

#endif
