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
#include <algorithm>
#include <memory>

#include <cmath>
#include <random>
#include <cstdlib>
#include <utility>

#include <cinttypes>
#include <limits>
#include <type_traits>

#include <chrono>
#include <ctime>
//#include <functional>

#include <execinfo.h>

#include <boost/functional/hash.hpp>

//#include <armadillo> // if <mlpack/core.hpp> is includeded, this line should be commented
//#include <mlpack/core.hpp>

#include "mxnet-cpp/MxNetCpp.h"
// Allow IDE to parse the types
//#include "mxnet-cpp/op.h"

using namespace mxnet::cpp;
using std::size_t;
using std::fstream;
using std::ostream;
using std::ostream_iterator;
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
using std::make_tuple;
using std::tie;
using std::to_string;
//using std::make_unique;
using std::exit;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

inline void CRY(const string &msg)
{
    cerr << "\nAn exception occurs!\n" << endl;

    cerr << "in file: " << __FILE__ << endl;
    cerr << "function: " << __func__ << endl;
    cerr << "line: " << __LINE__ << endl;
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
 * template overloading for printing containers
 */

const size_t CONTAINER_MAX_PRINT_TIME = 100;

template <typename T>
ostream &operator<<(ostream &os, const vector<vector<T>> &val)
{
    for (size_t i = 0; i < val.size() - 1 && i < CONTAINER_MAX_PRINT_TIME; ++i)
        os << val[i] << endl;
    os << val.back();

    return os;
}

template <typename T>
ostream &operator<<(ostream &os, const vector<T> &val)
{
    for (size_t i = 0; i < val.size() - 1 && i < CONTAINER_MAX_PRINT_TIME; ++i)
        os << val[i] << " ";
    os << val.back();

    return os;
}

#endif
