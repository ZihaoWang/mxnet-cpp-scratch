#ifndef ZIHAO_UTILS
#define ZIHAO_UTILS

#include "./common.h"

namespace zh
{

/*
 * convenient mxnet wrappers
 */

inline auto make_sym(const string &name){ return Symbol::Variable(name); }

/*
 * save and load whole models
 */

// arg 3:
// the name of arguments in a network that should not be saved (inputs, target, ...).
//
// save_model(exec, "./mlp.model", {"x", "y"})
void save_model(const Executor &exec, const string &path, const unordered_set<string> except_args = unordered_set<string>());

// load_model(exec, "./mlp.model")
void load_model(Executor *exec, const string &path);

/*
 * print the information of a symbol by using mxnet's static inference
 * 
 * This function is used for debugging. If you wish to use these information, just directly call the relevant function like InferShape().
 */

// arg1: the infomation of input symbol of the whole computation graph: a name-shape pair
// arg2: the symbol to be inferred
//
// x = make_sym("x");
// auto pred = FullyConnected(x, ...);
// infer_shape({"x", {10, 4}}, pred);
void print_sym_info(const map<string, vector<mx_uint>> x_info, const Symbol &sym);

} // namespace zh

#endif
