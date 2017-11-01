#ifndef ZIHAO_UTILS
#define ZIHAO_UTILS

#include "./common.h"

namespace zh
{

/*
 * convenient wrapper
 *
 * convert a Shape into a vector<mx_uint>
 */
vector<mx_uint> shape2vec(Shape shape);

/*
 * save and load whole models
 */

// arg2: path for saving the model
// arg3: the name of arguments in a network that should not be saved (inputs, target, ...).
//
// save_model(exec, "./mlp.model", {"x", "y"})
void save_model(const Executor &exec, const string &path, const unordered_set<string> except_args = unordered_set<string>());

// arg2: path of model to be loaded
//
// Executor exec;
// ...
// load_model(&exec, "./mlp.model")
void load_model(Executor *exec, const string &path);

/*
 * print the information of a symbol by using mxnet's static inference
 * 
 * This function is used for debugging. If you wish to use these information, just directly call the mxnet function like InferShape() or my wrapper infer_output_shape().
 * This function is not reenterable.
 */

// arg1: the symbol to be inferred
// arg2: the name of input symbol of the whole computation graph
// arg3: the shape of input symbol of the whole computation graph
//
// string input_name("x");
// Shape input_shape(10, 4);
// auto x = Symbol(input_name);
// auto pred = FullyConnected(x, ...);
// const auto &output_shape = infer_output_shape(pred, input_name, input_shape);
void print_sym_info(const Symbol &sym, const string &input_name, Shape input_shape);

/*
 * convenient wrapper for inferring output shape of a symbol
 *
 * This function is not reenterable.
 */

// the parameter and usage is same as print_sym_info()
// return: shapes of all outputs of sym
const vector<vector<mx_uint>> &infer_output_shape(const Symbol &sym, const string &input_name, Shape input_shape);

} // namespace zh

#endif
