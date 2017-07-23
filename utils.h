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

} // namespace zh

#endif
