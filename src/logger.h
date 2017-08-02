#ifndef ZIHAO_LOGGER
#define ZIHAO_LOGGER

#include "./common.h"

namespace zh
{

struct WatchingVarPrinter : public boost::static_visitor<>
{
    WatchingVarPrinter(ostream &os, ofstream &ofs): os(os), ofs(ofs) {}

    // for watching_var_t
    template <typename VAR_T>
    void operator()(const VAR_T &e) const
    {
        os << *e;
        ofs << *e;
    }

    private:
        ostream &os;
        ofstream &ofs;
};

struct HyperparameterPrinter : public boost::static_visitor<>
{
    HyperparameterPrinter(ostream &os, ofstream &ofs): os(os), ofs(ofs) {}

    template <typename VAR_T>
    void operator()(const VAR_T &e) const
    {
        os << e;
        ofs << e;
    }

    private:
        ostream &os;
        ofstream &ofs;
};

// size_t epoch = 1;
// double cost = 0.0;
// string log_dir("./result/mlp/");
// string log_prefix("mlp");
// 
// Logger logger(cout, log_dir, log_prefix);
// logger.add_var("epoch", &epoch);
// logger.add_var("cost", &cost);
//
// train the network...
//
// logger.log_vars();
class Logger
{
    public:
        Logger(ostream &os, const string &log_dir, const string &log_prefix);

        ~Logger();

        Logger &add_var(const string &name, const watching_var_t &var);

        Logger &del_var(const string &name);

        void make_log(const string &msg);

        Logger &make_log(const string &name, const hyperparameter_t &var);

        void make_log(const unordered_map<string, hyperparameter_t> &var);

        void log_watching_var();

    private:
        template <typename VAR_T>
        Logger &do_log(const VAR_T &val);

        void flush_log();

        ostream &console;
        ofstream file;
        WatchingVarPrinter var_printer;
        HyperparameterPrinter hyp_printer;
        vector<pair<string, watching_var_t>> watching_var;
};

} // namespace zh


#endif
