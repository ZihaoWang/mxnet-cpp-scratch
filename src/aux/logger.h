#ifndef ZIHAO_LOGGER
#define ZIHAO_LOGGER

#include "./common.h"
#include "hyp_container.h"

namespace zh
{

// Such types can be placed in the Logger::watching_var.
// We use pointers because we want a reference to variables out of logger.
typedef variant<const int *, const size_t *,
        const double *, const float *,
        const string *, const bool *> WatchingVar;

struct WatchingVarPrinter : public boost::static_visitor<>
{
    WatchingVarPrinter(ostream &os, ofstream *ofs): os(os), ofs(ofs) {}

    // for WatchingVar
    template <typename VAR_T>
    void operator()(const VAR_T &e) const
    {
        os << *e;
        if (ofs)
            *ofs << *e;
    }

    private:
        ostream &os;
        ofstream *ofs;
};

struct HyperparameterPrinter : public boost::static_visitor<>
{
    HyperparameterPrinter(ostream &os, ofstream *ofs): os(os), ofs(ofs) {}

    template <typename VAR_T>
    void operator()(const VAR_T &e) const
    {
        os << e;
        if (ofs)
            *ofs << e;
    }

    private:
        ostream &os;
        ofstream *ofs;
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
        // if log_dir == "", Logger will not log to the file and just print to the console
        Logger(ostream &os, const string &log_dir, const string &log_prefix);

        ~Logger();

        Logger &add_var(const string &name, const WatchingVar &var);

        Logger &del_var(const string &name);

        void make_log(const string &msg);

        void make_log(const HypContainer &hc);

        void log_watching_var();

    private:
        template <typename VAR_T>
        Logger &do_log(const VAR_T &val);

        void flush_log();

        ostream &console;
        unique_ptr<ofstream> file;
        WatchingVarPrinter var_printer;
        HyperparameterPrinter hyp_printer;
        vector<pair<string, WatchingVar>> watching_var;
};

} // namespace zh


#endif
