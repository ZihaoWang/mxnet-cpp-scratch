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

struct HyperparameterPrinter : public boost::static_visitor<>
{
    HyperparameterPrinter(): os(nullptr), ofs(nullptr) {}

    void set_stream(ostream *out_os, ofstream *out_ofs)
    {
        os = out_os;
        ofs = out_ofs;
    }

    template <typename VAR_T>
    void operator()(const VAR_T &e) const
    {
        *os << e;
        if (ofs)
            *ofs << e;
    }

    private:
        ostream *os;
        ofstream *ofs;
};

class Logger
{
    public:
        // if log_dir == "", Logger will not log to the file and just print to the console
        Logger(ostream &os, const string &log_dir, const string &log_prefix);

        ~Logger();

        void make_log(const string &msg);

        void make_log(const HypContainer &hc);

    private:
        template <typename VAR_T>
        Logger &do_log(const VAR_T &val);

        void flush_log();

        ostream &console;
        unique_ptr<ofstream> file;
        HyperparameterPrinter hyp_printer;
};

} // namespace zh


#endif
