#include "logger.h"

namespace zh
{

Logger::Logger(ostream &os, const string &log_dir, const string &log_prefix):
    console(os)
{
    time_t tt = system_clock::to_time_t(system_clock::now());
    string cur_time(ctime(&tt));
    size_t i = 0;
    for (; i < cur_time.size() - 1; ++i)
        if (cur_time[i] == ' ')
            cur_time[i] = '_';
    cur_time.erase(i); // erase the '\n' in the end

    string log_path(log_dir + log_prefix + "::" + cur_time + ".log");

    if (log_dir.empty())
        make_log("the log file is not created");
    else
    {
        file = make_unique<ofstream>();
        file->open(log_path);
        if (!*file)
            CRY("log file can't be created: " + log_path);
        make_log("the log file is at: " + log_path);
    }
    hyp_printer.set_stream(&console, file.get());
}

Logger::~Logger()
{
    file->close();
}

void Logger::make_log(const string &msg)
{
    do_log(msg);
    flush_log();
}

void Logger::make_log(const HypContainer &hc)
{
    auto hyps = make_unique<map<string, HypVal>>();
    for (const auto &duo : hc.hyp)
        hyps->emplace(duo);

    do_log("\nHyperparameters:\n");
    for (const auto &duo : *hyps)
    {
        string name(duo.first);
        do_log(name + " = ");
        boost::apply_visitor(hyp_printer, duo.second);
        flush_log();
    }
    flush_log();
}

template <typename VAR_T>
Logger &Logger::do_log(const VAR_T &val)
{
    console << val;
    if (file)
        *file << val;
    return *this;
}

void Logger::flush_log()
{
    console << endl;
    if (file)
        *file << endl;
}

} // namespace zh
