#include "./logger.h"

namespace zh
{

Logger::Logger(ostream &os, const string &log_dir, const string &log_prefix):
    console(os),
    file(),
    var_printer(console, file),
    hyp_printer(console, file)
{
    time_t tt = system_clock::to_time_t(system_clock::now());
    string cur_time(ctime(&tt));
    size_t i = 0;
    for (; i < cur_time.size() - 1; ++i)
        if (cur_time[i] == ' ')
            cur_time[i] = '_';
    cur_time.erase(i); // erase the '\n' in the end

    string log_path(log_dir + log_prefix + "::" + cur_time + ".log");

    file.open(log_path);
    if (!file)
        CRY("log file can't be created: " + log_path);

    console << "Logger has been initlized, the log file is at: " << log_path << endl;
    file << "Logger has been initlized, the log file is at: " << log_path << endl;
}

Logger::~Logger()
{
    file.close();
}

Logger &Logger::add_var(const string &name, const watching_var_t &var)
{
    watching_var.emplace_back(name, var);
    return *this;
}

Logger &Logger::del_var(const string &name)
{
    auto iter = watching_var.find(name);
    if (iter != watching_var.end())
        watching_var.erase(iter);
    return *this;
}

void Logger::make_log(const string &msg)
{
    do_log(msg);
}

Logger &Logger::make_log(const string &name, const hyperparameter_t &var)
{
    do_log(name).do_log(" = ");
    boost::apply_visitor(hyp_printer, var);
    flush_log();
    return *this;
}

void Logger::make_log(const unordered_map<string, hyperparameter_t> &var)
{
    for (const auto &duo : var)
        make_log(duo.first, duo.second);
}

void Logger::log_watching_var()
{
    for (const auto &duo : watching_var)
    {
        do_log(duo.first).do_log(" = ");
        boost::apply_visitor(var_printer, duo.second);
        do_log("\t");
    }
    flush_log();
}

template <typename VAR_T>
Logger &Logger::do_log(const VAR_T &val)
{
    console << val;
    file << val;
    return *this;
}

void Logger::flush_log()
{
    console << endl;
    file << endl;
}

} // namespace zh
