// ---------------------------------------------------------------------------
// core/logger.cpp -- logger implementation.
// ---------------------------------------------------------------------------
#include "core/logger.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>

namespace chappie
{

    const char *log_level_name(LogLevel level)
    {
        switch (level)
        {
        case LogLevel::Trace:
            return "TRACE";
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
        }
        return "?";
    }

    bool parse_log_level(const std::string &name, LogLevel &out)
    {
        if (name == "trace")
        {
            out = LogLevel::Trace;
            return true;
        }
        if (name == "debug")
        {
            out = LogLevel::Debug;
            return true;
        }
        if (name == "info")
        {
            out = LogLevel::Info;
            return true;
        }
        if (name == "warn" || name == "warning")
        {
            out = LogLevel::Warn;
            return true;
        }
        if (name == "error")
        {
            out = LogLevel::Error;
            return true;
        }
        return false;
    }

    Logger &Logger::instance()
    {
        static Logger logger;
        return logger;
    }

    void Logger::configure(LogLevel level, bool console, const std::string &file)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
        console_ = console;
        if (file_.is_open())
            file_.close();
        if (!file.empty())
        {
            file_.open(file, std::ios::app);
            if (!file_)
            {
                std::cerr << "[logger] warning: cannot open log file '" << file << "'\n";
            }
        }
    }

    namespace
    {
        std::string now_timestamp()
        {
            const std::time_t t = std::time(nullptr);
            std::tm tm_buf{};
#if defined(_WIN32)
            localtime_s(&tm_buf, &t);
#else
            localtime_r(&t, &tm_buf);
#endif
            std::ostringstream os;
            os << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
            return os.str();
        }
    } // namespace

    void Logger::log(LogLevel level, const std::string &message)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<int>(level) < static_cast<int>(level_))
            return;

        LogRecord record{level, message, now_timestamp()};

        // ring buffer
        history_.push_back(record);
        if (history_.size() > kHistoryLimit)
            history_.erase(history_.begin(),
                           history_.begin() + (history_.size() - kHistoryLimit));

        const char *tag = log_level_name(level);
        if (console_)
        {
            std::ostream &out =
                (level == LogLevel::Error || level == LogLevel::Warn) ? std::cerr
                                                                      : std::cout;
            out << "[" << record.timestamp << "] [" << tag << "] " << message << "\n";
            out.flush();
        }
        if (file_.is_open())
        {
            file_ << "[" << record.timestamp << "] [" << tag << "] " << message << "\n";
            file_.flush();
        }
    }

    std::vector<LogRecord> Logger::recent(std::size_t max) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (history_.size() <= max)
            return history_;
        return std::vector<LogRecord>(history_.end() - max, history_.end());
    }

    void Logger::clear_history()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        history_.clear();
    }

} // namespace chappie
