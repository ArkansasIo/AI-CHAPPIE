// ---------------------------------------------------------------------------
// core/logger.hpp -- leveled logging to console and file.
// ---------------------------------------------------------------------------
#pragma once

#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace chappie
{

    enum class LogLevel
    {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4
    };

    const char *log_level_name(LogLevel level);
    bool parse_log_level(const std::string &name, LogLevel &out);

    // A single log record (also stored in memory for the GUI to display).
    struct LogRecord
    {
        LogLevel level;
        std::string message;
        std::string timestamp;
    };

    class Logger
    {
    public:
        static Logger &instance();

        void configure(LogLevel level, bool console, const std::string &file);

        void log(LogLevel level, const std::string &message);

        // Formatted variants (printf-free; use the stream helpers below).
        void trace(const std::string &msg) { log(LogLevel::Trace, msg); }
        void debug(const std::string &msg) { log(LogLevel::Debug, msg); }
        void info(const std::string &msg) { log(LogLevel::Info, msg); }
        void warn(const std::string &msg) { log(LogLevel::Warn, msg); }
        void error(const std::string &msg) { log(LogLevel::Error, msg); }

        // In-memory ring buffer of recent records (for the GUI log panel).
        std::vector<LogRecord> recent(std::size_t max = 200) const;
        void clear_history();

        LogLevel level() const { return level_; }

    private:
        Logger() = default;

        mutable std::mutex mutex_;
        LogLevel level_ = LogLevel::Info;
        bool console_ = true;
        std::ofstream file_;
        std::vector<LogRecord> history_;
        static constexpr std::size_t kHistoryLimit = 1000;
    };

    // Convenience stream-based formatting: LOG_INFO() << "x = " << x;
    class LogStream
    {
    public:
        explicit LogStream(LogLevel level) : level_(level) {}
        ~LogStream() { Logger::instance().log(level_, stream_.str()); }
        template <typename T>
        LogStream &operator<<(const T &value)
        {
            stream_ << value;
            return *this;
        }

    private:
        LogLevel level_;
        std::ostringstream stream_;
    };

#define LOG_TRACE() ::chappie::LogStream(::chappie::LogLevel::Trace)
#define LOG_DEBUG() ::chappie::LogStream(::chappie::LogLevel::Debug)
#define LOG_INFO() ::chappie::LogStream(::chappie::LogLevel::Info)
#define LOG_WARN() ::chappie::LogStream(::chappie::LogLevel::Warn)
#define LOG_ERROR() ::chappie::LogStream(::chappie::LogLevel::Error)

} // namespace chappie
