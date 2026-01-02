#pragma once

#include <spdlog/common.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/base_sink.h>

#include <memory>
#include <mutex>

#include "api.h"

// Forward declare to avoid circular include
RUZINO_NAMESPACE_OPEN_SCOPE
class ImGui_Console;
RUZINO_NAMESPACE_CLOSE_SCOPE

RUZINO_NAMESPACE_OPEN_SCOPE

// Custom spdlog sink that forwards log messages to ImGui_Console
template<typename Mutex>
class console_sink : public spdlog::sinks::base_sink<Mutex> {
   public:
    console_sink() : console_(nullptr)
    {
    }

    void set_console(
        ImGui_Console* console);  // Forward declaration, implement in .cpp

   protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;
    void flush_() override
    {
    }

   private:
    ImGui_Console* console_;
};

using console_sink_mt = console_sink<std::mutex>;
using console_sink_st = console_sink<spdlog::details::null_mutex>;

// Global console sink instance
std::shared_ptr<console_sink_mt>& get_global_console_sink();

// Helper function to setup spdlog with console sink
RZCONSOLE_API void setup_console_logging(ImGui_Console* console);

// Helper function to create a logger with console sink
std::shared_ptr<spdlog::logger> create_console_logger(
    const std::string& logger_name,
    ImGui_Console* console);

RUZINO_NAMESPACE_CLOSE_SCOPE