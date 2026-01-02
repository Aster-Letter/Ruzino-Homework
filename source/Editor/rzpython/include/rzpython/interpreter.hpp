#pragma once

#include <rzconsole/ConsoleInterpreter.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "api.h"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace python {

class RZPYTHON_API PythonInterpreter : public console::Interpreter {
   public:
    PythonInterpreter();
    virtual ~PythonInterpreter();

    // Override base interpreter methods
    virtual Result Execute(std::string_view const cmdline) override;
    virtual std::vector<std::string> Suggest(
        std::string_view const cmdline,
        size_t cursor_pos) override;

   protected:
    // Override virtual methods for Python-specific behavior
    virtual bool ShouldHandleCommand(std::string_view command) const override;
    virtual Result HandleDirectExecution(std::string_view cmdline) override;
    virtual Result ExecuteCommand(
        std::string_view command,
        const std::vector<std::string>& args) override;
    virtual std::vector<std::string> SuggestCommand(
        std::string_view command,
        std::string_view cmdline,
        size_t cursor_pos) override;
    virtual bool IsValidCommand(std::string_view command) const override;

   private:
    Result ExecutePythonCode(std::string_view code);
    std::vector<std::string> SuggestPythonCompletion(std::string_view code);

    bool python_initialized_;
};

// Factory function for creating Python interpreter
RZPYTHON_API std::shared_ptr<console::Interpreter> CreatePythonInterpreter();

}  // namespace python

RUZINO_NAMESPACE_CLOSE_SCOPE
