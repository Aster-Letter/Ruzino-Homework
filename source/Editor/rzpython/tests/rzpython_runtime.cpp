#include <gtest/gtest.h>

#include "GUI/window.h"
#include "rzpython/rzpython.hpp"

using namespace USTC_CG;

TEST(RZPythonRuntimeTest, BasicFunctionality)
{
    python::initialize();

    Window window;
    window.run();

    python::reference("w", &window);  // Or some other kind of reference


    
}
