#include <nanobind/nanobind.h>

#include "GUI/widget.h"
#include "GUI/window.h"
#include "imgui.h"

using namespace USTC_CG;
namespace nb = nanobind;

NB_MODULE(GUI_py, m)
{
    m.doc() = "Python bindings for GUI Window and Widget classes";

    // Bind Window class - simplified version
    nb::class_<Window>(m, "Window")
        .def(nb::init<>(), "Create a new window")
        .def("run", &Window::run, "Start the main rendering loop")
        .def("get_elapsed_time", &Window::get_elapsed_time, "Get elapsed time in seconds");

    // Bind ImVec2 for drawing functions
    nb::class_<ImVec2>(m, "ImVec2")
        .def(nb::init<float, float>(), "Create a 2D vector")
        .def_rw("x", &ImVec2::x, "X coordinate")
        .def_rw("y", &ImVec2::y, "Y coordinate");

    // Bind ImColor
    nb::class_<ImColor>(m, "ImColor")
        .def(nb::init<float, float, float>(), "Create RGB color")
        .def(nb::init<float, float, float, float>(), "Create RGBA color");
}
