#pragma once
#include <nanobind/nanobind.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>

#include "rzpython.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE

namespace python {
namespace usd {

    namespace nb = nanobind;

    // Helper to ensure numpy and USD modules are available
    inline void ensure_usd_modules()
    {
        static bool modules_checked = false;
        if (!modules_checked) {
            try {
                python::call<void>("import numpy as np");
                python::call<void>("import pxr.Vt as Vt");
                python::call<void>("import pxr.Gf as Gf");
                modules_checked = true;
            }
            catch (...) {
                throw std::runtime_error(
                    "NumPy and USD modules are required for USD type "
                    "conversion but not available");
            }
        }
    }

    // Convert USD types to numpy arrays then to USD types in Python
    template<typename T>
    void send_usd(const std::string& name, const T& value)
    {
        ensure_usd_modules();

        if constexpr (std::is_same_v<T, pxr::VtArray<int>>) {
            // Create numpy array from VtArray<int> data
            std::vector<int> temp_data(value.begin(), value.end());
            python::send(name + "_np", temp_data);

            // Convert list to USD VtArray in Python - direct conversion
            python::call<void>(name + " = Vt.IntArray(" + name + "_np)");
        }
        else if constexpr (std::is_same_v<T, pxr::VtArray<float>>) {
            // Create numpy array from VtArray<float> data
            std::vector<float> temp_data(value.begin(), value.end());
            python::send(name + "_np", temp_data);

            // Convert to USD VtArray in Python - direct conversion
            python::call<void>(name + " = Vt.FloatArray(" + name + "_np)");
        }
        else if constexpr (std::is_same_v<T, pxr::GfVec3f>) {
            // Create numpy array from GfVec3f components
            std::vector<float> components = { value[0], value[1], value[2] };
            python::send(name + "_comp", components);

            // Convert to USD GfVec3f in Python - direct construction
            python::call<void>(
                name + " = Gf.Vec3f(" + name + "_comp[0], " + name +
                "_comp[1], " + name + "_comp[2])");
        }
        else if constexpr (std::is_same_v<T, pxr::GfVec4f>) {
            // Create numpy array from GfVec4f components
            std::vector<float> components = {
                value[0], value[1], value[2], value[3]
            };
            python::send(name + "_comp", components);

            // Convert to USD GfVec4f in Python - direct construction
            python::call<void>(
                name + " = Gf.Vec4f(" + name + "_comp[0], " + name +
                "_comp[1], " + name + "_comp[2], " + name + "_comp[3])");
        }
        else if constexpr (std::is_same_v<T, pxr::VtArray<pxr::GfVec3f>>) {
            // Flatten VtArray<GfVec3f> to numpy array
            std::vector<float> flattened;
            flattened.reserve(value.size() * 3);
            for (const auto& vec : value) {
                flattened.push_back(vec[0]);
                flattened.push_back(vec[1]);
                flattened.push_back(vec[2]);
            }

            // Send as numpy array and reshape directly
            python::send(name + "_flat", flattened);
            python::call<void>(
                name + "_reshaped = np.array(" + name +
                "_flat, dtype=np.float32).reshape(-1, 3)");
            // Convert numpy.float32 to Python float explicitly for Boost.Python
            // compatibility
            python::call<void>(
                name +
                " = Vt.Vec3fArray([Gf.Vec3f(float(row[0]), float(row[1]), "
                "float(row[2])) for row "
                "in " +
                name + "_reshaped])");
        }
        else if constexpr (std::is_same_v<T, pxr::VtArray<pxr::GfVec4f>>) {
            // Flatten VtArray<GfVec4f> to numpy array
            std::vector<float> flattened;
            flattened.reserve(value.size() * 4);
            for (const auto& vec : value) {
                flattened.push_back(vec[0]);
                flattened.push_back(vec[1]);
                flattened.push_back(vec[2]);
                flattened.push_back(vec[3]);
            }

            // Send as numpy array and reshape directly
            python::send(name + "_flat", flattened);
            python::call<void>(
                name + "_reshaped = np.array(" + name +
                "_flat, dtype=np.float32).reshape(-1, 4)");
            // Convert numpy.float32 to Python float explicitly for Boost.Python
            // compatibility
            python::call<void>(
                name +
                " = Vt.Vec4fArray([Gf.Vec4f(float(row[0]), float(row[1]), "
                "float(row[2]), float(row[3])) "
                "for row in " +
                name + "_reshaped])");
        }
        else {
            throw std::runtime_error("Unsupported USD type for send_usd");
        }
    }

    // Improved USD conversion using numpy as intermediate format
    template<typename T>
    T convert_boost_python_object(PyObject* py_obj)
    {
        ensure_usd_modules();

        if constexpr (std::is_same_v<T, pxr::VtArray<int>>) {
            // Convert USD VtArray to numpy array, then to std::vector, then to
            // VtArray Store the object temporarily to access it from Python
            PyDict_SetItemString(main_dict, "_temp_usd_obj", py_obj);

            try {
                // Convert to numpy array via Python - simplified
                python::call<void>("_temp_np = np.array(_temp_usd_obj)");
                std::vector<int> temp_vec =
                    python::call<std::vector<int>>("_temp_np.tolist()");

                // Convert to VtArray
                pxr::VtArray<int> result;
                result.assign(temp_vec.begin(), temp_vec.end());
                return result;
            }
            catch (...) {
                throw;
            }
        }
        else if constexpr (std::is_same_v<T, pxr::VtArray<float>>) {
            PyDict_SetItemString(main_dict, "_temp_usd_obj", py_obj);

            try {
                python::call<void>(
                    "_temp_np = np.array(_temp_usd_obj, dtype=np.float32)");
                std::vector<float> temp_vec =
                    python::call<std::vector<float>>("_temp_np.tolist()");

                pxr::VtArray<float> result;
                result.assign(temp_vec.begin(), temp_vec.end());
                return result;
            }
            catch (...) {
                throw;
            }
        }
        else if constexpr (std::is_same_v<T, pxr::GfVec3f>) {
            PyDict_SetItemString(main_dict, "_temp_usd_obj", py_obj);

            try {
                // Try to access components directly - simplified
                python::call<void>(
                    "_temp_comp = [float(_temp_usd_obj[0]), "
                    "float(_temp_usd_obj[1]), float(_temp_usd_obj[2])]");
                std::vector<float> components =
                    python::call<std::vector<float>>("_temp_comp");

                if (components.size() >= 3) {
                    return pxr::GfVec3f(
                        components[0], components[1], components[2]);
                }
            }
            catch (...) {
                // Fallback: try attribute access
                PyObject* x_attr = PyObject_GetAttrString(py_obj, "x");
                PyObject* y_attr = PyObject_GetAttrString(py_obj, "y");
                PyObject* z_attr = PyObject_GetAttrString(py_obj, "z");

                if (x_attr && y_attr && z_attr) {
                    float x = PyFloat_AsDouble(x_attr);
                    float y = PyFloat_AsDouble(y_attr);
                    float z = PyFloat_AsDouble(z_attr);
                    Py_DECREF(x_attr);
                    Py_DECREF(y_attr);
                    Py_DECREF(z_attr);
                    return pxr::GfVec3f(x, y, z);
                }
                Py_XDECREF(x_attr);
                Py_XDECREF(y_attr);
                Py_XDECREF(z_attr);
            }
        }
        else if constexpr (std::is_same_v<T, pxr::GfVec4f>) {
            PyDict_SetItemString(main_dict, "_temp_usd_obj", py_obj);

            try {
                // Try to access components directly - simplified
                python::call<void>(
                    "_temp_comp = [float(_temp_usd_obj[0]), "
                    "float(_temp_usd_obj[1]), float(_temp_usd_obj[2]), "
                    "float(_temp_usd_obj[3])]");
                std::vector<float> components =
                    python::call<std::vector<float>>("_temp_comp");

                if (components.size() >= 4) {
                    return pxr::GfVec4f(
                        components[0],
                        components[1],
                        components[2],
                        components[3]);
                }
            }
            catch (...) {
                // Fallback: try attribute access
                PyObject* x_attr = PyObject_GetAttrString(py_obj, "x");
                PyObject* y_attr = PyObject_GetAttrString(py_obj, "y");
                PyObject* z_attr = PyObject_GetAttrString(py_obj, "z");
                PyObject* w_attr = PyObject_GetAttrString(py_obj, "w");

                if (x_attr && y_attr && z_attr && w_attr) {
                    float x = PyFloat_AsDouble(x_attr);
                    float y = PyFloat_AsDouble(y_attr);
                    float z = PyFloat_AsDouble(z_attr);
                    float w = PyFloat_AsDouble(w_attr);
                    Py_DECREF(x_attr);
                    Py_DECREF(y_attr);
                    Py_DECREF(z_attr);
                    Py_DECREF(w_attr);
                    return pxr::GfVec4f(x, y, z, w);
                }
                Py_XDECREF(x_attr);
                Py_XDECREF(y_attr);
                Py_XDECREF(z_attr);
                Py_XDECREF(w_attr);
            }
        }

        throw std::runtime_error("Failed to convert Boost.Python USD object");
    }

    // USD-specific call function that handles Boost.Python objects
    template<typename T>
    T call_usd(const std::string& code)
    {
        PyObject* py_result = call_raw(code);
        if (!py_result) {
            throw std::runtime_error(
                "Failed to get result from Python code: " + code);
        }

        try {
            // For USD types, always use our custom conversion
            T result = convert_boost_python_object<T>(py_result);
            Py_DECREF(py_result);
            return result;
        }
        catch (const std::exception& e) {
            Py_DECREF(py_result);
            throw std::runtime_error(
                "Failed to convert USD result: " + std::string(e.what()));
        }
    }

}  // namespace usd
}  // namespace python

RUZINO_NAMESPACE_CLOSE_SCOPE
