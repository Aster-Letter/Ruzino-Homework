#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include "../../geometry_nodes/fem_bem/parameter_map.hpp"

using namespace USTC_CG::fem_bem;

TEST(ParameterMapPerformanceTest, ConstructionAndDestruction)
{
    const int iterations = 100000;
    
    // Test ParameterMap performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ParameterMapD param_map;
        param_map.insert_or_assign("u", 1.0);
        param_map.insert_or_assign("v", 2.0);
        
        auto* u_val = param_map.find("u");
        auto* v_val = param_map.find("v");
        
        if (u_val && v_val) {
            double result = *u_val + *v_val;
            (void)result; // Avoid unused variable warning
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto param_map_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test std::unordered_map performance for comparison
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::unordered_map<std::string, double> std_map;
        std_map["u"] = 1.0;
        std_map["v"] = 2.0;
        
        auto u_it = std_map.find("u");
        auto v_it = std_map.find("v");
        
        if (u_it != std_map.end() && v_it != std_map.end()) {
            double result = u_it->second + v_it->second;
            (void)result; // Avoid unused variable warning
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto std_map_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "ParameterMap: " << param_map_duration.count() << " microseconds\n";
    std::cout << "std::unordered_map: " << std_map_duration.count() << " microseconds\n";
    std::cout << "Speedup: " << static_cast<double>(std_map_duration.count()) / param_map_duration.count() << "x\n";
    
    // ParameterMap should be faster for small maps
    EXPECT_LT(param_map_duration.count(), std_map_duration.count());
}

TEST(ParameterMapPerformanceTest, InitializerListConstruction)
{
    const int iterations = 50000;
    
    // Test ParameterMap with initializer list
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ParameterMapD param_map = {{"u", 1.0}, {"v", 2.0}, {"w", 3.0}};
        
        auto* u_val = param_map.find("u");
        auto* v_val = param_map.find("v");
        auto* w_val = param_map.find("w");
        
        if (u_val && v_val && w_val) {
            double result = *u_val + *v_val + *w_val;
            (void)result;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto param_map_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test std::unordered_map with initializer list
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::unordered_map<std::string, double> std_map = {{"u", 1.0}, {"v", 2.0}, {"w", 3.0}};
        
        auto u_it = std_map.find("u");
        auto v_it = std_map.find("v");
        auto w_it = std_map.find("w");
        
        if (u_it != std_map.end() && v_it != std_map.end() && w_it != std_map.end()) {
            double result = u_it->second + v_it->second + w_it->second;
            (void)result;
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto std_map_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "ParameterMap (init list): " << param_map_duration.count() << " microseconds\n";
    std::cout << "std::unordered_map (init list): " << std_map_duration.count() << " microseconds\n";
    std::cout << "Speedup: " << static_cast<double>(std_map_duration.count()) / param_map_duration.count() << "x\n";
}

TEST(ParameterMapPerformanceTest, CopyAndMove)
{
    const int iterations = 50000;
    
    // Create a template map to copy/move
    ParameterMapD template_map = {{"x", 10.0}, {"y", 20.0}, {"z", 30.0}};
    
    // Test copy performance
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ParameterMapD copied_map = template_map;
        auto* x_val = copied_map.find("x");
        if (x_val) {
            double result = *x_val;
            (void)result;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test move performance
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ParameterMapD temp_map = template_map;
        ParameterMapD moved_map = std::move(temp_map);
        auto* x_val = moved_map.find("x");
        if (x_val) {
            double result = *x_val;
            (void)result;
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    auto move_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Copy: " << copy_duration.count() << " microseconds\n";
    std::cout << "Move: " << move_duration.count() << " microseconds\n";
    
    // Move should be faster than copy
    EXPECT_LT(move_duration.count(), copy_duration.count());
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}