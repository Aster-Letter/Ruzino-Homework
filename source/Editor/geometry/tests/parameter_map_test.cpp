#include <gtest/gtest.h>
#include "../../geometry_nodes/fem_bem/parameter_map.hpp"

using namespace USTC_CG::fem_bem;

TEST(ParameterMapTest, BasicOperations)
{
    ParameterMapD map;
    
    // Test insertion
    map.insert_or_assign("x", 1.0);
    map.insert_or_assign("y", 2.0);
    
    EXPECT_EQ(map.size(), 2);
    EXPECT_FALSE(map.empty());
    
    // Test find
    auto* x_ptr = map.find("x");
    auto* y_ptr = map.find("y");
    auto* z_ptr = map.find("z");
    
    ASSERT_NE(x_ptr, nullptr);
    ASSERT_NE(y_ptr, nullptr);
    EXPECT_EQ(z_ptr, nullptr);
    
    EXPECT_DOUBLE_EQ(*x_ptr, 1.0);
    EXPECT_DOUBLE_EQ(*y_ptr, 2.0);
    
    // Test contains
    EXPECT_TRUE(map.contains("x"));
    EXPECT_TRUE(map.contains("y"));
    EXPECT_FALSE(map.contains("z"));
}

TEST(ParameterMapTest, InitializerListConstructor)
{
    ParameterMapD map = { {"a", 10.0}, {"b", 20.0}, {"c", 30.0} };
    
    EXPECT_EQ(map.size(), 3);
    EXPECT_DOUBLE_EQ(*map.find("a"), 10.0);
    EXPECT_DOUBLE_EQ(*map.find("b"), 20.0);
    EXPECT_DOUBLE_EQ(*map.find("c"), 30.0);
}

TEST(ParameterMapTest, RangeBasedLoop)
{
    ParameterMapD map = { {"x", 1.0}, {"y", 2.0} };
    
    int count = 0;
    for (const auto pair : map) {
        const std::string& name = pair.first;
        const double& value = pair.second;
        
        if (name == "x") {
            EXPECT_DOUBLE_EQ(value, 1.0);
        } else if (name == "y") {
            EXPECT_DOUBLE_EQ(value, 2.0);
        }
        count++;
    }
    
    EXPECT_EQ(count, 2);
}

TEST(ParameterMapTest, UpdateValues)
{
    ParameterMapD map = { {"x", 1.0} };
    
    // Update existing value
    map.insert_or_assign("x", 5.0);
    EXPECT_EQ(map.size(), 1);
    EXPECT_DOUBLE_EQ(*map.find("x"), 5.0);
    
    // Add new value
    map.insert_or_assign("y", 10.0);
    EXPECT_EQ(map.size(), 2);
    EXPECT_DOUBLE_EQ(*map.find("y"), 10.0);
}

TEST(ParameterMapTest, ClearAndEmpty)
{
    ParameterMapD map = { {"x", 1.0}, {"y", 2.0} };
    
    EXPECT_FALSE(map.empty());
    EXPECT_EQ(map.size(), 2);
    
    map.clear();
    
    EXPECT_TRUE(map.empty());
    EXPECT_EQ(map.size(), 0);
    EXPECT_EQ(map.find("x"), nullptr);
    EXPECT_EQ(map.find("y"), nullptr);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}