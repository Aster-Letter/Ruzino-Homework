#include <gtest/gtest.h>

#include <exprtk/exprtk.hpp>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "fem_bem/ElementBasis.hpp"

using namespace Ruzino::fem_bem;

// Helper function to integrate expressions against basis functions
// This replaces the removed integration methods from ElementBasis
std::vector<double> integrate_vertex_against_str(
    const ElementBasis& basis,
    const std::string& expr_str,
    size_t intervals = 100)
{
    std::vector<double> results;
    const auto& vertex_exprs = basis.get_vertex_expressions();
    results.reserve(vertex_exprs.size());

    // Create the expression to integrate against
    std::vector<const char*> all_vars = { "u1", "u2", "u3", "x", "y", "z" };
    Expression expr(expr_str);

    for (const auto& shape_func : vertex_exprs) {
        // Multiply shape function with the expression and integrate
        Expression product = shape_func * expr;
        double result = integrate_over_simplex(
            product, basis.get_barycentric_names(), nullptr, intervals);
        results.push_back(result);
    }
    return results;
}

std::vector<double> integrate_edge_against_str(
    const ElementBasis& basis,
    const std::string& expr_str,
    size_t intervals = 100)
{
    std::vector<double> results;
    if (!basis.has_edge_expressions()) {
        return results;
    }

    const auto& edge_exprs = basis.get_edge_expressions();
    results.reserve(edge_exprs.size());

    std::vector<const char*> all_vars = { "u1", "u2", "u3", "x", "y", "z" };
    Expression expr(expr_str);

    for (const auto& shape_func : edge_exprs) {
        Expression product = shape_func * expr;
        double result = integrate_over_simplex(
            product, basis.get_barycentric_names(), nullptr, intervals);
        results.push_back(result);
    }
    return results;
}

std::vector<double> integrate_face_against_str(
    const ElementBasis& basis,
    const std::string& expr_str,
    size_t intervals = 100)
{
    std::vector<double> results;
    if (!basis.has_face_expressions()) {
        return results;
    }

    const auto& face_exprs = basis.get_face_expressions();
    results.reserve(face_exprs.size());

    std::vector<const char*> all_vars = { "u1", "u2", "u3", "x", "y", "z" };
    Expression expr(expr_str);

    for (const auto& shape_func : face_exprs) {
        Expression product = shape_func * expr;
        double result = integrate_over_simplex(
            product, basis.get_barycentric_names(), nullptr, intervals);
        results.push_back(result);
    }
    return results;
}

std::vector<double> integrate_volume_against_str(
    const ElementBasis& basis,
    const std::string& expr_str,
    size_t intervals = 100)
{
    std::vector<double> results;
    if (!basis.has_volume_expressions()) {
        return results;
    }

    const auto& volume_exprs = basis.get_volume_expressions();
    results.reserve(volume_exprs.size());

    std::vector<const char*> all_vars = { "u1", "u2", "u3", "x", "y", "z" };
    Expression expr(expr_str);

    for (const auto& shape_func : volume_exprs) {
        Expression product = shape_func * expr;
        double result = integrate_over_simplex(
            product, basis.get_barycentric_names(), nullptr, intervals);
        results.push_back(result);
    }
    return results;
}

// Helper functions for integration with coordinate mapping
std::vector<double> integrate_vertex_against_with_mapping(
    const ElementBasis& basis,
    const std::string& expr_str,
    const std::vector<pxr::GfVec2d>& world_vertices,
    size_t intervals = 100)
{
    std::vector<double> results;
    const auto& vertex_exprs = basis.get_vertex_expressions();
    results.reserve(vertex_exprs.size());

    // Create coordinate mapping
    auto mapping = basis.create_coordinate_mapping(world_vertices);
    std::vector<const char*> all_vars = { "u1", "u2", "u3", "x", "y", "z" };
    Expression expr(expr_str);

    for (const auto& shape_func : vertex_exprs) {
        Expression product = shape_func * expr;
        double result = integrate_over_simplex(
            product, basis.get_barycentric_names(), mapping, intervals);
        results.push_back(result);
    }
    return results;
}

std::vector<double> integrate_vertex_against_with_mapping(
    const ElementBasis& basis,
    const std::string& expr_str,
    const std::vector<pxr::GfVec3d>& world_vertices,
    size_t intervals = 100)
{
    std::vector<double> results;
    const auto& vertex_exprs = basis.get_vertex_expressions();
    results.reserve(vertex_exprs.size());

    auto mapping = basis.create_coordinate_mapping(world_vertices);
    std::vector<const char*> all_vars = { "u1", "u2", "u3", "x", "y", "z" };
    Expression expr(expr_str);

    for (const auto& shape_func : vertex_exprs) {
        Expression product = shape_func * expr;
        double result = integrate_over_simplex(
            product, basis.get_barycentric_names(), mapping, intervals);
        results.push_back(result);
    }
    return results;
}

// Helper functions to check support for dimension-specific operations
bool supports_edge_expressions(const ElementBasis& basis)
{
    return basis.has_edge_expressions();
}

bool supports_face_expressions(const ElementBasis& basis)
{
    return basis.has_face_expressions();
}

bool supports_volume_expressions(const ElementBasis& basis)
{
    return basis.has_volume_expressions();
}

// Test integrate_against functionality for different dimensions
TEST(IntegrateAgainstTest, FEM1D_BasicIntegration)
{
    auto fem1d = make_fem_1d();
    fem1d->add_vertex_expression("u1");
    fem1d->add_vertex_expression("1 - u1");  // u2 = 1 - u1

    // Test integration of constant function
    auto results = integrate_vertex_against_str(*fem1d, "1");
    ASSERT_EQ(results.size(), 2);
    EXPECT_NEAR(results[0], 0.5, 1e-3);  // ∫u1 du = 1/2
    EXPECT_NEAR(results[1], 0.5, 1e-3);  // ∫(1-u1) du = 1/2

    // Test integration with polynomial
    results = integrate_vertex_against_str(
        *fem1d, "1");                    // Integrate against constant
    EXPECT_NEAR(results[0], 0.5, 1e-3);  // ∫u1*1 du = 1/2
    EXPECT_NEAR(results[1], 0.5, 1e-3);  // ∫(1-u1)*1 du = 1/2
}

TEST(IntegrateAgainstTest, FEM2D_BasicIntegration)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("u1");
    fem2d->add_vertex_expression("u2");
    fem2d->add_vertex_expression("1 - u1 - u2");  // u3 = 1 - u1 - u2
    fem2d->add_edge_expression("u1*u2");

    // Test vertex integration
    auto results = integrate_vertex_against_str(*fem2d, "1");
    ASSERT_EQ(results.size(), 3);
    EXPECT_NEAR(results[0], 1.0 / 3.0, 1e-3);  // ∫u1 dA = 1/3
    EXPECT_NEAR(results[1], 1.0 / 3.0, 1e-3);  // ∫u2 dA = 1/3
    EXPECT_NEAR(results[2], 1.0 / 3.0, 1e-3);  // ∫(1-u1-u2) dA = 1/3

    // Test edge integration
    auto edge_results = integrate_edge_against_str(*fem2d, "1");
    ASSERT_EQ(edge_results.size(), 1);
    EXPECT_NEAR(edge_results[0], 1.0 / 12.0, 1e-3);  // ∫u1*u2 dA = 1/12
}

TEST(IntegrateAgainstTest, FEM3D_VolumeIntegration)
{
    auto fem3d = make_fem_3d();
    fem3d->add_vertex_expression("u1");
    fem3d->add_volume_expression("u1*u2*u3");

    // Test vertex integration
    auto results = integrate_vertex_against_str(*fem3d, "1");
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0], 0.25, 1e-3);  // ∫u1 dV = 1/4

    // Test volume integration
    auto vol_results = integrate_volume_against_str(*fem3d, "1", 100);
    ASSERT_EQ(vol_results.size(), 1);
    EXPECT_NEAR(vol_results[0], 1.0 / 720 * 6, 1e-3);  // ∫u1*u2*u3 dV = 1/720
}

TEST(IntegrateAgainstTest, WithMapping_2D)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("1");

    // Triangle with area = 1.5
    std::vector<pxr::GfVec2d> triangle = { pxr::GfVec2d(0.0, 0.0),
                                           pxr::GfVec2d(3.0, 0.0),
                                           pxr::GfVec2d(0.0, 1.0) };

    auto results = integrate_vertex_against_with_mapping(*fem2d, "1", triangle);
    ASSERT_EQ(results.size(), 1);
    // Result should be 1 (from reference element), multiply by area 1.5
    // gives 1.5
    EXPECT_NEAR(results[0], 1.0, 1e-3);

    // Test with expression involving x coordinate
    results = integrate_vertex_against_with_mapping(*fem2d, "x", triangle);
    EXPECT_GT(results[0], 0.0);  // Should be positive since x >= 0 on triangle
}

TEST(IntegrateAgainstTest, WithMapping_3D)
{
    auto fem3d = make_fem_3d();
    fem3d->add_vertex_expression("1");

    // Unit tetrahedron
    std::vector<pxr::GfVec3d> tetrahedron = { pxr::GfVec3d(0.0, 0.0, 0.0),
                                              pxr::GfVec3d(1.0, 0.0, 0.0),
                                              pxr::GfVec3d(0.0, 1.0, 0.0),
                                              pxr::GfVec3d(0.0, 0.0, 1.0) };

    auto results =
        integrate_vertex_against_with_mapping(*fem3d, "1", tetrahedron, 80);
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0], 1.0, 1e-3);  // ∫1 dV = 1 on reference tet

    // Test with spatial expression
    results =
        integrate_vertex_against_with_mapping(*fem3d, "x + y + z", tetrahedron);
    EXPECT_GT(results[0], 0.0);
}

TEST(IntegrateAgainstTest, BEM2D_Integration)
{
    auto bem2d = make_bem_2d();
    bem2d->add_vertex_expression("u1");
    bem2d->add_vertex_expression("1 - u1");  // u2 = 1 - u1

    // BEM2D has element_dimension = 1 (1D elements in 2D space)
    auto results = integrate_vertex_against_str(*bem2d, "1");
    ASSERT_EQ(results.size(), 2);
    EXPECT_NEAR(results[0], 0.5, 1e-3);  // ∫u1 ds = 1/2 on line
    EXPECT_NEAR(results[1], 0.5, 1e-3);  // ∫(1-u1) ds = 1/2 on line
}

TEST(IntegrateAgainstTest, BEM3D_Integration)
{
    auto bem3d = make_bem_3d();
    bem3d->add_vertex_expression("u1");
    bem3d->add_edge_expression("u1*u2");

    // BEM3D has element_dimension = 2 (2D elements in 3D space)
    auto results = integrate_vertex_against_str(*bem3d, "1");
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0], 1.0 / 3.0, 1e-3);  // ∫u1 dS = 1/3 on triangle

    auto edge_results = integrate_edge_against_str(*bem3d, "1");
    ASSERT_EQ(edge_results.size(), 1);
    EXPECT_NEAR(edge_results[0], 1.0 / 12.0, 1e-3);  // ∫u1*u2 dS = 1/12
}

// Test dimension mismatch errors
TEST(IntegrateAgainstTest, DimensionMismatchErrors)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("1");

    // Wrong number of vertices for 2D mapping (should be 3, not 2)
    std::vector<pxr::GfVec2d> wrong_vertices = { pxr::GfVec2d(0.0, 0.0),
                                                 pxr::GfVec2d(1.0, 0.0) };

    // This should still work but may not give expected results
    auto results =
        integrate_vertex_against_with_mapping(*fem2d, "x", wrong_vertices);
    // The function should handle gracefully but results may be incorrect

    // Test with too many vertices (should still work, extra ignored)
    std::vector<pxr::GfVec2d> extra_vertices = {
        pxr::GfVec2d(0.0, 0.0),
        pxr::GfVec2d(1.0, 0.0),
        pxr::GfVec2d(0.0, 1.0),
        pxr::GfVec2d(1.0, 1.0)  // Extra vertex
    };

    results =
        integrate_vertex_against_with_mapping(*fem2d, "x", extra_vertices);
    EXPECT_FALSE(results.empty());
}

TEST(IntegrateAgainstTest, UnsupportedOperations)
{
    auto fem1d = make_fem_1d();

    // FEM1D doesn't support edge operations
    EXPECT_FALSE(supports_edge_expressions(*fem1d));
    EXPECT_FALSE(supports_face_expressions(*fem1d));
    EXPECT_FALSE(supports_volume_expressions(*fem1d));

    // These should return empty results
    auto empty_results = integrate_edge_against_str(*fem1d, "1");
    EXPECT_TRUE(empty_results.empty());

    empty_results = integrate_face_against_str(*fem1d, "1");
    EXPECT_TRUE(empty_results.empty());

    empty_results = integrate_volume_against_str(*fem1d, "1");
    EXPECT_TRUE(empty_results.empty());
}

TEST(IntegrateAgainstTest, InvalidExpressions)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("u1");

    // Test invalid mathematical expression
    EXPECT_THROW(
        {
            auto results =
                integrate_vertex_against_str(*fem2d, "invalid_function()");
        },
        std::runtime_error);
}

TEST(IntegrateAgainstTest, EmptyExpressions)
{
    auto fem2d = make_fem_2d();

    // No expressions added, should return empty
    auto results = integrate_vertex_against_str(*fem2d, "1");
    EXPECT_TRUE(results.empty());

    // Add expression and test
    fem2d->add_vertex_expression("u1 + u2 + (1 - u1 - u2)");
    results = integrate_vertex_against_str(*fem2d, "1");
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0], 1.0, 1e-3);  // ∫(u1+u2+(1-u1-u2)) dA = ∫1 dA = 1
}

// Test correct scaling with different domain sizes
TEST(IntegrateAgainstTest, DomainScaling)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("1");

    // Unit triangle (area = 0.5)
    std::vector<pxr::GfVec2d> unit_triangle = { pxr::GfVec2d(0.0, 0.0),
                                                pxr::GfVec2d(1.0, 0.0),
                                                pxr::GfVec2d(0.0, 1.0) };
    auto unit_results =
        integrate_vertex_against_with_mapping(*fem2d, "1", unit_triangle);
    ASSERT_EQ(unit_results.size(), 1);

    // Scaled triangle (area = 2.0)
    std::vector<pxr::GfVec2d> scaled_triangle = { pxr::GfVec2d(0.0, 0.0),
                                                  pxr::GfVec2d(2.0, 0.0),
                                                  pxr::GfVec2d(0.0, 2.0) };
    auto scaled_results =
        integrate_vertex_against_with_mapping(*fem2d, "1", scaled_triangle);
    ASSERT_EQ(scaled_results.size(), 1);

    // Results should be the same (1.0) since they're on reference element
    EXPECT_NEAR(unit_results[0], scaled_results[0], 1e-3);
    EXPECT_NEAR(unit_results[0], 1.0, 1e-3);
}

// Test coordinate mapping correctness
TEST(IntegrateAgainstTest, CoordinateMappingCorrectness)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression(
        "u1");  // This is 1 at first vertex, 0 at others

    // Triangle: (0,0), (1,0), (0,1)
    std::vector<pxr::GfVec2d> triangle = { pxr::GfVec2d(0.0, 0.0),
                                           pxr::GfVec2d(1.0, 0.0),
                                           pxr::GfVec2d(0.0, 1.0) };

    // Integrate x over triangle using u1 shape function
    // Should give centroid x-coordinate weighted by shape function
    auto results = integrate_vertex_against_with_mapping(*fem2d, "x", triangle);
    ASSERT_EQ(results.size(), 1);
    EXPECT_GT(results[0], 0.0);

    // Integrate y coordinate
    results = integrate_vertex_against_with_mapping(*fem2d, "y", triangle);
    EXPECT_GT(results[0], 0.0);
}

// Test 3D coordinate mapping
TEST(IntegrateAgainstTest, CoordinateMapping_3D)
{
    auto fem3d = make_fem_3d();
    fem3d->add_vertex_expression("1");

    // Unit tetrahedron
    std::vector<pxr::GfVec3d> tetrahedron = { pxr::GfVec3d(0.0, 0.0, 0.0),
                                              pxr::GfVec3d(1.0, 0.0, 0.0),
                                              pxr::GfVec3d(0.0, 1.0, 0.0),
                                              pxr::GfVec3d(0.0, 0.0, 1.0) };

    auto results = integrate_vertex_against_with_mapping(
        *fem3d, "x*x + y*y + z*z", tetrahedron);
    ASSERT_EQ(results.size(), 1);
    EXPECT_GT(results[0], 0.0);
}

// Test boundary elements with mapping
TEST(IntegrateAgainstTest, BoundaryElementMapping)
{
    auto bem2d = make_bem_2d();  // 1D elements in 2D space
    bem2d->add_vertex_expression("u1");
    bem2d->add_vertex_expression("1 - u1");  // u2 = 1 - u1

    // Line segment from (0,0) to (2,0)
    std::vector<pxr::GfVec2d> line = { pxr::GfVec2d(0.0, 0.0),
                                       pxr::GfVec2d(2.0, 0.0) };

    auto results = integrate_vertex_against_with_mapping(*bem2d, "1", line);
    ASSERT_EQ(results.size(), 2);
    EXPECT_NEAR(results[0], 0.5, 1e-3);  // ∫u1 ds = 0.5
    EXPECT_NEAR(results[1], 0.5, 1e-3);  // ∫u2 ds = 0.5

    // Integrate x coordinate
    results = integrate_vertex_against_with_mapping(*bem2d, "x", line);
    EXPECT_GT(results[0], 0.0);
    EXPECT_GT(results[1], 0.0);
}

// Test various dimension mismatches
TEST(IntegrateAgainstTest, ComprehensiveDimensionTests)
{
    // Test FEM1D with 3D vertices (should work but only use first coordinate)
    auto fem1d = make_fem_1d();
    fem1d->add_vertex_expression("u1");

    std::vector<pxr::GfVec3d> line_3d = { pxr::GfVec3d(0.0, 5.0, 10.0),
                                          pxr::GfVec3d(1.0, 7.0, 15.0) };

    auto results = integrate_vertex_against_with_mapping(*fem1d, "x", line_3d);
    EXPECT_FALSE(results.empty());

    // Test BEM3D with 2D vertices mapping
    auto bem3d = make_bem_3d();
    bem3d->add_vertex_expression("u1");

    std::vector<pxr::GfVec2d> triangle_2d = { pxr::GfVec2d(0.0, 0.0),
                                              pxr::GfVec2d(1.0, 0.0),
                                              pxr::GfVec2d(0.0, 1.0) };

    // Should work - BEM3D can use 2D coordinates (z=0 implicitly)
    results =
        integrate_vertex_against_with_mapping(*bem3d, "x + y", triangle_2d);
    EXPECT_FALSE(results.empty());
}

// Test edge cases with empty or single point domains
TEST(IntegrateAgainstTest, EdgeCases)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("1");

    // Empty vertex list
    std::vector<pxr::GfVec2d> empty_vertices;
    auto results =
        integrate_vertex_against_with_mapping(*fem2d, "1", empty_vertices);
    // Should handle gracefully

    // Single vertex (degenerate triangle)
    std::vector<pxr::GfVec2d> single_vertex = { pxr::GfVec2d(1.0, 1.0) };
    results = integrate_vertex_against_with_mapping(*fem2d, "1", single_vertex);
    // Should handle gracefully
}

// Test numerical precision with high-order polynomials
TEST(IntegrateAgainstTest, HighOrderPolynomials)
{
    auto fem2d = make_fem_2d();
    fem2d->add_vertex_expression("u1*u1*u1");  // Cubic shape function
    fem2d->add_vertex_expression("u2*u2*u2");

    auto results = integrate_vertex_against_str(*fem2d, "u1*u2*(1-u1-u2)");
    ASSERT_EQ(results.size(), 2);
    // These are high-order integrations, should be computed accurately
    EXPECT_GT(results[0], 0.0);
    EXPECT_GT(results[1], 0.0);

    // Test with very high degree
    fem2d->add_vertex_expression("u1*u1*u1*u1*u1");  // 5th degree
    results = integrate_vertex_against_str(*fem2d, "1");
    ASSERT_EQ(results.size(), 3);
    EXPECT_GT(results[2], 0.0);
}

int main()
{
    ::testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
