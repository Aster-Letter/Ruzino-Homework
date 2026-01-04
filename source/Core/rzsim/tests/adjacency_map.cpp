#include <GCore/Components/MeshComponent.h>
#include <GCore/GOP.h>
#include <gtest/gtest.h>
#include <rzsim/rzsim.h>

// Forward declarations for CUDA initialization
namespace Ruzino {
namespace cuda {
    extern int cuda_init();
    extern int cuda_shutdown();
}  // namespace cuda
}  // namespace Ruzino
#include <iostream>
#include <unordered_set>

using namespace Ruzino;

// Helper function to verify offset buffer validity
void verify_offset_buffer(
    const std::vector<unsigned>& adjacency_data,
    const std::vector<unsigned>& offset_buffer,
    size_t expected_vertex_count)
{
    // Verify offset buffer size
    ASSERT_EQ(offset_buffer.size(), expected_vertex_count);

    // Verify offsets are monotonically increasing
    for (size_t i = 0; i < offset_buffer.size(); i++) {
        EXPECT_LT(offset_buffer[i], adjacency_data.size())
            << "Offset for vertex " << i << " is out of bounds";
        if (i > 0) {
            EXPECT_LE(offset_buffer[i - 1], offset_buffer[i])
                << "Offsets are not monotonically increasing at vertex " << i;
        }
    }

    // Verify each vertex has valid neighbor count and data
    for (size_t i = 0; i < offset_buffer.size(); i++) {
        unsigned offset = offset_buffer[i];
        unsigned count = adjacency_data[offset];
        EXPECT_GT(count, 0) << "Vertex " << i << " has 0 neighbors";
        EXPECT_LT(offset + count, adjacency_data.size())
            << "Vertex " << i << " neighbors exceed buffer bounds";
    }
}

// Verify that every vertex has the expected unique neighbor set
void expect_neighbors(
    const std::vector<unsigned>& adjacency_data,
    const std::vector<unsigned>& offset_buffer,
    const std::vector<std::vector<unsigned>>& expected)
{
    ASSERT_EQ(offset_buffer.size(), expected.size());

    for (size_t v = 0; v < expected.size(); ++v) {
        unsigned offset = offset_buffer[v];
        unsigned count = adjacency_data[offset];

        ASSERT_EQ(count, expected[v].size())
            << "Vertex " << v << " neighbor count mismatch";

        std::unordered_set<unsigned> seen;
        for (unsigned i = 0; i < count; ++i) {
            unsigned n = adjacency_data[offset + 1 + i];
            ASSERT_TRUE(seen.insert(n).second)
                << "Duplicate neighbor " << n << " at vertex " << v;
        }

        for (unsigned n : expected[v]) {
            ASSERT_TRUE(seen.count(n))
                << "Missing neighbor " << n << " at vertex " << v;
        }
    }
}

TEST(AdjacencyMap, SimpleTriangle)
{
    // Create a simple triangle mesh
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Triangle vertices: 0, 1, 2
    std::vector<glm::vec3> vertices = { glm::vec3(0.0f, 0.0f, 0.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f),
                                        glm::vec3(0.0f, 1.0f, 0.0f) };

    std::vector<int> faceVertexCounts = { 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    // Test GPU version - now returns tuple
    auto [adjacencyGPU, offsetGPU] = get_adjcency_map_gpu(mesh);
    ASSERT_NE(adjacencyGPU, nullptr);
    ASSERT_NE(offsetGPU, nullptr);

    // Test CPU version - now returns tuple
    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 3);  // 3 vertices

    std::cout << "Adjacency map for triangle:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer and unique neighbors
    verify_offset_buffer(adjacencyCPU, offsetCPU, 3);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 2 },
            { 0, 2 },
            { 1, 0 },
        });
}

TEST(AdjacencyMap, Quad)
{
    // Create a quad mesh
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Quad vertices: 0--1
    //                |  |
    //                3--2
    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 1.0f, 0.0f),  // 0
        glm::vec3(1.0f, 1.0f, 0.0f),  // 1
        glm::vec3(1.0f, 0.0f, 0.0f),  // 2
        glm::vec3(0.0f, 0.0f, 0.0f)   // 3
    };

    std::vector<int> faceVertexCounts = { 4 };
    std::vector<int> faceVertexIndices = { 0, 1, 2, 3 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 4);  // 4 vertices

    std::cout << "Adjacency map for quad:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer and unique neighbors
    verify_offset_buffer(adjacencyCPU, offsetCPU, 4);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 3 },
            { 0, 2 },
            { 1, 3 },
            { 2, 0 },
        });
}

TEST(AdjacencyMap, TwoTriangles)
{
    // Create two triangles sharing an edge
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    // Vertices:  0
    //           /|\
    //          / | \
    //         1--2--3
    // Faces: [0,1,2] and [0,2,3]
    std::vector<glm::vec3> vertices = {
        glm::vec3(0.5f, 1.0f, 0.0f),  // 0
        glm::vec3(0.0f, 0.0f, 0.0f),  // 1
        glm::vec3(0.5f, 0.0f, 0.0f),  // 2
        glm::vec3(1.0f, 0.0f, 0.0f)   // 3
    };

    std::vector<int> faceVertexCounts = { 3, 3 };
    std::vector<int> faceVertexIndices = { 0, 1, 2, 0, 2, 3 };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 4);  // 4 vertices

    std::cout << "Adjacency map for two triangles:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer and unique neighbors
    verify_offset_buffer(adjacencyCPU, offsetCPU, 4);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 2, 3 },
            { 0, 2 },
            { 0, 1, 3 },
            { 0, 2 },
        });
}

// Complex test: Pyramid (4 triangular faces + 1 quad base)
TEST(AdjacencyMap, Pyramid)
{
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 1.0f, 0.0f),  // 0
        glm::vec3(1.0f, 1.0f, 0.0f),  // 1
        glm::vec3(1.0f, 0.0f, 0.0f),  // 2
        glm::vec3(0.0f, 0.0f, 0.0f),  // 3
        glm::vec3(0.5f, 0.5f, 1.0f)   // 4 (apex)
    };

    std::vector<int> faceVertexCounts = { 3, 3, 3, 3, 4 };
    std::vector<int> faceVertexIndices = {
        0, 4, 1,    // Face 0: front triangle
        1, 4, 2,    // Face 1: right triangle
        2, 4, 3,    // Face 2: back triangle
        3, 4, 0,    // Face 3: left triangle
        0, 1, 2, 3  // Face 4: quad base
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 5);  // 5 vertices

    std::cout << "Adjacency map for pyramid:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer
    verify_offset_buffer(adjacencyCPU, offsetCPU, 5);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 4, 1, 3 },
            { 0, 4, 2 },
            { 1, 4, 3 },
            { 2, 4, 0 },
            { 0, 1, 2, 3 },
        });
}

// Complex test: Two separate quads (non-connected mesh)
TEST(AdjacencyMap, TwoSeparateQuads)
{
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = { // First quad
                                        glm::vec3(0.0f, 1.0f, 0.0f),
                                        glm::vec3(1.0f, 1.0f, 0.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f),
                                        glm::vec3(0.0f, 0.0f, 0.0f),
                                        // Second quad
                                        glm::vec3(2.0f, 1.0f, 0.0f),
                                        glm::vec3(3.0f, 1.0f, 0.0f),
                                        glm::vec3(3.0f, 0.0f, 0.0f),
                                        glm::vec3(2.0f, 0.0f, 0.0f)
    };

    std::vector<int> faceVertexCounts = { 4, 4 };
    std::vector<int> faceVertexIndices = {
        0, 1, 2, 3,  // First quad
        4, 5, 6, 7   // Second quad
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 8);  // 8 vertices

    std::cout << "Adjacency map for two separate quads:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer
    verify_offset_buffer(adjacencyCPU, offsetCPU, 8);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 3 },
            { 0, 2 },
            { 1, 3 },
            { 2, 0 },
            { 5, 7 },
            { 4, 6 },
            { 5, 7 },
            { 6, 4 },
        });
}

// Complex test: Hexagon (6 triangles sharing a central vertex)
TEST(AdjacencyMap, Hexagon)
{
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.5f, 0.5f, 0.0f),    // 0 (center)
        glm::vec3(1.0f, 0.5f, 0.0f),    // 1
        glm::vec3(0.75f, 1.0f, 0.0f),   // 2
        glm::vec3(0.0f, 1.0f, 0.0f),    // 3
        glm::vec3(-0.25f, 0.5f, 0.0f),  // 4
        glm::vec3(0.0f, 0.0f, 0.0f),    // 5
        glm::vec3(0.75f, 0.0f, 0.0f)    // 6
    };

    std::vector<int> faceVertexCounts = { 3, 3, 3, 3, 3, 3 };
    std::vector<int> faceVertexIndices = {
        0, 1, 2,  // Triangle 0
        0, 2, 3,  // Triangle 1
        0, 3, 4,  // Triangle 2
        0, 4, 5,  // Triangle 3
        0, 5, 6,  // Triangle 4
        0, 6, 1   // Triangle 5
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 7);  // 7 vertices

    std::cout << "Adjacency map for hexagon:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer
    verify_offset_buffer(adjacencyCPU, offsetCPU, 7);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 2, 3, 4, 5, 6 },
            { 0, 2, 6 },
            { 0, 1, 3 },
            { 0, 2, 4 },
            { 0, 3, 5 },
            { 0, 4, 6 },
            { 0, 5, 1 },
        });
}

// Complex test: Cube (6 quads)
TEST(AdjacencyMap, CubeQuads)
{
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f),  // 0
        glm::vec3(1.0f, 0.0f, 0.0f),  // 1
        glm::vec3(1.0f, 1.0f, 0.0f),  // 2
        glm::vec3(0.0f, 1.0f, 0.0f),  // 3
        glm::vec3(0.0f, 0.0f, 1.0f),  // 4
        glm::vec3(1.0f, 0.0f, 1.0f),  // 5
        glm::vec3(1.0f, 1.0f, 1.0f),  // 6
        glm::vec3(0.0f, 1.0f, 1.0f)   // 7
    };

    std::vector<int> faceVertexCounts = { 4, 4, 4, 4, 4, 4 };
    std::vector<int> faceVertexIndices = {
        0, 1, 2, 3,  // Bottom
        4, 7, 6, 5,  // Top
        0, 4, 5, 1,  // Front
        2, 6, 7, 3,  // Back
        0, 3, 7, 4,  // Left
        1, 5, 6, 2   // Right
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 8);  // 8 vertices

    std::cout << "Adjacency map for cube (quads):\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer
    verify_offset_buffer(adjacencyCPU, offsetCPU, 8);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 3, 4 },
            { 0, 2, 5 },
            { 1, 3, 6 },
            { 0, 2, 7 },
            { 5, 7, 0 },
            { 4, 1, 6 },
            { 2, 7, 5 },
            { 4, 6, 3 },
        });
}

// Complex test: Pentagonal pyramid
TEST(AdjacencyMap, PentagonalPyramid)
{
    Geometry mesh = Geometry::CreateMesh();
    auto meshComp = mesh.get_component<MeshComponent>();

    std::vector<glm::vec3> vertices = {
        glm::vec3(0.3f, 0.3f, 1.0f),      // 0 (apex)
        glm::vec3(1.0f, 0.0f, 0.0f),      // 1
        glm::vec3(0.31f, 0.95f, 0.0f),    // 2
        glm::vec3(-0.81f, 0.59f, 0.0f),   // 3
        glm::vec3(-0.81f, -0.59f, 0.0f),  // 4
        glm::vec3(0.31f, -0.95f, 0.0f)    // 5
    };

    std::vector<int> faceVertexCounts = { 3, 3, 3, 3, 3, 5 };
    std::vector<int> faceVertexIndices = {
        0, 1, 2,       // Triangle 0
        0, 2, 3,       // Triangle 1
        0, 3, 4,       // Triangle 2
        0, 4, 5,       // Triangle 3
        0, 5, 1,       // Triangle 4
        1, 2, 3, 4, 5  // Pentagon base
    };

    meshComp->set_vertices(vertices);
    meshComp->set_face_vertex_counts(faceVertexCounts);
    meshComp->set_face_vertex_indices(faceVertexIndices);

    auto [adjacencyCPU, offsetCPU] = get_adjcency_map(mesh);
    ASSERT_GT(adjacencyCPU.size(), 0);
    ASSERT_EQ(offsetCPU.size(), 6);  // 6 vertices

    std::cout << "Adjacency map for pentagonal pyramid:\n";
    for (size_t i = 0; i < adjacencyCPU.size(); i++) {
        std::cout << adjacencyCPU[i] << " ";
    }
    std::cout << "\nOffset buffer: ";
    for (size_t i = 0; i < offsetCPU.size(); i++) {
        std::cout << "V" << i << "@" << offsetCPU[i] << " ";
    }
    std::cout << std::endl;

    // Verify offset buffer
    verify_offset_buffer(adjacencyCPU, offsetCPU, 6);
    expect_neighbors(
        adjacencyCPU,
        offsetCPU,
        {
            { 1, 2, 3, 4, 5 },
            { 0, 2, 5 },
            { 0, 1, 3 },
            { 0, 2, 4 },
            { 0, 3, 5 },
            { 0, 4, 1 },
        });
}

int main(int argc, char** argv)
{
    // Initialize CUDA
    Ruzino::cuda::cuda_init();

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    // Cleanup
    Ruzino::cuda::cuda_shutdown();

    return result;
}
