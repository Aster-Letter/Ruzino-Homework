#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include "glm/glm.hpp"

#include "pxr/usd/usdGeom/xform.h"

namespace USTC_CG::mass_spring {

inline auto flatten = [&](const Eigen::MatrixXd& A) {
    Eigen::MatrixXd A_flatten = A.transpose();
    A_flatten.resize(A.rows() * A.cols(), 1);
    return A_flatten;
};
inline auto unflatten = [&](const Eigen::MatrixXd& A_flatten) {
    Eigen::MatrixXd A = A_flatten;
    A.resize(3, A_flatten.rows() / 3);
    A.transposeInPlace();
    return A;
};

inline Eigen::MatrixXi usd_faces_to_eigen(
    const std::vector<int>& faceVertexCount,
    const std::vector<int>& faceVertexIndices)
{
    int n_triangles = 0;
    for (int count : faceVertexCount) {
        if (count >= 3) {
            n_triangles += count - 2;
        }
    }

    Eigen::MatrixXi F(n_triangles, 3);
    int triangle_id = 0;
    int index_offset = 0;
    for (int count : faceVertexCount) {
        if (count < 3 ||
            index_offset + count >
                static_cast<int>(faceVertexIndices.size())) {
            index_offset += std::max(count, 0);
            continue;
        }

        // USD meshes may contain triangles, quads, or general polygons. The
        // mass-spring graph works on triangle edges, so polygonal faces are
        // converted with a fan triangulation around the first corner. Keeping
        // this conversion here avoids desynchronizing faceVertexIndices when a
        // textured quad mesh is used as input.
        const int first = faceVertexIndices[index_offset];
        for (int local = 1; local + 1 < count; ++local) {
            F(triangle_id, 0) = first;
            F(triangle_id, 1) = faceVertexIndices[index_offset + local];
            F(triangle_id, 2) = faceVertexIndices[index_offset + local + 1];
            ++triangle_id;
        }
        index_offset += count;
    }
    if (triangle_id != n_triangles) {
        F.conservativeResize(triangle_id, 3);
    }
    return F;
}

inline Eigen::MatrixXd usd_vertices_to_eigen(
    const std::vector<glm::vec3>& v)
{
    unsigned nVertices = v.size();
    Eigen::MatrixXd V(nVertices, 3);
    for (int i = 0; i < nVertices; i++) {
        for (int j = 0; j < 3; j++) {
            V(i, j) = v[i][j];
        }
    }
    return V;
}

inline std::vector<glm::vec3> eigen_to_usd_vertices(
    const Eigen::MatrixXd& V)
{
    std::vector<glm::vec3> vertices;
    for (int i = 0; i < V.rows(); i++) {
        vertices.push_back(glm::vec3(V(i, 0), V(i, 1), V(i, 2)));
    }
    return vertices;
}

using Edge = std::pair<int, int>;
using EdgeSet = std::set<Edge>;
using EdgeWeightMap = std::map<Edge, double>;

inline Edge make_edge(int v0, int v1)
{
    if (v0 > v1) {
        std::swap(v0, v1);
    }
    return std::make_pair(v0, v1);
}

// Here F is of shape [nFaces, 3] for triangular mesh
inline EdgeSet get_edges(const Eigen::MatrixXi& F)
{
    EdgeSet edges;
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < F.cols(); j++) {
            edges.insert(make_edge(F(i, j), F(i, (j + 1) % F.cols())));
        }
    }
    return edges;
}

inline double median_edge_length(const EdgeSet& edges, const Eigen::MatrixXd& V)
{
    std::vector<double> lengths;
    lengths.reserve(edges.size());
    for (const auto& edge : edges) {
        const double length = (V.row(edge.first) - V.row(edge.second)).norm();
        if (length > 1e-12) {
            lengths.push_back(length);
        }
    }
    if (lengths.empty()) {
        return 0.0;
    }

    const auto median_iter = lengths.begin() + lengths.size() / 2;
    std::nth_element(lengths.begin(), median_iter, lengths.end());
    return *median_iter;
}

inline EdgeSet add_balanced_shear_edges(
    const EdgeSet& base_edges,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& V)
{
    EdgeSet edges = base_edges;
    const double base_length = median_edge_length(base_edges, V);
    if (base_length <= 0.0) {
        return edges;
    }

    // The homework grid is triangulated with the same diagonal direction in
    // every quad. If we only use mesh edges as springs, that single diagonal
    // direction becomes much stiffer than the other shear direction and the
    // cloth tends to fold into nonphysical diagonal panels. For an interior
    // diagonal shared by two triangles, the two opposite vertices form the
    // missing diagonal of the quad. Adding it balances the shear response
    // without changing the input mesh connectivity used for rendering.
    std::map<Edge, std::vector<int>> opposite_vertices;
    for (int i = 0; i < F.rows(); ++i) {
        const int a = F(i, 0);
        const int b = F(i, 1);
        const int c = F(i, 2);
        opposite_vertices[make_edge(a, b)].push_back(c);
        opposite_vertices[make_edge(b, c)].push_back(a);
        opposite_vertices[make_edge(c, a)].push_back(b);
    }

    constexpr double kDiagonalThreshold = 1.2;
    for (const auto& item : opposite_vertices) {
        const auto& shared_edge = item.first;
        const auto& opposites = item.second;
        if (opposites.size() != 2) {
            continue;
        }

        const double shared_length =
            (V.row(shared_edge.first) - V.row(shared_edge.second)).norm();
        if (shared_length < kDiagonalThreshold * base_length) {
            continue;
        }

        edges.insert(make_edge(opposites[0], opposites[1]));
    }
    return edges;
}

inline EdgeSet add_bending_edges(
    const EdgeSet& base_edges,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& V,
    double stiffness_scale,
    EdgeWeightMap& edge_stiffness_scales)
{
    EdgeSet edges = base_edges;
    const double base_length = median_edge_length(base_edges, V);
    if (base_length <= 0.0 || stiffness_scale <= 0.0) {
        return edges;
    }

    // A hinge is an interior mesh edge shared by two triangles. Connecting the
    // two vertices opposite that hinge adds a weak long-range spring. It does
    // not model true dihedral-angle bending, but it supplies a cheap bending
    // resistance that prevents the cloth from folding into zero-cost rigid
    // panels along grid lines.
    std::map<Edge, std::vector<int>> opposite_vertices;
    for (int i = 0; i < F.rows(); ++i) {
        const int a = F(i, 0);
        const int b = F(i, 1);
        const int c = F(i, 2);
        opposite_vertices[make_edge(a, b)].push_back(c);
        opposite_vertices[make_edge(b, c)].push_back(a);
        opposite_vertices[make_edge(c, a)].push_back(b);
    }

    constexpr double kStructuralEdgeThreshold = 1.2;
    for (const auto& item : opposite_vertices) {
        const auto& shared_edge = item.first;
        const auto& opposites = item.second;
        if (opposites.size() != 2) {
            continue;
        }

        const double shared_length =
            (V.row(shared_edge.first) - V.row(shared_edge.second)).norm();
        if (shared_length > kStructuralEdgeThreshold * base_length) {
            continue;
        }

        const Edge bending_edge = make_edge(opposites[0], opposites[1]);
        if (base_edges.find(bending_edge) == base_edges.end()) {
            edges.insert(bending_edge);
            edge_stiffness_scales[bending_edge] = stiffness_scale;
        }
    }
    return edges;
}

inline std::vector<bool> VtIntArray_to_vector_bool(const pxr::VtArray<float>& v)
{
    std::vector<bool> mask;
    for (int i = 0; i < v.size(); i++) {
        mask.push_back(v[i] > 0);
    }
    return mask;
}

}  // namespace USTC_CG::mass_spring
