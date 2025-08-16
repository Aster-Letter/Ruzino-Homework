#include <gtest/gtest.h>
#include <pxr/base/vt/array.h>

#include <fem_bem/fem_bem.hpp>

#include "GCore/Components/MeshComponent.h"
#include "GCore/create_geom.h"

using namespace USTC_CG::fem_bem;
using namespace USTC_CG;

TEST(FEMBEMProblem, Laplacian)
{
    Geometry circle = create_circle(64, 1);
}
