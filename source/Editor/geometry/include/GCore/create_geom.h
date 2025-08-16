#pragma once

#include "GCore/api.h"
#include "GOP.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

Geometry GEOMETRY_API create_grid(int resolution, float size);
Geometry GEOMETRY_API create_circle(int resolution, float radius);
Geometry GEOMETRY_API create_cylinder_section(float height, float radius, float angle, int resolution);
Geometry GEOMETRY_API create_spiral(int resolution, float R1, float R2, float circle_count, float height);
Geometry GEOMETRY_API create_uv_sphere(int segments, int rings, float radius);
Geometry GEOMETRY_API create_ico_sphere(int subdivisions, float radius);
Geometry GEOMETRY_API create_point(float x, float y, float z, float size);
Geometry GEOMETRY_API create_wave_mesh(int resolution, float size, float period_count, float wave_height);
Geometry GEOMETRY_API create_diamond(float height, float section_height, float top_width, float section_width, int segments);
Geometry GEOMETRY_API create_trefoil(int resolution, float radius, float tube_radius);
Geometry GEOMETRY_API create_cube(float width, float height, float depth);

USTC_CG_NAMESPACE_CLOSE_SCOPE
