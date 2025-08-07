add_nodes(
	TARGET_NAME geometry_nodes 
	SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/mesh2SDF ${CMAKE_CURRENT_SOURCE_DIR}/fem_bem
	DEP_LIBS stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle exprtk
	COMPILE_DEFS NOMINMAX 
)

add_library(tetgen STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tetgen/tetgen.cxx ${CMAKE_CURRENT_SOURCE_DIR}/tetgen/tetgen.h ${CMAKE_CURRENT_SOURCE_DIR}/tetgen/predicates.cxx)
set_target_properties(tetgen PROPERTIES ${OUTPUT_DIR})
target_link_libraries(node_tetgen PRIVATE tetgen)
target_include_directories(node_tetgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tetgen)