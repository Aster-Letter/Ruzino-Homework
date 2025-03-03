add_nodes(
	TARGET_NAME geometry_nodes 
	SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/mesh2SDF
	DEP_LIBS stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle
	COMPILE_DEFS NOMINMAX 
)