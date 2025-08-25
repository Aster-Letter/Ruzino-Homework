USTC_CG_ADD_LIB(
	geometry 
	SHARED
	PUBLIC_LIBS OpenMeshCore spdlog::spdlog shaders_interface nodes_core igl_core triangle::triangle
	COMPILE_DEFS
		NOMINMAX 
		USTC_CG_BACKEND_NVRHI
)


target_compile_definitions(geometry PRIVATE GEOM_COMPUTE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/algorithms/CS/")