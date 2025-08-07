USTC_CG_ADD_LIB(
	geometry 
	SHARED
	PUBLIC_LIBS usd usdVol OpenMeshCore usdGeom usdSkel stage hioOpenVDB spdlog::spdlog usdShade RHI shaders_interface GPUContext nodes_core exprtk
	COMPILE_DEFS
		NOMINMAX 
		USTC_CG_BACKEND_NVRHI
)


target_compile_definitions(geometry PRIVATE GEOM_COMPUTE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/algorithms/CS/")