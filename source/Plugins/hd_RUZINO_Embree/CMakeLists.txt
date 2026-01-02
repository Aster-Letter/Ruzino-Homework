option (ENABLE_EMBREE_BASED_RENDERER "Enable Embree based renderer" OFF)

if (ENABLE_EMBREE_BASED_RENDERER)

	set(TBB_ROOT ${PROJECT_SOURCE_DIR}/SDK/OpenUSD/${SDK_FOLDER})
	add_subdirectory(embree4)
	set_target_properties(embree PROPERTIES ${OUTPUT_DIR})
#
	RUZINO_ADD_LIB(
		hd_RUZINO_Embree
		SHARED
		PUBLIC_LIBS spdlog::spdlog
		PRIVATE_LIBS usd RHI hdx nodes_system usdMtlx  usdImaging embree
		COMPILE_DEFS
			NOMINMAX
		USD_RESOURCE_DIRS resources
		SKIP_DIRS embree4
		RESOURCE_COPY_TARGET ${OUT_BINARY_DIR}/usd/hd_RUZINO_Embree/resources
	)

endif()