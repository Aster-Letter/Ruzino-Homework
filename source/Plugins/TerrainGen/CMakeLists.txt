RUZINO_ADD_LIB(
	TerrainGen
	SHARED
	SKIP_DIRS geometry_nodes
)

target_link_libraries(TerrainGen PUBLIC glm::glm)

target_include_directories(TerrainGen PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(geometry_nodes)
