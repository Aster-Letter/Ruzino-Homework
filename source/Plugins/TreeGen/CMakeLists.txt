RUZINO_ADD_LIB(
	TreeGen
	SHARED
	SKIP_DIRS nodes geometry_nodes
)

target_link_libraries(TreeGen PUBLIC glm::glm)

target_include_directories(TreeGen PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(nodes)
add_subdirectory(geometry_nodes)
