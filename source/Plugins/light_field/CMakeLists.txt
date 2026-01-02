RUZINO_ADD_LIB(
	light_field
	SHARED
	PUBLIC_LIBS GUI RHI shaders_interface glm::glm GUI spdlog::spdlog
    SKIP_DIRS nodes renderer_nodes
)

add_subdirectory(nodes)
add_subdirectory(renderer_nodes)