RUZINO_ADD_LIB(
	rzsim
	SHARED
	WITH_CUDA
	PUBLIC_LIBS RHI Eigen3::Eigen
	SKIP_DIRS nodes renderer_nodes geometry_nodes
)

add_subdirectory(nodes)
add_subdirectory(renderer_nodes)
add_subdirectory(geometry_nodes)
