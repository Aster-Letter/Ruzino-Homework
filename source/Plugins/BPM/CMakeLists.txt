RUZINO_ADD_LIB(
	BPM
	SHARED
	WITH_CUDA
	SKIP_DIRS nodes renderer_nodes RHI Eigen3::Eigen
	PYTHON_WRAP_DIR python
)

add_subdirectory(nodes)
