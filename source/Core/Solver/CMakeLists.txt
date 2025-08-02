# Remove the /wd5054 flag that Eigen injects into cuda_eigen_test
if(MSVC)
    get_target_property(_cuda_opts eigen INTERFACE_COMPILE_OPTIONS)
    list(REMOVE_ITEM _cuda_opts "/wd5054")
    set_target_properties(eigen PROPERTIES
        INTERFACE_COMPILE_OPTIONS "${_cuda_opts}"
    )
endif()

USTC_CG_ADD_LIB(
    RZSolver
    SHARED
    WITH_CUDA
    PUBLIC_LIBS RHI Eigen3::Eigen CUDA::cublas CUDA::cusparse
)