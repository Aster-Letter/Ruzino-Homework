# c/cpp only
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} THISSRCS)

# interface
add_library(sph_fluid_utils INTERFACE)
target_link_libraries(sph_fluid_utils INTERFACE
    stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle
)
target_include_directories(sph_fluid_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# static lib
if(THISSRCS)
    message(STATUS "[Found SRC for utility lib]")
    add_library(
        sph_fluid_utils_static STATIC
        ${THISSRCS}
    )
    target_include_directories(sph_fluid_utils_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(sph_fluid_utils_static
        stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle
    )

    # link to the interface
    target_link_libraries(
        sph_fluid_utils INTERFACE
        sph_fluid_utils_static
    )
endif()

set(sph_fluid_utils_AVAILABLE TRUE PARENT_SCOPE)