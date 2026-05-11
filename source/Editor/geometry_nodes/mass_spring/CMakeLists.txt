# c/cpp only
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} THISSRCS)

# interface
add_library(mass_spring_utils INTERFACE)
target_link_libraries(mass_spring_utils INTERFACE
    stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle
)
target_include_directories(mass_spring_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# static lib
if(THISSRCS)
    message(STATUS "[Found SRC for utility lib]")
    add_library(
        mass_spring_utils_static STATIC
        ${THISSRCS}
    )
    target_include_directories(mass_spring_utils_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(mass_spring_utils_static
        stage nodes_system usd geometry usdShade Eigen3::Eigen autodiff igl::core igl_restricted::triangle
    )

    # link to the interface
    target_link_libraries(
        mass_spring_utils INTERFACE
        mass_spring_utils_static
    )
endif()

set(mass_spring_utils_AVAILABLE TRUE PARENT_SCOPE)