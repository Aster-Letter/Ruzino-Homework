USTC_CG_ADD_LIB(
	stage
	SHARED
	PUBLIC_LIBS
		usd
		usdImaging
		usdShade
	PYTHON_WRAP_DIR python
	PRIVATE_LIBS 
		geometry
		spdlog::spdlog
		nodes_system
		MaterialXCore
		MaterialXFormat
		usdLux
	COMPILE_DEFS
		NOMINMAX=1
		USTC_CG_BUILD_MODULE=1
)

# Link Boost.Python for USD interoperability
# usd_python contains Boost.Python bindings used by pxr module
target_link_libraries(stage PRIVATE usd_python)

# Link usd_python and geometry for stage_py (needs Boost.Python for interop + GeomPayload)
target_link_libraries(stage_py PRIVATE usd_python geometry nodes_system)

# Add usd_python library directory
set(USD_PYTHON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../SDK/OpenUSD/Debug/lib)
target_link_directories(stage PRIVATE ${USD_PYTHON_LIB_DIR})
target_link_directories(stage_py PRIVATE ${USD_PYTHON_LIB_DIR})

# Add Boost include path for BOTH stage and stage_py
# The Python wrapper needs boost/python.hpp for true interop
set(BOOST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../SDK/OpenUSD/Debug/include/boost-1_86)
target_include_directories(stage PRIVATE ${BOOST_INCLUDE_DIR})
target_include_directories(stage_py PRIVATE ${BOOST_INCLUDE_DIR})

# CRITICAL: Define NOMINMAX for stage_py to prevent Windows min/max macro issues
# USD's robin_growth_policy.h uses std::numeric_limits<>::max() which conflicts with Windows macros
target_compile_definitions(stage_py PRIVATE NOMINMAX)