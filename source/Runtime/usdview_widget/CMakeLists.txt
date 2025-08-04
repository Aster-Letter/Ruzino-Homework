if(USTC_CG_WITH_OPENUSD)
	set(PRIVATE_LIBS spdlog::spdlog usd usdImaging usdImagingGL hdSt glf garch hgi RHI)
	set(COMPILE_DEFS NOMINMAX=1 USTC_CG_BUILD_MODULE=1)

set(PRIVATE_LIBS spdlog::spdlog GUI usd usdImaging usdImagingGL hdSt glf garch hgi RHI)
set(COMPILE_DEFS NOMINMAX=1 USTC_CG_BUILD_MODULE=1)

USTC_CG_ADD_LIB(
	usdview_widget
	SHARED
	PUBLIC_LIBS stage geometry
	PRIVATE_LIBS ${PRIVATE_LIBS}
	COMPILE_DEFS ${COMPILE_DEFS}
)
endif()