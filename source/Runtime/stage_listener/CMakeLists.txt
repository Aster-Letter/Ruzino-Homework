USTC_CG_ADD_LIB(
    stage_listener
    SHARED
    PUBLIC_LIBS stage usd usdGeom
    PRIVATE_LIBS spdlog::spdlog
    COMPILE_DEFS NOMINMAX=1 USTC_CG_BUILD_MODULE=1
)