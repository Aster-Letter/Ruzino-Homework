USTC_CG_ADD_LIB(
	rzpython
	SHARED
	PRIVATE_LIBS 
		GUI glm::glm spdlog::spdlog rzconsole
	COMPILE_DEFS
		NOMINMAX=1
)