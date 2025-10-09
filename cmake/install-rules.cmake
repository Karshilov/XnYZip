install(
    TARGETS XnYSZ_exe
    RUNTIME COMPONENT XnYSZ_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
