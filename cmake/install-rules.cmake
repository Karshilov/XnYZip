install(
    TARGETS tonSZ_exe
    RUNTIME COMPONENT tonSZ_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
