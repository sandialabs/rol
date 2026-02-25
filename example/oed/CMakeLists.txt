
ROL_ADD_EXECUTABLE(
  PolynomialRegression1D
  SOURCES example_01.cpp
  ADD_DIR_TO_NAME
  )

IF(${PROJECT_NAME}_ENABLE_Teuchos)
TRIBITS_INCLUDE_DIRECTORIES(${${PACKAGE_NAME}_SOURCE_DIR}/adapters/teuchos/src/sol)

ROL_ADD_EXECUTABLE(
  PolynomialRegression1D_MO
  SOURCES example_02.cpp
  ADD_DIR_TO_NAME
  )

ENDIF()

ROL_COPY_FILES_TO_BINARY_DIR(
  OEDExampleDataCopy
  SOURCE_FILES
    input.xml input_02.xml plotResults.m pointsGL.txt weightsGL.txt
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}"
)
