
ROL_ADD_EXECUTABLE(
  example_01
  SOURCES example_01.cpp
  ADD_DIR_TO_NAME
)

ROL_COPY_FILES_TO_BINARY_DIR(
  StochProx_Logistic
  SOURCE_FILES
    input.xml
    label_vector.txt 
    info.txt 
    data_matrix.txt
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}"
)
