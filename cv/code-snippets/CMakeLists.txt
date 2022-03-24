set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin/)


add_executable(combine-imgs combine-imgs.cpp)
target_link_libraries(combine-imgs ${OpenCV_LIBS})