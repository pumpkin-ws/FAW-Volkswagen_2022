set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FAW-Volkswagen_SOURCE_DIR}/lib/)

add_library(
    LapBlend SHARED
    source/lap_blend.hpp
    source/lap_blend.cpp
)

add_executable(combine-imgs combine-imgs.cpp)
target_link_libraries(combine-imgs ${OpenCV_LIBS})

add_executable(laplacian-blending laplacian-blending.cpp)
target_link_libraries(laplacian-blending ${OpenCV_LIBS} LapBlend)