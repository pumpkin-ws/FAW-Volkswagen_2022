set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${FAW-Volkswagen_SOURCE_DIR}/lib/)

find_package(Boost REQUIRED filesystem)

add_library(
    LapBlend SHARED
    source/lap_blend.hpp
    source/lap_blend.cpp
)

add_library(
    PerspectiveTransform SHARED
    source/image_warper.hpp
    source/image_warper.cpp
)

add_library(
    DetectTraget SHARED
    source/target_detector.hpp
    source/target_detector.cpp
)

add_library(
    FileManip SHARED
    source/file_manip.hpp
    source/file_manip.cpp
)

target_link_libraries(FileManip ${Boost_LIBRARIES})

add_executable(combine-imgs combine-imgs.cpp)
target_link_libraries(combine-imgs ${OpenCV_LIBS})

add_executable(laplacian-blending laplacian-blending.cpp)
target_link_libraries(laplacian-blending ${OpenCV_LIBS} LapBlend)

add_executable(unwarp-imgs unwarp-imgs.cpp)
target_link_libraries(unwarp-imgs ${OpenCV_LIBS} DetectTraget FileManip PerspectiveTransform)