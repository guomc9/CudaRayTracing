cmake_minimum_required(VERSION 3.0.0)
project(CudaRayTracing CUDA CXX)

include(CTest)
enable_testing()

include_directories("./include")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
add_executable(CudaRayTracing src/main.cu)

# Find OpenCV
find_package(OpenCV REQUIRED)
# Include OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
# Link OpenCV
if(WIN32)
    find_library(OpenCV_world_LIB opencv_world*d HINTS ${OpenCV_LIB_DIR})
    find_library(OpenCV_videoio_LIB opencv_videoio*d HINTS ${OpenCV_LIB_DIR})
    target_link_libraries(CudaRayTracing ${OpenCV_world_LIB} ${OpenCV_videoio_LIB})
endif()
if(UNIX)
    target_link_libraries(CudaRayTracing ${OpenCV_LIBS})
endif()

# Find and link against the CUDA package
find_package(CUDA REQUIRED)
# Include CUDA
include_directories(${CUDA_INCLUDE_DIRS})
# Link against the curand library
set_target_properties(CudaRayTracing PROPERTIES CUDA_ARCHITECTURES "89;86;75;70;61;60")
target_link_libraries(CudaRayTracing ${CUDA_LIBRARIES})
target_compile_features(CudaRayTracing PRIVATE cxx_std_14)
if(WIN32)
    set(CMAKE_EXE_LINKER_FLAGS /NODEFAULTLIB:"MSVCRT.lib")
    target_link_options(CudaRayTracing PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()
include(CPack)
