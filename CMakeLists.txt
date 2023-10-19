cmake_minimum_required(VERSION 3.5.0)
project(CudaRayTracing CUDA CXX C)

include(CTest)
enable_testing()
find_package ( OpenGL REQUIRED )

include_directories("./include")
set ( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set ( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
set ( GLFW_HOME "d:/ENV/GLFW" )
set ( GLAD_HOME "d:/ENV/GLAD" )

set ( GLFW_INC "${GLFW_HOME}/include" )
set ( GLFW_LIB "${GLFW_HOME}/lib-vc2022/glfw3.lib" )

set ( GLAD_INC "${GLAD_HOME}/include" )
set ( GLAD_SRC "${GLAD_HOME}/src" )

file ( GLOB glad_src_file "${GLAD_SRC}/glad.c" )
file ( GLOB glad_header_files "${GLAD_INC}/glad/glad.h" "${GLAD_INC}/KHR/khrplatform.h" )

add_executable(CudaRayTracing src/main.cu ${glad_header_files} ${glad_src_file})

# Include CUDA
include_directories(${CUDA_INCLUDE_DIRS})
# Link against the curand library
set_target_properties(CudaRayTracing PROPERTIES CUDA_ARCHITECTURES "86;75;70;61;60")
target_link_libraries(CudaRayTracing ${CUDA_LIBRARIES} ${GLFW_LIB} ${OPENGL_LIBRARIES})
target_compile_features(CudaRayTracing PRIVATE cxx_std_14)
# if(WIN32)
#     set(CMAKE_EXE_LINKER_FLAGS /NODEFAULTLIB:"MSVCRT.lib")
#     target_link_options(CudaRayTracing PRIVATE "/NODEFAULTLIB:LIBCMT")
# endif()
include(CPack)

target_include_directories ( ${PROJECT_NAME}
	PRIVATE 
    ${GLFW_INC}
    ${GLAD_INC}
)