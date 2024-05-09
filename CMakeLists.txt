cmake_minimum_required(VERSION 3.5.0)
project(CudaRayTracing CUDA CXX C)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
include(CTest)
enable_testing()
find_package ( OpenGL REQUIRED )
include_directories(
    "./include"
    "ext/imgui"
    "ext/imgui/backends"
)
set ( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set ( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
set ( GLFW_HOME "ext/glfw-3.3.9.bin.WIN64" )
link_directories( "${GLFW_HOME}/lib-vc2022" )
set ( GLAD_HOME "ext/glad" )
set ( IMGUI_HOME "ext/imgui" )
set ( GLFW_INC "${GLFW_HOME}/include" )
set ( GLFW_LIB "glfw3" )

set ( GLAD_INC "${GLAD_HOME}/include" )
set ( GLAD_SRC "${GLAD_HOME}/src" )

file ( GLOB glad_src_files "${GLAD_SRC}/glad.c" )
file ( GLOB glad_header_files "${GLAD_INC}/glad/glad.h" "${GLAD_INC}/KHR/khrplatform.h" )

file ( GLOB imgui_src_files "${IMGUI_HOME}/imgui.cpp" 
                            "${IMGUI_HOME}/imgui_draw.cpp"
                            "${IMGUI_HOME}/imgui_tables.cpp"
                            "${IMGUI_HOME}/imgui_widgets.cpp"
                            "${IMGUI_HOME}/backends/imgui_impl_glfw.cpp"
                            "${IMGUI_HOME}/backends/imgui_impl_opengl3.cpp"
                            )

add_executable(CudaRayTracing src/main.cu ${glad_header_files} ${glad_src_files} ${imgui_src_files})

# Include CUDA
include_directories(${CUDA_INCLUDE_DIRS})
# Link against the curand library
set_target_properties(CudaRayTracing PROPERTIES CUDA_ARCHITECTURES "86;75;70;61;60")
target_link_libraries(CudaRayTracing ${CUDA_LIBRARIES} ${GLFW_LIB} ${OPENGL_LIBRARIES})
target_compile_features(CudaRayTracing PRIVATE cxx_std_17)
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