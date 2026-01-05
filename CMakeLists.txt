cmake_minimum_required(VERSION 3.20)
project(glimbus LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/external)

include_directories(
    ${CMAKE_SOURCE_DIR}/external/glad/include 
    ${CMAKE_SOURCE_DIR}/external/tinyobjloader
    ${CMAKE_SOURCE_DIR}/external/stb_image_write
    ${CMAKE_SOURCE_DIR}/include
)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG latest
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	0af55ccecd98d4e5a8d1fad7de25ba429d60e863 #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

# Sources
# file(GLOB_RECURSE PROJECT_SOURCES src/*.cpp)

set(PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp 
    ${CMAKE_SOURCE_DIR}/src/Triangle.cpp 
    ${CMAKE_SOURCE_DIR}/src/Lens.cpp 
    ${CMAKE_SOURCE_DIR}/src/bvh.cpp 
    ${CMAKE_SOURCE_DIR}/src/random.cpp 
    ${CMAKE_SOURCE_DIR}/src/LoadObj.cpp 
    ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c
)

# Build executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c)

# Link libraries
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw OpenGL::GL glm::glm)