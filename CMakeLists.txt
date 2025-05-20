cmake_minimum_required(VERSION 3.20)
project(MyVulkanEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Vulkan REQUIRED COMPONENTS glslc)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm REQUIRED) # Added GLM dependency

# Shader compilation
find_program(GLSLC glslc HINTS ${Vulkan_GLSLC_EXECUTABLE})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
set(SHADER_SOURCES shaders/triangle.vert shaders/triangle.frag)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SHADER_OUTPUT ${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.spv) # Output to source dir
    add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/${SHADER} -o ${SHADER_OUTPUT}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
endforeach()
add_custom_target(Shaders ALL DEPENDS ${SHADER_OUTPUTS})

# Add executable
add_executable(MyVulkanEngine
        src/main.cpp
        src/VulkanUtils.h
        src/VulkanUtils.cpp
        src/VulkanRenderer.h
        src/VulkanRenderer.cpp
)
add_dependencies(MyVulkanEngine Shaders)
target_link_libraries(MyVulkanEngine PRIVATE Vulkan::Vulkan glfw glm::glm)
target_include_directories(MyVulkanEngine PRIVATE src) # Ensure src is included

# Build options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(MyVulkanEngine PRIVATE -g -O0)
    target_compile_definitions(MyVulkanEngine PRIVATE VK_ENABLE_VALIDATION_LAYERS)
else()
    target_compile_options(MyVulkanEngine PRIVATE -O3 -march=native)
endif()