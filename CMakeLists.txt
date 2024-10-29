cmake_minimum_required(VERSION 3.21)

# Gets project directory name
get_filename_component(PROJECT_NAME ${CMAKE_SOURCE_DIR} NAME)

# Sets project name
project(${PROJECT_NAME})

# Enables generation of compile_commands.json
# Works fine on linux. Needs ninja on windows.
# This project uses a compile_flags.txt though
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prints project name
message("Project Name:: ${PROJECT_NAME}")


# Sets the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Prints the compiler version
message(STATUS "CMAKE_CXX_COMPILER_VERSION:: ${CMAKE_CXX_COMPILER_VERSION}")


# Find llama.cpp installation
set(LLAMA_CPP_DIR "C:/opt/llama.cpp" CACHE PATH "Path to llama.cpp installation")
find_library(LLAMA_LIBRARY llama PATHS "${LLAMA_CPP_DIR}/lib" REQUIRED)
find_path(LLAMA_INCLUDE_DIR llama.h PATHS "${LLAMA_CPP_DIR}/include" REQUIRED)


# Adds source file
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Removes main.cpp from SOURCES. This is mainly for when testing so as not to have 2 main methods
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

# Adds back main.cpp when generating executable for main app
add_executable(${PROJECT_NAME} "src/main.cpp" ${SOURCES})

# Sets include directories for the main target
target_include_directories(${PROJECT_NAME} PRIVATE include ${LLAMA_INCLUDE_DIR})


# Link any necessary libraries here, if needed
target_link_libraries(${PROJECT_NAME} PRIVATE ${LLAMA_LIBRARY})


# Windows DLL handling for main executable
if(WIN32)
  add_custom_command( TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${LLAMA_CPP_DIR}/bin/llama.dll"
      $<TARGET_FILE_DIR:${PROJECT_NAME}>
  )
endif()


# Finds Google Test
find_package(GTest CONFIG REQUIRED)

# Add test source files
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")

# Generates test executable
add_executable(${PROJECT_NAME}_test ${TEST_SOURCES} ${SOURCES})

# Set include directories for the test target
target_include_directories(${PROJECT_NAME}_test PRIVATE include ${GTEST_INCLUDE_DIRS} ${LLAMA_INCLUDE_DIR})

# Link Google Test and raylib to test executable
target_link_libraries(${PROJECT_NAME}_test PRIVATE GTest::gtest GTest::gtest_main ${LLAMA_LIBRARY})


# Windows DLL handling for test executable
if(WIN32)
  add_custom_command( TARGET ${PROJECT_NAME}_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${LLAMA_CPP_DIR}/bin/llama.dll"
      $<TARGET_FILE_DIR:${PROJECT_NAME}_test>
  )
endif()


# Custom target to run the executable after building
add_custom_target( run
  COMMAND ${PROJECT_NAME}
  DEPENDS ${PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
