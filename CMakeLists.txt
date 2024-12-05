cmake_minimum_required(VERSION 3.10)

project(cgaows)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python
find_package(Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE = ${Python_EXECUTABLE}")
message(STATUS "Python_STDLIB = ${Python_STDLIB}")
message(STATUS "Python_INCLUDE_DIRS = ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES = ${Python_LIBRARIES}")
message(STATUS "Python_LIBRARY_DIRS = ${Python_LIBRARY_DIRS}")
message(STATUS "Python_RUNTIME_LIBRARY_DIRS = ${Python_RUNTIME_LIBRARY_DIRS}")
message(STATUS "Python_VERSION = ${Python_VERSION}")
message(STATUS "Python_VERSION_MAJOR = ${Python_VERSION_MAJOR}")
message(STATUS "Python_VERSION_MINOR = ${Python_VERSION_MINOR}")
message(STATUS "Python_VERSION_PATCH = ${Python_VERSION_PATCH}")

# define install loacation
file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX})  # Define via cmake -DCMAKE_INSTALL_PREFIX="/path/to/install/" ..
message( STATUS "Default value of CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}" )

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python_module)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows)

# prepare module's python files
configure_file(src/__init__.py.in         ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/__init__.py)
configure_file(src/cusped_orbitals.py     ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/cusped_orbitals.py         COPYONLY)
configure_file(src/basis_set_info.py      ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/basis_set_info.py          COPYONLY)
configure_file(src/integrand_at_rtp.py    ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/integrand_at_rtp.py        COPYONLY)

# Set up library
add_library(cuspedgaussians STATIC ${CMAKE_SOURCE_DIR}/src/cusped_gaussians.cpp)

# Specify include directories
target_include_directories(cuspedgaussians PUBLIC include) 

# Install the library and header file in build dir
install(TARGETS cuspedgaussians DESTINATION lib)
install(FILES ${CMAKE_SOURCE_DIR}/src/cusped_gaussians.hpp DESTINATION include)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/__init__.py            DESTINATION  python_module/cgaows)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/cusped_orbitals.py     DESTINATION  python_module/cgaows)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/basis_set_info.py      DESTINATION  python_module/cgaows)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_module/cgaows/integrand_at_rtp.py    DESTINATION  python_module/cgaows)

enable_testing()

## disabling these tests for now
#if(FALSE)

# Note - still working on file organization here and turning this into a test rather than an example
#############################
###### Neon example #########
#############################
# Specify the executable
add_executable(run_example examples/neon/example_Ne.cpp)
#add_executable(run_example ${CMAKE_SOURCE_DIR}/examples/neon/example_Ne.cpp)

# Link the library to the executable
target_link_libraries(run_example cuspedgaussians)

# neon example with predetermined cusp information
#add_test(NAME run_example 
#         COMMAND run_example 
#        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/examples/neon
#        )
##############################

# Test to make sure txt files were generated correctly and read into C++ correctly
add_executable(Ne_test ${CMAKE_SOURCE_DIR}/examples/neon/Ne_test.cpp)

# Add the test to CTest 
add_test(NAME ne_test COMMAND Ne_test)




