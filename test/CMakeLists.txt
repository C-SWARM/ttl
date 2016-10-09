# Trivial example using gtest and gmock
add_executable(index Index.cpp)
add_executable(tensor Tensor.cpp)

target_link_libraries(tensor gtest_main)

add_test(NAME index_test COMMAND index)
add_test(NAME tensor_test COMMAND tensor)

set_property(TARGET gtest PROPERTY CXX_STANDARD 11)
set_property(TARGET gtest_main PROPERTY CXX_STANDARD 11)
set_property(TARGET index PROPERTY CXX_STANDARD 11)
set_property(TARGET tensor PROPERTY CXX_STANDARD 11)
