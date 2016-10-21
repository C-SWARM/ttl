# Trivial example using gtest and gmock
add_executable(index Index.cpp)
add_executable(tensor Tensor.cpp)
add_executable(delta Delta.cpp)
add_executable(index_map IndexMap.cpp)

target_link_libraries(tensor gtest_main)
target_link_libraries(delta gtest_main)
target_link_libraries(index_map gtest_main)

add_test(NAME index_test COMMAND index)
add_test(NAME tensor_test COMMAND tensor)
add_test(NAME delta_test COMMAND delta)
add_test(NAME index_map_test COMMAND index_map)

set_property(TARGET gtest PROPERTY CXX_STANDARD 11)
set_property(TARGET gtest_main PROPERTY CXX_STANDARD 11)
set_property(TARGET index PROPERTY CXX_STANDARD 11)
set_property(TARGET tensor PROPERTY CXX_STANDARD 11)
set_property(TARGET delta PROPERTY CXX_STANDARD 11)
set_property(TARGET index_map PROPERTY CXX_STANDARD 11)
