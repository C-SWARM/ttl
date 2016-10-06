# Trivial example using gtest and gmock
add_executable(index Index.cpp)
add_executable(tensor Tensor.cpp)
# add_executable(scalar Scalar.cpp)
# add_executable(vector Vector.cpp)

target_link_libraries(tensor gtest_main)
# target_link_libraries(scalar gtest gmock_main)
# target_link_libraries(vector gtest gmock_main)

add_test(NAME index_test COMMAND index)
add_test(NAME tensor_test COMMAND tensor)
# add_test(NAME scalar_test COMMAND scalar)
# add_test(NAME vector_test COMMAND vector)

set_property(TARGET gtest PROPERTY CXX_STANDARD 11)
set_property(TARGET gtest_main PROPERTY CXX_STANDARD 11)
set_property(TARGET index PROPERTY CXX_STANDARD 11)
set_property(TARGET tensor PROPERTY CXX_STANDARD 11)
# set_property(TARGET scalar PROPERTY CXX_STANDARD 14)
# set_property(TARGET vector PROPERTY CXX_STANDARD 14)
