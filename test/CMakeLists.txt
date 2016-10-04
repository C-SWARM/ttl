# Trivial example using gtest and gmock
add_executable(index Index.cpp)
# add_executable(scalar Scalar.cpp)
# add_executable(vector Scalar.cpp)

target_link_libraries(index gtest gmock_main)
# target_link_libraries(scalar gtest gmock_main)
# target_link_libraries(vector gtest gmock_main)

add_test(NAME index_test COMMAND scalar)
# add_test(NAME scalar_test COMMAND scalar)
# add_test(NAME vector_test COMMAND vector)

set_property(TARGET index PROPERTY CXX_STANDARD 14)
# set_property(TARGET scalar PROPERTY CXX_STANDARD 14)
# set_property(TARGET vector PROPERTY CXX_STANDARD 14)
