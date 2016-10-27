# Trivial example using gtest and gmock
add_executable(index Index.cpp)
add_executable(tensor Tensor.cpp)
add_executable(delta Delta.cpp)
add_executable(epsilon Epsilon.cpp)
add_executable(tensor_bind TensorBind.cpp)
add_executable(index_map IndexMap.cpp)
add_executable(operators Operators.cpp)
add_executable(trees Trees.cpp)


target_link_libraries(tensor gtest_main)
target_link_libraries(delta gtest_main)
target_link_libraries(epsilon gtest_main)
target_link_libraries(tensor_bind gtest_main)
target_link_libraries(index_map gtest_main)
target_link_libraries(operators gtest_main)
target_link_libraries(trees gtest_main)

add_test(NAME index_test COMMAND index)
add_test(NAME tensor_test COMMAND tensor)
add_test(NAME delta_test COMMAND delta)
add_test(NAME epsilon_test COMMAND epsilon)
add_test(NAME tensor_bind_test COMMAND tensor_bind)
add_test(NAME index_map_test COMMAND index_map)
add_test(NAME operators_test COMMAND operators)
add_test(NAME trees_test COMMAND trees)

set_property(GLOBAL PROPERTY CXX_STANDARD 11)
