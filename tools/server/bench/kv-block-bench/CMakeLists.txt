set(TARGET llama-kv-block-bench)

add_executable(${TARGET} kv-block-bench.cpp ../../server-prompt-tree.cpp)
target_include_directories(${TARGET} PRIVATE ../..)
target_link_libraries(${TARGET} PRIVATE llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
