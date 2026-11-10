set(TARGET llama-kv-block-bench)

add_executable(${TARGET} kv-block-bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
