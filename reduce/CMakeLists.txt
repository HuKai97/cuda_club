add_executable(reduce_v0_global_memory 
    reduce_v0_global_memory.cu
)

target_link_libraries(reduce_v0_global_memory PRIVATE
    CUDA::cudart
)

add_executable(reduce_v1_shared_memory 
    reduce_v1_shared_memory.cu
)

target_link_libraries(reduce_v1_shared_memory  PRIVATE
    CUDA::cudart
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(reduce_v0_global_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    target_compile_options(reduce_v1_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()