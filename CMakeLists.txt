#------------------------------------------------------------------------------#
# Copyright 2023 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

set(CPU_SOURCES
  src/convergence.cc src/convergence.h
  src/expxs.cc src/expxs.h
  src/init.cc src/init.h
  src/inner.cc src/inner.h
  src/main.cc
  src/mapper.cc
  src/mms.cc src/mms.h
  src/outer.cc src/outer.h
  src/snap.cc src/snap.h
  src/sweep.cc src/sweep.h

  src/snap_cuda_help.h
  src/snap_types.h
)
if(Legion_USE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
  set(GPU_SOURCES
    src/gpu_outer.cu
    src/gpu_inner.cu
    src/gpu_sweep.cu
    src/gpu_expxs.cu
  )
  cuda_add_executable(snap ${CPU_SOURCES} ${GPU_SOURCES})
  target_compile_definitions(snap PRIVATE USE_GPU_KERNELS)
elseif(Legion_USE_HIP)
  set(GPU_SOURCES
    src/gpu_outer.cu
    src/gpu_inner.cu
    src/gpu_sweep.cu
    src/gpu_expxs.cu
  )
  if(Legion_HIP_TARGET STREQUAL "CUDA")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
    cuda_add_executable(snap ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(snap PRIVATE __HIP_PLATFORM_NVIDIA__)
  elseif (Legion_HIP_TARGET STREQUAL "ROCM")
    hip_add_executable(snap ${CPU_SOURCES} ${GPU_SOURCES})
    target_compile_definitions(snap PRIVATE __HIP_PLATFORM_AMD__)
  endif()
  target_compile_definitions(snap PRIVATE USE_GPU_KERNELS)
else()
  add_executable(snap ${CPU_SOURCES})
endif()

target_compile_definitions(snap PRIVATE DISABLE_SPECULATION SNAP_USE_RELAXED_COHERENCE)  # Do this until Realm supports multi-field reduction instances

target_compile_options(snap PRIVATE -march=native) # Apparently can't build without this

target_compile_options(snap PRIVATE -Wno-ignored-attributes) # FIXME: https://github.com/StanfordLegion/Legion-SNAP/issues/2
target_compile_options(snap PRIVATE -Wno-maybe-uninitialized -Wno-unused-variable)

target_link_libraries(snap Legion::Legion ResilienceStatic)

set_target_properties(snap PROPERTIES CXX_STANDARD 11)

add_test(NAME snap COMMAND ${PROJECT_SOURCE_DIR}/tests/runner.py $<TARGET_FILE:snap> ${CMAKE_CURRENT_SOURCE_DIR}/input/mms.in)
