/* Copyright 2017 NVIDIA Corporation
 *
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract B609478 with Lawrence Livermore National Security, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __SWEEP_H__
#define __SWEEP_H__

#include "snap.h"
#include "resilience.h"

class MiniKBATask : public SnapTask<MiniKBATask, Snap::MINI_KBA_TASK_ID> {
public:
  static const int NON_GHOST_REQUIREMENTS = 3;
public:
  struct MiniKBAArgs {
  public:
    MiniKBAArgs(int c, int start, int stop)
      : corner(c), group_start(start), group_stop(stop) { }
  public:
    int corner;
    int group_start;
    int group_stop; // inclusive
  };
public:
  MiniKBATask(const Snap &snap, const Predicate &pred, 
              const SnapArray<3> &flux, const SnapArray<3> &fluxm,
              const SnapArray<3> &qtot, const SnapArray<1> &vdelt, 
              const SnapArray<3> &dinv, const SnapArray<3> &t_xs, 
              const SnapArray<3> &time_flux_in, 
              const SnapArray<3> &time_flux_out,
              const SnapArray<3> &qim, const SnapArray<2> &flux_xy,
              const SnapArray<2> &flux_yz, const SnapArray<2> &flux_xz,
              int group_start, int group_stop, int corner, 
              const int ghost_offsets[3]);
public:
  MiniKBAArgs mini_kba_args;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void sse_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void avx_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __SWEEP_H__

