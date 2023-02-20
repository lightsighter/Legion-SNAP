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

#ifndef __OUTER_H__
#define __OUTER_H__

#include "snap.h"
#include "resilience.h"

class CalcOuterSource : public SnapTask<CalcOuterSource, 
                                        Snap::CALC_OUTER_SOURCE_TASK_ID> {
public:
  CalcOuterSource(const Snap &snap, const Predicate &pred,
                  const SnapArray<3> &qi, const SnapArray<2> &slgg,
                  const SnapArray<3> &mat, const SnapArray<3> &q2rgp0, 
                  const SnapArray<3> &q2grpm, const SnapArray<3> &flux0,
                  const SnapArray<3> &fluxm);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class TestOuterConvergence : public SnapTask<TestOuterConvergence,
                                             Snap::TEST_OUTER_CONVERGENCE_TASK_ID> {
public:
  TestOuterConvergence(const Snap &snap, const Predicate &pred,
                       const SnapArray<3> &flux0, const SnapArray<3> &flux0po,
                       const Future &inner_converged, const Future &true_future,
                       int group_start, int group_stop);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static bool cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static DeferredValue<bool> gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __OUTER_H__

