/* Copyright 2016 NVIDIA Corporation
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
#include "legion.h"

using namespace Legion;

class CalcOuterSource : public SnapTask<CalcOuterSource> {
public:
  static const Snap::SnapTaskID TASK_ID = Snap::CALC_OUTER_SOURCE_TASK_ID;
  static const Snap::SnapReductionID REDOP = Snap::NO_REDUCTION_ID;
public:
  CalcOuterSource(const Snap &snap, const Predicate &pred,
                  const SnapArray &qi, const SnapArray &slgg,
                  const SnapArray &mat, const SnapArray &q2rgp0, 
                  const SnapArray &q2grpm);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class TestOuterConvergence : public SnapTask<TestOuterConvergence> {
public:
  static const Snap::SnapTaskID TASK_ID = Snap::TEST_OUTER_CONVERGENCE_TASK_ID;
  static const Snap::SnapReductionID REDOP = Snap::AND_REDUCTION_ID;
public:
  TestOuterConvergence(const Snap &snap, const Predicate &pred,
                       const SnapArray &flux0, const SnapArray &flux0po);
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static bool cpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static bool gpu_implementation(const Task *task,
     const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __OUTER_H__

