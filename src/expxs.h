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

#ifndef __EXPXS_H__
#define __EXPXS_H__

#include "snap.h"
#include "resilience.h"

class ExpandCrossSection : public SnapTask<ExpandCrossSection,
                                           Snap::EXPAND_CROSS_SECTION_TASK_ID> {
public:
  ExpandCrossSection(const Snap &snap, const SnapArray<1> &sig, 
                     const SnapArray<3> &mat, const SnapArray<3> &xs, 
                     int group_start, int group_stop);
public:
  const int group_start;
  const int group_stop;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class ExpandScatteringCrossSection : public SnapTask<ExpandScatteringCrossSection,
                                    Snap::EXPAND_SCATTERING_CROSS_SECTION_TASK_ID> {
public:
  ExpandScatteringCrossSection(const Snap &snap, const SnapArray<2> &slgg,
                               const SnapArray<3> &mat, const SnapArray<3> &s_xs, 
                               int group_start, int group_stop);
public:
  const int group_start;
  const int group_stop;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class CalculateGeometryParam : public SnapTask<CalculateGeometryParam,
                                               Snap::CALCULATE_GEOMETRY_PARAM_TASK_ID> {
public:
  CalculateGeometryParam(const Snap &snap, const SnapArray<3> &t_xs, 
                         const SnapArray<1> &vdelt, const SnapArray<3> &dinv, 
                         int group_start, int group_stop);
public:
  const int group_start;
  const int group_stop;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __EXPXS_H__

