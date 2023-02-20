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

#ifndef __MMS_H__
#define __MMS_H__

#include "snap.h"
#include "resilience.h"

class MMSInitFlux : public SnapTask<MMSInitFlux, Snap::MMS_INIT_FLUX_TASK_ID> {
public:
  MMSInitFlux(const Snap &snap, const SnapArray<3> &ref_flux, 
              const SnapArray<3> &ref_fluxm);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSInitSource : public SnapTask<MMSInitSource, Snap::MMS_INIT_SOURCE_TASK_ID> {
public:
  MMSInitSource(const Snap &snap, const SnapArray<3> &ref_flux, 
                const SnapArray<3> &ref_fluxm, const SnapArray<3> &mat,
                const SnapArray<1> &sigt, const SnapArray<2> &slgg,
                const SnapArray<3> &qim, int corner);
public:
  const int corner;
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSInitTimeDependent : public SnapTask<MMSInitTimeDependent, 
                                             Snap::MMS_INIT_TIME_DEPENDENT_TASK_ID> {
public:
  MMSInitTimeDependent(const Snap &snap, const SnapArray<1> &vel,
                       const SnapArray<3> &ref_flux, const SnapArray<3> &qi);
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSScale : public SnapTask<MMSScale, Snap::MMS_SCALE_TASK_ID> {
public:
  MMSScale(const Snap &snap, const SnapArray<3> &qim, double factor);
public:
  const double scale_factor;
public:
  static void preregister_cpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSCompare : public SnapTask<MMSCompare, Snap::MMS_COMPARE_TASK_ID> {
public:
  MMSCompare(const Snap &snap, const SnapArray<3> &flux, const SnapArray<3> &ref_flux);
public:
  static void preregister_cpu_variants(void);
public:
  static MomentTriple cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class MMSReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::MMS_REDUCTION_ID;
public:
  typedef MomentTriple LHS;
  typedef MomentTriple RHS;
  static const MomentTriple identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

#endif // __MMS_H__

