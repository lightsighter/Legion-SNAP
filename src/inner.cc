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

#include <cmath>

#include "snap.h"
#include "inner.h"

extern Legion::Logger log_snap;

//------------------------------------------------------------------------------
CalcInnerSource::CalcInnerSource(const Snap &snap, const Predicate &pred,
                       const SnapArray<3> &s_xs, const SnapArray<3> &flux0,
                       const SnapArray<3> &fluxm, const SnapArray<3> &q2grp0,
                       const SnapArray<3> &q2grpm, const SnapArray<3> &qtot,
                       int group_start, int group_stop)
  : SnapTask<CalcInnerSource, Snap::CALC_INNER_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  if (group_start == group_stop) {
    // Special case for a single field
    const Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group_start);
    s_xs.add_projection_requirement(READ_ONLY, *this, group_field);
    flux0.add_projection_requirement(READ_ONLY, *this, group_field);
    q2grp0.add_projection_requirement(READ_ONLY, *this, group_field);
    qtot.add_projection_requirement(WRITE_DISCARD, *this, group_field);
    // only include this requirement if we have more than one moment
    if (Snap::num_moments > 1) {
      fluxm.add_projection_requirement(READ_ONLY, *this, group_field);
      q2grpm.add_projection_requirement(READ_ONLY, *this, group_field);
    } else {
      fluxm.add_projection_requirement(NO_ACCESS, *this);
      q2grpm.add_projection_requirement(NO_ACCESS, *this);
    }
  } else {
    // General case for arbitrary set of fields
    std::vector<Snap::SnapFieldID> group_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      group_fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    s_xs.add_projection_requirement(READ_ONLY, *this, group_fields);
    flux0.add_projection_requirement(READ_ONLY, *this, group_fields);
    q2grp0.add_projection_requirement(READ_ONLY, *this, group_fields);
    qtot.add_projection_requirement(WRITE_DISCARD, *this, group_fields);
    // only include this requirement if we have more than one moment
    if (Snap::num_moments > 1) {
      fluxm.add_projection_requirement(READ_ONLY, *this, group_fields);
      q2grpm.add_projection_requirement(READ_ONLY, *this, group_fields);
    } else {
      fluxm.add_projection_requirement(NO_ACCESS, *this);
      q2grpm.add_projection_requirement(NO_ACCESS, *this);
    }
  } 
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 6; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 6; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<gpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Calc Inner Source");

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const bool multi_moment = (Snap::num_moments > 1);
#ifndef NDEBUG
  const unsigned num_groups = task->regions[0].privilege_fields.size();
#endif
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[2].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    AccessorRO<MomentQuad,3> fa_sxs(regions[0], *it);
    AccessorRO<double,3> fa_flux0(regions[1], *it);
    AccessorRO<double,3> fa_q2grp0(regions[2], *it);
    AccessorRW<MomentQuad,3> fa_qtot(regions[3], *it);
    if (multi_moment) {
      AccessorRO<MomentTriple,3> fa_fluxm(regions[4], *it);
      AccessorRO<MomentTriple,3> fa_q2grpm(regions[5], *it);
      for (DomainIterator<3> itr(dom); itr(); itr++)
      {
        MomentQuad sxs_quad = fa_sxs[*itr];
        const double q0 = fa_q2grp0[*itr];
        const double flux0 = fa_flux0[*itr];
        MomentQuad quad;
        quad[0] = q0 + flux0 * sxs_quad[0];
        MomentTriple qom = fa_q2grpm[*itr];
        MomentTriple fm = fa_fluxm[*itr];
        int moment = 0;
        for (int l = 1; l < Snap::num_moments; l++) {
          for (int i = 0; i < Snap::lma[l]; i++)
            quad[moment+i+1] = qom[moment+i] + fm[moment+i] * sxs_quad[l];
          moment += Snap::lma[l];
        }
        fa_qtot[*itr] = quad;
      }
    } else {
      for (DomainIterator<3> itr(dom); itr(); itr++)
      {
        MomentQuad sxs_quad = fa_sxs[*itr];
        const double q0 = fa_q2grp0[*itr];
        const double flux0 = fa_flux0[*itr];
        MomentQuad quad;
        quad[0] = q0 + flux0 * sxs_quad[0];
        fa_qtot[*itr] = quad;
      }
    }
  }
#endif
}

#ifdef USE_GPU_KERNELS
  extern void run_inner_source_single_moment(const Rect<3> subgrid_bounds,
                                    const AccessorRO<MomentQuad,3> fa_sxs,
                                    const AccessorRO<double,3> fa_flux0,
                                    const AccessorRO<double,3> fa_q2grp0,
                                    const AccessorWO<MomentQuad,3> fa_qtot);
  extern void run_inner_source_multi_moment(const Rect<3> subgrid_bounds,
                                   const AccessorRO<MomentQuad,3> fa_sxs,
                                   const AccessorRO<double,3> fa_flux0,
                                   const AccessorRO<double,3> fa_q2grp0,
                                   const AccessorRO<MomentTriple,3> fa_fluxm,
                                   const AccessorRO<MomentTriple,3> fa_q2grpm,
                                   const AccessorWO<MomentQuad,3> fa_qtot,
                                            const int num_moments, const int lma[4]);
#endif

//------------------------------------------------------------------------------
/*static*/ void CalcInnerSource::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const bool multi_moment = (Snap::num_moments > 1);
#ifndef NDEBUG
  const unsigned num_groups = task->regions[0].privilege_fields.size();
#endif
  assert(num_groups == task->regions[1].privilege_fields.size());
  assert(num_groups == task->regions[2].privilege_fields.size());

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    AccessorRO<MomentQuad,3> fa_sxs(regions[0], *it);
    AccessorRO<double,3> fa_flux0(regions[1], *it);
    AccessorRO<double,3> fa_q2grp0(regions[2], *it);
    AccessorWO<MomentQuad,3> fa_qtot(regions[3], *it);
    if (multi_moment) {
      AccessorRO<MomentTriple,3> fa_fluxm(regions[4], *it);
      AccessorRO<MomentTriple,3> fa_q2grpm(regions[5], *it);
      run_inner_source_multi_moment(dom.bounds, fa_sxs, fa_flux0, fa_q2grp0,
                                    fa_fluxm, fa_q2grpm, fa_qtot,
                                    Snap::num_moments, Snap::lma);
    } else {
      run_inner_source_single_moment(dom.bounds, fa_sxs, fa_flux0, 
                                     fa_q2grp0, fa_qtot); 
    }
  }
#else
  assert(false);
#endif
#endif
}

//------------------------------------------------------------------------------
TestInnerConvergence::TestInnerConvergence(const Snap &snap, 
                                           const Predicate &pred,
                                           const SnapArray<3> &flux0,
                                           const SnapArray<3> &flux0pi,
                                           const Future &true_future,
                                           int group_start, int group_stop)
  : SnapTask<TestInnerConvergence, Snap::TEST_INNER_CONVERGENCE_TASK_ID>(
      snap, snap.get_launch_bounds(), pred)
//------------------------------------------------------------------------------
{
  if (group_start == group_stop) {
    // Special case for a single field
    const Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group_start);
    flux0.add_projection_requirement(READ_ONLY, *this, group_field);
    flux0pi.add_projection_requirement(READ_ONLY, *this, group_field);
  } else {
    // General case for arbitrary set of fields
    std::vector<Snap::SnapFieldID> group_fields((group_stop - group_start) + 1);
    for (int group = group_start; group <= group_stop; group++)
      group_fields[group-group_start] = SNAP_ENERGY_GROUP_FIELD(group);
    flux0.add_projection_requirement(READ_ONLY, *this, group_fields);
    flux0pi.add_projection_requirement(READ_ONLY, *this, group_fields);
  }
  predicate_false_future = true_future;
}

//------------------------------------------------------------------------------
/*static*/ void TestInnerConvergence::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need x86 CPU
  execution_constraints.add_constraint(ISAConstraint(X86_ISA));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 2; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/,
                                             Snap::get_soa_layout());
  register_cpu_variant<bool, cpu_implementation>(execution_constraints,
                                                 layout_constraints,
                                                 true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void TestInnerConvergence::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  // Need a CUDA GPU with at least sm30
  execution_constraints.add_constraint(ISAConstraint(CUDA_ISA | SM_30_ISA));
  // Need at least 128B of shared memory
  execution_constraints.add_constraint(
      ResourceConstraint(SHARED_MEMORY_SIZE, GE_EK/*>=*/, 128/*B*/));
  TaskLayoutConstraintSet layout_constraints;
  // All regions need to be SOA
  for (unsigned idx = 0; idx < 2; idx++)
    layout_constraints.add_layout_constraint(idx/*index*/, 
                                             Snap::get_soa_layout());
  register_gpu_variant<DeferredValue<bool>, gpu_implementation>(
                                                 execution_constraints,
                                                 layout_constraints,
                                                 true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ bool TestInnerConvergence::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running Test Inner Convergence");

  // Get the index space domain for iteration
  assert(task->regions[0].region.get_index_space() == 
         task->regions[1].region.get_index_space());
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const double tolr = 1.0e-12;
  const double epsi = Snap::convergence_eps;
  // Iterate over all the energy groups
  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    AccessorRO<double,3> fa_flux0(regions[0], *it);
    AccessorRO<double,3> fa_flux0pi(regions[1], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++)
    {
      double flux0pi = fa_flux0pi[*itr];
      double df = 1.0;
      if (fabs(flux0pi) < tolr) {
        flux0pi = 1.0;
        df = 0.0;
      }
      double flux0 = fa_flux0[*itr];
      df = fabs( (flux0 / flux0pi) - df );
      if (df > epsi)
        return false;
    }
  }
  return true;
#else
  return false;
#endif
}

#ifdef USE_GPU_KERNELS
extern void run_inner_convergence(const Rect<3> subgrid_bounds,
                            const DeferredValue<bool> &result,
                            const std::vector<AccessorRO<double,3> > &fa_flux0,
                            const std::vector<AccessorRO<double,3> > &fa_flux0pi,
                            const double epsi);
#endif

//------------------------------------------------------------------------------
/*static*/ DeferredValue<bool> 
            TestInnerConvergence::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  log_snap.info("Running GPU Test Inner Convergence");
  DeferredValue<bool> result(false);
#ifndef NO_COMPUTE
#ifdef USE_GPU_KERNELS
  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));
  const double epsi = Snap::convergence_eps;

  assert(task->regions[0].privilege_fields.size() == 
         task->regions[1].privilege_fields.size());
  std::vector<AccessorRO<double,3> > fa_flux0(
      task->regions[0].privilege_fields.size());
  std::vector<AccessorRO<double,3> > fa_flux0pi(fa_flux0.size());
  unsigned idx = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, idx++)
  {
    fa_flux0[idx] = AccessorRO<double,3>(regions[0], *it);
    fa_flux0pi[idx] = AccessorRO<double,3>(regions[1], *it);
  }
  run_inner_convergence(dom.bounds, result, fa_flux0, fa_flux0pi, epsi);
#else
  assert(false);
#endif
#endif
  return result;
}

