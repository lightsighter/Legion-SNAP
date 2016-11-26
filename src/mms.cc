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

#include "snap.h"
#include "mms.h"

#include <cmath>

using namespace LegionRuntime::Accessor;

template<bool COS>
void mms_trigint(const int lc, const double d, const double del, 
                 const double *cb, double *fn);

template<>
void mms_trigint<true/*COS*/>(const int lc, const double d, const double del, 
                              const double *cb, double *fn)
{
  memset(fn, 0, lc * sizeof(double));
  const double denom = d * del;
  for (int i = 0; i < lc; i++) {
    fn[i] = cos(d * cb[i]) - cos(d * cb[i+1]);
    fn[i] /= denom;
  }
}

template<>
void mms_trigint<false/*COS*/>(const int lc, const double d, const double del, 
                               const double *cb, double *fn)
{
  memset(fn, 0, lc * sizeof(double));
  for (int i = 0; i < lc; i++) {
    fn[i] = sin(d * cb[i+1]) - sin(d * cb[i]);
    fn[i] /= del;
  }
}

//------------------------------------------------------------------------------
MMSInitFlux::MMSInitFlux(const Snap &snap, const SnapArray &ref_flux, 
                         const SnapArray &ref_fluxm)
  : SnapTask<MMSInitFlux, Snap::MMS_INIT_FLUX_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  ref_flux.add_projection_requirement(READ_WRITE, *this);
  ref_fluxm.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitFlux::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitFlux::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  double *tx = (double*)malloc(Snap::nx * sizeof(double));
  double *ty = (double*)malloc(Snap::ny * sizeof(double));
  double *tz = (double*)malloc(Snap::nz * sizeof(double));
  double *ib = (double*)malloc((Snap::nx+1) * sizeof(double));
  double *jb = (double*)malloc((Snap::ny+1) * sizeof(double));
  double *kb = (double*)malloc((Snap::nz+1) * sizeof(double));

  const double a = PI / Snap::lx;
  const double b = PI / Snap::ly;
  const double c = PI / Snap::lz;
  const double dx = Snap::lx / double(Snap::nx);
  const double dy = Snap::ly / double(Snap::ny);
  const double dz = Snap::lz / double(Snap::nz);
  ib[0] = subgrid_bounds.lo[0] * dx;
  for (int i = 1; i <= Snap::nx; i++)
    ib[i] = ib[i-1] + dx;
  jb[0] = subgrid_bounds.lo[1] * dy;
  for (int j = 1; j <= Snap::ny; j++)
    jb[j] = jb[j-1] + dx;
  kb[0] = subgrid_bounds.lo[2] * dz;
  for (int k = 1; k <= Snap::nz; k++)
    kb[k] = kb[k-1] + dx;

  mms_trigint<true/*COS*/>(Snap::nx, a, dx, ib, tx);
  if (Snap::num_dims > 1) {
    mms_trigint<true/*COS*/>(Snap::ny, b, dy, jb, ty);
    if (Snap::num_dims > 2) {
      mms_trigint<true/*COS*/>(Snap::nz, c, dz, kb, tz);
    } else {
      for (int k = 0; k < Snap::nz; k++)
        tz[k] = 1.0;
    }
  } else {
    for (int j = 0; j < Snap::ny; j++)
      ty[j] = 1.0;
    for (int k = 0; k < Snap::nz; k++)
      tz[k] = 1.0;
  }

  unsigned g = 1;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_flux = 
      regions[0].get_field_accessor(*it).typeify<double>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++) {    
      int i = itr.p[0] - subgrid_bounds.lo[0];
      assert(i < Snap::nx);
      int j = itr.p[1] - subgrid_bounds.lo[1];
      assert(j < Snap::ny);
      int k = itr.p[2] - subgrid_bounds.lo[2];
      assert(k < Snap::nz);
      double value = (double(g) * tx[i] * ty[j] * tz[k]);
      fa_flux.write(DomainPoint::from_point<3>(itr.p), value);
    }
  }

  double p[3] = { 0.0, 0.0, 0.0 };
  for (int c = 0; c < Snap::num_corners; c++) {
    for (int l = 1; l < Snap::num_moments; l++) {
      unsigned offset = (l + c * Snap::num_moments) * Snap::num_angles;
      for (int ang = 0; ang < Snap::num_angles; ang++)
        p[l-1] += Snap::w[ang] * Snap::ec[offset + ang];
    }
  }

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    RegionAccessor<AccessorType::Generic,double> fa_flux = 
      regions[0].get_field_accessor(*it).typeify<double>();
    RegionAccessor<AccessorType::Generic,MomentTriple> fa_fluxm = 
      regions[1].get_field_accessor(*it).typeify<MomentTriple>();
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++) {
      const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      double flux = fa_flux.read(dp); 
      MomentTriple result;
      for (int l = 0; l < 3; l++)
        result[l] = p[l] * flux;
      fa_fluxm.write(dp, result);
    }
  }

  free(tx);
  free(ty);
  free(tz);
#endif
}

//------------------------------------------------------------------------------
MMSInitSource::MMSInitSource(const Snap &snap, const SnapArray &ref_flux,
                             const SnapArray &ref_fluxm, 
                             const SnapArray &qim,int c)
  : SnapTask<MMSInitSource, Snap::MMS_INIT_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), corner(c)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&corner, sizeof(corner));
  ref_flux.add_projection_requirement(READ_ONLY, *this);
  ref_fluxm.add_projection_requirement(READ_ONLY, *this);
  qim.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitSource::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE

#endif
}

//------------------------------------------------------------------------------
MMSScale::MMSScale(const Snap &snap, const SnapArray &qim, double f)
  : SnapTask<MMSScale, Snap::MMS_SCALE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), scale_factor(f)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&scale_factor, sizeof(scale_factor));
  qim.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSScale::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSScale::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(task->arglen == sizeof(double));
  const double scale_factor = *((double*)task->args);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  const size_t angle_buffer_size = Snap::num_angles * sizeof(double);
  double *angle_buffer = (double*)malloc(angle_buffer_size);

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    RegionAccessor<AccessorType::Generic> fa_qim = 
      regions[0].get_field_accessor(*it);
    for (GenericPointInRectIterator<3> itr(subgrid_bounds); itr; itr++)
    {
      const DomainPoint dp = DomainPoint::from_point<3>(itr.p);
      fa_qim.read_untyped(dp, angle_buffer, angle_buffer_size);
      for (unsigned ang = 0; ang < Snap::num_angles; ang++)
        angle_buffer[ang] *= scale_factor;
      fa_qim.write_untyped(dp, angle_buffer, angle_buffer_size);
    }
  }
  free(angle_buffer);
#endif
}

//------------------------------------------------------------------------------
MMSVerify::MMSVerify(const Snap &snap, const SnapArray &flux, 
                     const SnapArray &ref_flux)
  : SnapTask<MMSVerify, Snap::MMS_VERIFY_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  flux.add_projection_requirement(READ_ONLY, *this);
  ref_flux.add_projection_requirement(READ_ONLY, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSVerify::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSVerify::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE

#endif
}

