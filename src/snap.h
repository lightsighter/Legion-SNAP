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

#ifndef __SNAP_H__
#define __SNAP_H__

#include "resilience.h"
#include "default_mapper.h"
#include "snap_types.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifndef SNAP_MAX_ENERGY_GROUPS
#define SNAP_MAX_ENERGY_GROUPS            1024
#endif
#ifndef SNAP_MAX_WAVEFRONTS
#define SNAP_MAX_WAVEFRONTS               1024
#endif

#ifndef PI
#define PI (3.14159265358979)
#endif

// Hack: this seems to be the shortest way to override the namespace
#define Legion ResilientLegion

template<int DIM>
using Point = Legion::Point<DIM,long long>;
template<int DIM>
using Rect = Legion::Rect<DIM,long long>;
template<int DIM1, int DIM2>
using Matrix = Legion::Transform<DIM1,DIM2,long long>;
template<int DIM>
using IndexSpace = Legion::IndexSpaceT<DIM,long long>;
template<int DIM>
using IndexPartition = Legion::IndexPartitionT<DIM,long long>;
typedef Legion::FieldSpace FieldSpace;
template<int DIM>
using LogicalRegion = Legion::LogicalRegionT<DIM,long long>;
template<int DIM>
using LogicalPartition = Legion::LogicalPartitionT<DIM,long long>;
template<int DIM>
using Domain = Legion::DomainT<DIM, long long>;
template<typename FT, int N, typename T = long long>
using AccessorRO = Legion::FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = long long>
using AccessorWO = Legion::FieldAccessor<WRITE_DISCARD,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = long long>
using AccessorRW = Legion::FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<int DIM, typename T = long long>
using DomainIterator = Legion::PointInDomainIterator<DIM,T>;
template<int DIM, typename T = long long>
using RectIterator = Legion::PointInRectIterator<DIM,T>;
template<typename T>
using DeferredValue = Legion::DeferredValue<T>;
template<typename T, int DIM, typename COORD_T = long long, bool CHECK_BOUNDS = false>
using DeferredBuffer = Legion::DeferredBuffer<T,DIM,COORD_T,CHECK_BOUNDS>;
typedef Legion::Runtime Runtime;
typedef Legion::Context Context;
typedef Legion::PhysicalRegion PhysicalRegion;
typedef Legion::TaskLauncher TaskLauncher;
typedef Legion::CopyLauncher CopyLauncher;
typedef Legion::FillLauncher FillLauncher;
typedef Legion::InlineLauncher InlineLauncher;
typedef Legion::IndexTaskLauncher IndexTaskLauncher;
typedef Legion::IndexCopyLauncher IndexCopyLauncher;
typedef Legion::IndexFillLauncher IndexFillLauncher;
typedef Legion::Predicate Predicate;
typedef Legion::Future Future;
typedef Legion::FutureMap FutureMap;
typedef Legion::ArgumentMap ArgumentMap;
typedef Legion::TaskArgument TaskArgument;
typedef Legion::RegionRequirement RegionRequirement;
typedef Legion::Mappable Mappable;
typedef Legion::Task Task;
typedef Legion::Copy Copy;
typedef Legion::Close Close;
typedef Legion::Partition Partition;
typedef Legion::Fill Fill;
typedef Legion::ProjectionID ProjectionID;
typedef Legion::ShardID ShardID;
typedef Legion::LayoutConstraintID LayoutConstraintID;
typedef Legion::FieldID FieldID;
typedef Legion::PrivilegeMode PrivilegeMode;
typedef Legion::Machine Machine;
typedef Legion::Processor Processor;
typedef Legion::Memory Memory;
typedef Legion::ExecutionConstraintSet ExecutionConstraintSet;
typedef Legion::TaskLayoutConstraintSet TaskLayoutConstraintSet;
typedef Legion::TaskVariantRegistrar TaskVariantRegistrar;
typedef Legion::ProcessorConstraint ProcessorConstraint;
typedef Legion::ProjectionFunctor ProjectionFunctor;
typedef Legion::FieldAllocator FieldAllocator;
typedef Legion::PredicateLauncher PredicateLauncher;
typedef Legion::LayoutConstraintRegistrar LayoutConstraintRegistrar;
typedef Legion::SpecializedConstraint SpecializedConstraint;
typedef Legion::DimensionKind DimensionKind;
typedef Legion::OrderingConstraint OrderingConstraint;
typedef Legion::ISAConstraint ISAConstraint;
typedef Legion::ResourceConstraint ResourceConstraint;
typedef Legion::LaunchConstraint LaunchConstraint;
typedef Legion::FieldConstraint FieldConstraint;
typedef Legion::LayoutConstraintSet LayoutConstraintSet;
typedef Legion::MemoryConstraint MemoryConstraint;

extern Legion::Logger log_snap;

template<int DIM>
class SnapArray;

class Snap {
public:
  enum SnapTaskID {
    SNAP_TOP_LEVEL_TASK_ID,
    INIT_MATERIAL_TASK_ID,
    INIT_SOURCE_TASK_ID,
    INIT_SCATTERING_TASK_ID,
    INIT_VELOCITY_TASK_ID,
    INIT_GPU_SWEEP_TASK_ID,
    CALC_OUTER_SOURCE_TASK_ID,
    TEST_OUTER_CONVERGENCE_TASK_ID,
    CALC_INNER_SOURCE_TASK_ID,
    TEST_INNER_CONVERGENCE_TASK_ID,
    MINI_KBA_TASK_ID,
    EXPAND_CROSS_SECTION_TASK_ID,
    EXPAND_SCATTERING_CROSS_SECTION_TASK_ID,
    CALCULATE_GEOMETRY_PARAM_TASK_ID,
    MMS_INIT_FLUX_TASK_ID,
    MMS_INIT_SOURCE_TASK_ID,
    MMS_INIT_TIME_DEPENDENT_TASK_ID,
    MMS_SCALE_TASK_ID,
    MMS_COMPARE_TASK_ID,
    BIND_INNER_CONVERGENCE_TASK_ID,
    BIND_OUTER_CONVERGENCE_TASK_ID,
    SUMMARY_TASK_ID,
    LAST_TASK_ID, // must be last
  };
#define SNAP_TASK_NAMES                 \
    "Top_Level_Task",                   \
    "Initialize_Material",              \
    "Initialize_Source",                \
    "Initialize_Scattering",            \
    "Initialize_Velocity",              \
    "Initialize_GPU Sweep",             \
    "Calc_Outer_Source",                \
    "Test_Outer_Convergence",           \
    "Calc_Inner_Source",                \
    "Test_Inner_Convergence",           \
    "Mini_KBA",                         \
    "Expand_Cross_Section",             \
    "Expand_Scattering_Cross_Section",  \
    "Calcuate_Geometry Param",          \
    "MMS_Init_Flux",                    \
    "MMS_Init_Source",                  \
    "MMS_Init_Time Dependent",          \
    "MMS_Scale",                        \
    "MMS_Compare",                      \
    "Bind_Inner_Convergence",           \
    "Bind_Outer_Convergence",           \
    "Summary"
  static const char* task_names[LAST_TASK_ID];
  enum MaterialLayout {
    HOMOGENEOUS_LAYOUT = 0,
    CENTER_LAYOUT = 1,
    CORNER_LAYOUT = 2,
  };
  enum SourceLayout {
    EVERYWHERE_SOURCE = 0,
    CENTER_SOURCE = 1,
    CORNER_SOURCE = 2,
    MMS_SOURCE = 3,
  };
  enum SnapTunable {
    OUTER_RUNAHEAD_TUNABLE = Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_LAST,
    INNER_RUNAHEAD_TUNABLE = Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_LAST+1,
    SWEEP_ENERGY_CHUNKS_TUNABLE = Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_LAST+2,
    GPU_SMS_PER_SWEEP_TUNABLE = Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_LAST+3,
  };
  enum SnapReductionID {
    NO_REDUCTION_ID = 0,
    AND_REDUCTION_ID = 1,
    SUM_REDUCTION_ID = 2,
    TRIPLE_REDUCTION_ID = 3,
    MMS_REDUCTION_ID = 4,
  };
  enum SnapFieldID {
    FID_SINGLE = 0, // For field spaces with just one field
    // Fields for energy groups
    FID_GROUP_0 = FID_SINGLE,
    // ...
    FID_GROUP_MAX = FID_GROUP_0 + SNAP_MAX_ENERGY_GROUPS,
    FID_FLUX_START = FID_GROUP_MAX,
    FID_FLUX_MAX = FID_FLUX_START + 8/*corners*/*SNAP_MAX_ENERGY_GROUPS,
  };
#define SNAP_ENERGY_GROUP_FIELD(group)    \
  ((Snap::SnapFieldID)(Snap::FID_GROUP_0 + (group)))
#define SNAP_FLUX_GROUP_FIELD(group, corner)          \
  ((Snap::SnapFieldID)(Snap::FID_FLUX_START + (group * 8) + corner))
  enum SnapPartitionID {
    DISJOINT_PARTITION = 0,
  };
#define SNAP_XY_PROJECTION(forward)        \
  ((Snap::SnapProjectionID)(Snap::XY_PROJECTION + (forward ? 0 : 1)))
#define SNAP_YZ_PROJECTION(forward)        \
  ((Snap::SnapProjectionID)(Snap::YZ_PROJECTION + (forward ? 0 : 1)))
#define SNAP_XZ_PROJECTION(forward)        \
  ((Snap::SnapProjectionID)(Snap::XZ_PROJECTION + (forward ? 0 : 1)))
  enum SnapProjectionID {
    XY_PROJECTION = 1,
    // ...
    YZ_PROJECTION = XY_PROJECTION + 2,
    // ...
    XZ_PROJECTION = YZ_PROJECTION + 2,
  };
#define SNAP_SHARDING_ID        1
public:
  Snap(Context c, Runtime *rt)
    : ctx(c), runtime(rt) { }
public:
  inline const Rect<3>& get_simulation_bounds(void) const 
    { return simulation_bounds; }
  inline const IndexSpace<3>& get_launch_bounds(void) const
    { return launch_bounds; }
public:
  void setup(void);
  void transport_solve(void);
protected:
  void initialize_scattering(const SnapArray<1> &sigt, const SnapArray<1> &siga,
                             const SnapArray<1> &sigs, const SnapArray<2> &slgg) const;
  void initialize_velocity(const SnapArray<1> &vel, const SnapArray<1> &vdelt) const;
  void save_fluxes(const Predicate &pred, const SnapArray<3> &src, 
                   const SnapArray<3> &dst, int energy_group_chunks) const;
  void calculate_inner_source(const Predicate &pred, const SnapArray<3> &s_xs,
                              const SnapArray<3> &flux0, const SnapArray<3> &fluxm,
                              const SnapArray<3> &q2grp0, const SnapArray<3> &q2grpm,
                              const SnapArray<3> &qtot, int energy_group_chunks) const;
  void perform_sweeps(const Predicate &pred, const SnapArray<3> &flux,
                      const SnapArray<3> &fluxm, const SnapArray<3> &qtot, 
                      const SnapArray<1> &vdelt, const SnapArray<3> &dinv, 
                      const SnapArray<3> &t_xs, SnapArray<3> *time_flux_in[8], 
                      SnapArray<3> *time_flux_out[8], SnapArray<3> *qim[8],
                      const SnapArray<2> &flux_xy, const SnapArray<2> &flux_yz,
                      const SnapArray<2> &flux_xz, int energy_group_chunks) const;
  Predicate test_inner_convergence(const Predicate &pred, const SnapArray<3> &flux0,
                      const SnapArray<3> &flux0pi, const Future &pred_false_result,
                      int energy_group_chunks) const;
  Predicate test_outer_convergence(const Predicate &pred, const SnapArray<3> &flux0,
                      const SnapArray<3> &flux0po, const Future &inner_converged,
                      const Future &pred_false_result,
                      int energy_group_chunks) const;
private:
  const Context ctx;
  Runtime *const runtime;
private:
  // Simulation bounds
  Rect<3> simulation_bounds;
private:
  IndexSpace<3> simulation_is;
  IndexSpace<3> launch_bounds;
  IndexPartition<3> spatial_ip;
  IndexSpace<1> material_is;
  IndexSpace<2> slgg_is;
  IndexSpace<1> point_is;
  IndexSpace<2> xy_flux_is;
  IndexPartition<2> xy_flux_ip;
  IndexSpace<2> yz_flux_is;
  IndexPartition<2> yz_flux_ip;
  IndexSpace<2> xz_flux_is;
  IndexPartition<2> xz_flux_ip;
private:
  FieldSpace group_fs;
  FieldSpace flux_fs;
  FieldSpace moment_fs;
  FieldSpace flux_moment_fs;
  FieldSpace mat_fs;
  FieldSpace angle_fs;
public:
  static void snap_top_level_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime); 
public:
  static void parse_arguments(int argc, char **argv);
  static void compute_derived_globals(void);
  static void report_arguments(Runtime *runtime, Context ctx);
  static void perform_registrations(void);
  static void mapper_registration(Machine machine, Runtime *runtime,
                                  const std::set<Processor> &local_procs);
  static LayoutConstraintID get_soa_layout(void);
  static LayoutConstraintID get_reduction_layout(void);
public:
  // Configuration parameters read from input file
  static int num_dims; // originally ndimen 1-3
  static int nx_chunks; // originally ichunk 1 <= # <= nx
  static int ny_chunks; // originally npey 1 <= # <= ny
  static int nz_chunks; // originally npez 1 <= # <= nz
  static int nx; // 4 <= #
  static double lx; // 0.0 < #
  static int ny; // 4 <= #
  static double ly; // 0.0 < #
  static int nz; // 4 <= #
  static double lz; // 0.0 < #
  static int num_moments; // originally nmom 1 <= # <= 4
  static int num_angles; // originally nang 1 <= #
  static int num_groups; // originally ng 1 <= #
  static double convergence_eps; // originally epsi 0.0 < # < 1e-2
  static int max_inner_iters; // originally iitm 1 <= # 
  static int max_outer_iters; // originally oitm 1 <= #
  static bool time_dependent; // originally timedep
  static double total_sim_time; // originally tf 0.0 <= # if time dependent
  static int num_steps; // originally nsteps 1 <= #
  static MaterialLayout material_layout; // originally mat_opt
  static SourceLayout source_layout; // originally src_opt
  static bool dump_scatter; // originally scatp
  static bool dump_iteration; // originally it_dep
  static int dump_flux; // originally fluxp 0,1,2
  static bool flux_fixup; // originally fixup
  static bool dump_solution; // originally soloutp
  static int dump_kplane; // originally kplane 0,1,2
  static int dump_population;  // originally popout
  static bool minikba_sweep; // originally swp_typ
  static bool single_angle_copy; // originally angcpy
public: // derived
  static int num_corners; // orignally ncor
  static int nx_per_chunk;
  static int ny_per_chunk;
  static int nz_per_chunk;
public:
  static double dt; 
  static int cmom;
  static int num_octants;
  static double hi, hj, hk;
  static double *mu; // num angles
  static double *w; // num angles
  static double *wmu; // num angles
  static double *eta; // num angles
  static double *weta; // num angles
  static double *xi; // num angles
  static double *wxi; // num angles
  static double *ec; // num angles x num moments x num_octants
  static double *dinv; // num_angles x nx x ny x nz x 
  static int lma[4];
public:
  // Snap mapper derived from the default mapper
  class SnapMapper : public Legion::Mapping::DefaultMapper {
  public:
    typedef Legion::Mapping::MapperRuntime MapperRuntime;
    typedef Legion::Mapping::MapperContext MapperContext;
    typedef Legion::Mapping::PhysicalInstance PhysicalInstance;
    typedef Legion::VariantID VariantID;
    typedef Legion::LogicalRegion LogicalRegion;
    typedef Legion::AddressSpace AddressSpace;
  public:
    SnapMapper(MapperRuntime *rt, Machine machine, Processor local,
               const char *mapper_name);
  public:
    virtual void select_tunable_value(const MapperContext ctx,
                                      const Task& task,
                                      const SelectTunableInput& input,
                                            SelectTunableOutput& output);
    virtual void speculate(const MapperContext ctx,
                           const Copy &copy,
                                 SpeculativeOutput &output);
    virtual void map_copy(const MapperContext ctx,
                          const Copy &copy,
                          const MapCopyInput &input,
                                MapCopyOutput &output);
    virtual void select_task_options(const MapperContext ctx,
                                     const Task& task,
                                           TaskOptions& options);
    virtual void slice_task(const MapperContext ctx,
                            const Task &task,
                            const SliceTaskInput &input,
                                  SliceTaskOutput &output);
    virtual void speculate(const MapperContext ctx,
                           const Task &task,
                                 SpeculativeOutput &output);
    virtual void map_task(const MapperContext ctx,
                          const Task &task,
                          const MapTaskInput &input,
                                MapTaskOutput &output);
  public:
    virtual void select_sharding_functor(const MapperContext ctx,
                                         const Task &task,
                                         const SelectShardingFunctorInput &input,
                                               SelectShardingFunctorOutput &output);
    virtual void select_sharding_functor(const MapperContext ctx,
                                         const Copy &copy,
                                         const SelectShardingFunctorInput &input,
                                               SelectShardingFunctorOutput &output);
    virtual void select_sharding_functor(const MapperContext ctx,
                                         const Close &close,
                                         const SelectShardingFunctorInput &input,
                                               SelectShardingFunctorOutput &output);
    virtual void select_sharding_functor(const MapperContext ctx,
                                         const Partition &partition,
                                         const SelectShardingFunctorInput &input,
                                               SelectShardingFunctorOutput &output);
    virtual void select_sharding_functor(const MapperContext ctx,
                                         const Fill &fill,
                                         const SelectShardingFunctorInput &input,
                                               SelectShardingFunctorOutput &output);
  protected:
    void update_variants(const MapperContext ctx);
    void map_snap_array(const MapperContext ctx, 
                        LogicalRegion region, Memory target,
                        std::vector<PhysicalInstance> &instances);
#ifdef LOCAL_MAP_TASKS
  protected:
    Memory get_associated_sysmem(Processor proc);
    Memory get_associated_framebuffer(Processor proc);
    Memory get_associated_zerocopy(Processor proc);
    void get_associated_procs(Processor proc, std::vector<Processor> &procs);
#endif
  protected:
    bool has_variants;
    std::map<SnapTaskID,VariantID> cpu_variants;
    std::map<SnapTaskID,VariantID> gpu_variants;
  protected:
    Memory local_sysmem, local_zerocopy, local_framebuffer;
    std::map<std::pair<LogicalRegion,Memory>,PhysicalInstance> local_instances;
    // Copy instances always go in the system memory
    std::map<LogicalRegion,PhysicalInstance> copy_instances;
#ifdef LOCAL_MAP_TASKS
    std::map<Processor,Memory> associated_sysmems;
    std::map<Processor,Memory> associated_framebuffers;
    std::map<Processor,Memory> associated_zerocopy;
    std::map<Processor,std::vector<Processor> > associated_procs;
#endif
  protected:
    std::map<Point<3>,Processor> global_cpu_mapping;
    std::map<Point<3>,Processor> global_gpu_mapping;
  };
};

template<typename T, Snap::SnapTaskID TASK_ID, int DIM=3> 
class SnapTask : public IndexTaskLauncher {
public:
  SnapTask(const Snap &snap, const IndexSpace<DIM> &launch_domain, const Predicate &pred)
    : IndexTaskLauncher(TASK_ID, launch_domain, 
                        TaskArgument(), ArgumentMap(), pred) { }
  SnapTask(const Snap &snap, const Rect<DIM> &launch_space, const Predicate &pred)
    : IndexTaskLauncher(TASK_ID, launch_space, 
                        TaskArgument(), ArgumentMap(), pred) { }
public:
  void dispatch(Context ctx, Runtime *runtime, bool block = false)
  { 
    log_snap.info("Dispatching Task %s (ID %d)", 
        Snap::task_names[TASK_ID], TASK_ID);
    if (block) {
      FutureMap fm = runtime->execute_index_space(ctx, *this);
      fm.wait_all_results(true/*silence warnings*/);
    } else
      runtime->execute_index_space(ctx, *this);
  }
  template<typename OP>
  Future dispatch(Context ctx, Runtime *runtime, bool block = false)
  {
    log_snap.info("Dispatching Task %s (ID %d) with Reduction %d", 
                  Snap::task_names[TASK_ID], TASK_ID, OP::REDOP);
    if (block) {
      Future f = runtime->execute_index_space(ctx, *this, OP::REDOP);
      f.get_void_result(true/*silence warnings*/);
      return f;
    } else
      return runtime->execute_index_space(ctx, *this, OP::REDOP);
  }
public:
  static void preregister_all_variants(void)
  {
    T::preregister_cpu_variants();
    T::preregister_gpu_variants();
  }
  static void register_task_name(Runtime *runtime)
  {
    runtime->attach_name(TASK_ID, Snap::task_names[TASK_ID]);
  }
public:
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void snap_task_wrapper(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
  {
    log_snap.info("Running Task %s (UID %lld) on Processor " IDFMT "",
        task->get_task_name(), task->get_unique_id(), 
        runtime->get_executing_processor(ctx).id);
    (*TASK_PTR)(task, regions, ctx, runtime);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static RET_T snap_task_wrapper(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
  {
    log_snap.info("Running Task %s (UID %lld) on Processor " IDFMT "",
        task->get_task_name(), task->get_unique_id(), 
        runtime->get_executing_processor(ctx).id);
    RET_T result = (*TASK_PTR)(task, regions, ctx, runtime);
    return result;
  }
protected:
  // For registering CPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<
      SnapTask<T,TASK_ID>::template snap_task_wrapper<TASK_PTR> >(
          registrar, Snap::task_names[TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_cpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "CPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<RET_T,
      SnapTask<T,TASK_ID>::template snap_task_wrapper<RET_T,TASK_PTR> >(
                                         registrar, Snap::task_names[TASK_ID]);
  }
protected:
  // For registering GPU variants
  template<void (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "GPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<
      SnapTask<T,TASK_ID>::template snap_task_wrapper<TASK_PTR> >(
          registrar, Snap::task_names[TASK_ID]);
  }
  template<typename RET_T, RET_T (*TASK_PTR)(const Task*,
      const std::vector<PhysicalRegion>&, Context, Runtime*)>
  static void register_gpu_variant(const ExecutionConstraintSet &execution_constraints,
                                   const TaskLayoutConstraintSet &layout_constraints,
                                   bool leaf = false, bool inner = false)
  {
    char variant_name[128];
    strcpy(variant_name, "GPU ");
    strncat(variant_name, Snap::task_names[TASK_ID], 123);
    TaskVariantRegistrar registrar(TASK_ID, true/*global*/, variant_name);
    registrar.execution_constraints = execution_constraints;
    registrar.layout_constraints = layout_constraints;
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.leaf_variant = leaf;
    registrar.inner_variant = inner;
    Runtime::preregister_task_variant<RET_T,
      SnapTask<T,TASK_ID>::template snap_task_wrapper<RET_T,TASK_PTR> >(
                                         registrar, Snap::task_names[TASK_ID]);
  }
};

template<int DIM>
class SnapArray {
public:
  SnapArray(IndexSpace<DIM> is, IndexPartition<DIM> ip, FieldSpace fs, 
            Context ctx, Runtime *runtime, const char *name);
  ~SnapArray(void);
private:
  SnapArray(const SnapArray &rhs);
  SnapArray& operator=(const SnapArray &rhs);
public:
  inline LogicalRegion<DIM> get_region(void) const { return lr; }
  inline LogicalPartition<DIM> get_partition(void) const { return lp; }
  inline const std::set<FieldID>& get_all_fields(void) const 
    { return all_fields; }
  LogicalRegion<DIM> get_subregion(const Point<DIM> &color) const;
public:
  void initialize(Predicate pred = Predicate::TRUE_PRED) const; 
  template<typename T>
  void initialize(T value, Predicate pred = Predicate::TRUE_PRED) const;
  PhysicalRegion map(void) const;
  void unmap(const PhysicalRegion &region) const;
public:
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv,
                                         T &launcher) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, 0/*proj id*/,
                                                      priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields = all_fields;
  }
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv, T &launcher,
                        Snap::SnapFieldID field, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, priv,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_projection_requirement(PrivilegeMode priv, T &launcher,
   const std::vector<Snap::SnapFieldID> &fields, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, priv,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(
                                        fields.begin(), fields.end());
  }
  template<typename T>
  inline void add_projection_requirement(T &launcher, Snap::SnapReductionID reduction,
      Snap::SnapFieldID field, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, reduction,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_projection_requirement(T &launcher, Snap::SnapReductionID reduction,
      const std::vector<Snap::SnapFieldID> &fields, ProjectionID proj_id = 0) const
  {
    launcher.add_region_requirement(RegionRequirement(lp, proj_id, reduction,
                                                      EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(
                                        fields.begin(), fields.end());
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv,
                                     T &launcher) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields = all_fields;
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv, T &launcher,
                                     Snap::SnapFieldID field) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(field);
  }
  template<typename T>
  inline void add_region_requirement(PrivilegeMode priv, T &launcher,
                  const std::vector<Snap::SnapFieldID> &fields) const
  {
    launcher.add_region_requirement(RegionRequirement(lr, priv, EXCLUSIVE, lr));
    launcher.region_requirements.back().privilege_fields.insert(
                                                  fields.begin(), fields.end());
  }
protected:
  const Context ctx;
  Runtime *const runtime;
protected:
  LogicalRegion<DIM> lr;
  LogicalPartition<DIM> lp;
  std::set<FieldID> all_fields;
  Rect<DIM> color_space;
  mutable std::map<Point<DIM>,LogicalRegion<DIM> > subregions;
  void *fill_buffer;
  size_t field_size;
};

class FluxProjectionFunctor : public ProjectionFunctor {
public:
  FluxProjectionFunctor(Snap::SnapProjectionID kind, const bool forward);
public:
  virtual Legion::LogicalRegion project(const Mappable *mappable, unsigned index,
                                Legion::LogicalRegion upper_bound,
                                const Legion::DomainPoint &point);
  virtual Legion::LogicalRegion project(const Mappable *mappable, unsigned index,
                                Legion::LogicalPartition upper_bound,
                                const Legion::DomainPoint &point);
  virtual Legion::LogicalRegion project(Legion::LogicalRegion upper_bound, 
                                const Legion::DomainPoint &point,
                                const Legion::Domain &launch_domain);
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                const Legion::DomainPoint &point,
                                const Legion::Domain &launch_domain);
  virtual void invert(Legion::LogicalRegion region, Legion::LogicalPartition upper,
                      const Legion::Domain &launch_domain,
                      std::vector<Legion::DomainPoint> &ordered_points);
  virtual unsigned get_depth(void) const { return 0; }
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual bool is_invertible(void) const { return true; }
public:
  const Snap::SnapProjectionID projection_kind;
  const bool forward;
};

class SnapShardingFunctor : public Legion::ShardingFunctor {
public:
  SnapShardingFunctor(int x_chunks, int y_chunks, int z_chunks);
public:
  size_t linearize_point(const Point<3> &point) const;
public:
  virtual ShardID shard(const Legion::DomainPoint &point,
                        const Legion::Domain &full_space,
                        const size_t total_shards);
public:
  const int x_chunks, y_chunks, z_chunks;
};

class AndReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::AND_REDUCTION_ID;
public:
  typedef bool LHS;
  typedef bool RHS;
  static const bool identity = true;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class SumReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::SUM_REDUCTION_ID;
public:
  typedef double LHS;
  typedef double RHS;
  static const double identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

class TripleReduction {
public:
  static const Snap::SnapReductionID REDOP = Snap::TRIPLE_REDUCTION_ID;
public:
  typedef MomentTriple LHS;
  typedef MomentTriple RHS;
  static const MomentTriple identity;
public:
  template<bool EXCLUSIVE> static void apply(LHS &lhs, RHS rhs);
  template<bool EXCLUSIVE> static void fold(RHS &rhs1, RHS rhs2);
};

#endif // __SNAP_H__

