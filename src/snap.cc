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
#include "outer.h"
#include "inner.h"

#include <cstdio>

LegionRuntime::Logger::Category log_snap("snap");

const char* Snap::task_names[LAST_TASK_ID] = { SNAP_TASK_NAMES };

//------------------------------------------------------------------------------
void Snap::setup(void)
//------------------------------------------------------------------------------
{
  // This is the index space for all our regions
  const int upper[3] = { nx-1, ny-1, nz-1 };
  simulation_bounds = Rect<3>(Point<3>::ZEROES(), Point<3>(upper));
  simulation_is = 
    runtime->create_index_space(ctx, Domain::from_rect<3>(simulation_bounds));
  // Create the disjoint partition of the index space 
  const int chunks[3] = { nx_chunks-1, ny_chunks-1, nz_chunks-1 };
  launch_bounds = Rect<3>(Point<3>::ZEROES(), Point<3>(chunks)); 
  const int bf[3] = { nx_chunks, ny_chunks, nz_chunks };
  Point<3> blocking_factor(bf);
  Blockify<3> spatial_map(blocking_factor);
  spatial_ip = 
    runtime->create_index_partition(ctx, simulation_is,
                                    spatial_map, DISJOINT_PARTITION);
  // Create the ghost partitions for each subregion

  // Make some of our other field spaces
  const int nmat = (material_layout == HOMOGENEOUS_LAYOUT) ? 1 : 2;
  material_is = runtime->create_index_space(ctx,
      Domain::from_rect<1>(Rect<1>(Point<1>(0), Point<1>(nmat-1))));
  const int slgg_upper[2] = { nmat-1, num_groups-1 };
  Rect<2> slgg_bounds(Point<2>::ZEROES(), Point<2>(slgg_upper));
  slgg_is = runtime->create_index_space(ctx, Domain::from_rect<2>(slgg_bounds));
  // Make a field space for all the energy groups
  group_fs = runtime->create_field_space(ctx); 
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, group_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> group_fields(num_groups);
    for (unsigned idx = 0; idx < num_groups; idx++)
      group_fields[idx] = FID_GROUP_0 + idx;
    std::vector<size_t> group_sizes(num_groups, sizeof(double));
    allocator.allocate_fields(group_sizes, group_fields);
  }
  // Make a fields space for the moments for each energy group
  moment_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, moment_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> moment_fields(num_groups);
    for (unsigned idx = 0; idx < num_groups; idx++)
      moment_fields[idx] = FID_GROUP_0 + idx;
    // Notice that the field size is as big as necessary to store all moments
    std::vector<size_t> moment_sizes(num_groups, num_moments*sizeof(double));
    allocator.allocate_fields(moment_sizes, moment_fields);
  }
  flux_moment_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, flux_moment_fs);
    assert(num_groups <= SNAP_MAX_ENERGY_GROUPS);
    std::vector<FieldID> moment_fields(num_groups);
    for (unsigned idx = 0; idx < num_groups; idx++)
      moment_fields[idx] = FID_GROUP_0 + idx;
    // Storing number of moments - 1
    std::vector<size_t> moment_sizes(num_groups, 
                                      (num_moments-1)*sizeof(double));
    allocator.allocate_fields(moment_sizes, moment_fields);
  }
  mat_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, mat_fs);
    allocator.allocate_field(sizeof(int), FID_SINGLE);
  }
}

//------------------------------------------------------------------------------
void Snap::transport_solve(void)
//------------------------------------------------------------------------------
{
  // Use a tunable variable to decide how far ahead the outer loop gets
  Future outer_runahead_future = 
    runtime->select_tunable_value(ctx, OUTER_RUNAHEAD_TUNABLE);
  // Same thing with the inner loop
  Future inner_runahead_future = 
    runtime->select_tunable_value(ctx, INNER_RUNAHEAD_TUNABLE);

  // Create our important arrays
  SnapArray flux0(simulation_is, spatial_ip, group_fs, ctx, runtime);
  SnapArray flux0po(simulation_is,spatial_ip, group_fs, ctx, runtime);
  SnapArray flux0pi(simulation_is, spatial_ip, group_fs, ctx, runtime);
  SnapArray fluxm(simulation_is, spatial_ip, flux_moment_fs, ctx, runtime);

  SnapArray qi(simulation_is, spatial_ip, group_fs, ctx, runtime);
  SnapArray q2grp0(simulation_is, spatial_ip, group_fs, ctx, runtime);
  SnapArray q2grpm(simulation_is, spatial_ip, moment_fs, ctx, runtime);
  SnapArray qtot(simulation_is, spatial_ip, moment_fs, ctx, runtime);

  SnapArray mat(simulation_is, spatial_ip, mat_fs, ctx, runtime);
  SnapArray slgg(slgg_is, IndexPartition::NO_PART, moment_fs, ctx, runtime);
  SnapArray s_xs(simulation_is, spatial_ip, moment_fs, ctx, runtime);

  // Initialize our data
  flux0.initialize();
  flux0po.initialize();
  flux0pi.initialize();
  if (num_moments > 1)
    fluxm.initialize();

  qi.initialize();
  q2grp0.initialize();
  q2grpm.initialize();
  qtot.initialize();

  mat.initialize<int>(1);
  slgg.initialize();
  s_xs.initialize();

  // Tunables should be ready by now
  const unsigned outer_runahead = outer_runahead_future.get_result<unsigned>();
  assert(outer_runahead > 0);
  const unsigned inner_runahead = inner_runahead_future.get_result<unsigned>();
  assert(inner_runahead > 0);
  // Loop over time steps
  std::deque<Future> outer_converged_tests;
  std::deque<Future> inner_converged_tests;
  // Iterate over time steps
  for (unsigned cy = 0; cy < num_steps; ++cy)
  {
    outer_converged_tests.clear();
    Predicate outer_pred = Predicate::TRUE_PRED;
    // The outer solve loop    
    for (unsigned otno = 0; otno < max_outer_iters; ++otno)
    {
      // Do the outer source calculation 
      CalcOuterSource outer_src(*this, outer_pred, qi, 
                                slgg, mat, q2grp0, q2grpm);
      outer_src.dispatch(ctx, runtime);
      // Save the fluxes
      save_fluxes(outer_pred, flux0, flux0po);
      // Do the inner solve
      inner_converged_tests.clear();
      Predicate inner_pred = Predicate::TRUE_PRED;
      // The inner solve loop
      for (unsigned inno=0; inno < max_inner_iters; ++inno)
      {
        // Do the inner source calculation
        CalcInnerSource inner_src(*this, inner_pred, s_xs, flux0, fluxm,
                                  q2grp0, q2grpm, qtot);
        inner_src.dispatch(ctx, runtime);
        // Save the fluxes
        save_fluxes(inner_pred, flux0, flux0pi);
        // Perform the sweeps

        // Test for inner convergence
        TestInnerConvergence inner_conv(*this, inner_pred, flux0, flux0pi);
        Future inner_converged = 
          inner_conv.dispatch<AndReduction>(ctx, runtime);
        inner_converged_tests.push_back(inner_converged);
        // Update the next predicate
        Predicate converged = runtime->create_predicate(ctx, inner_converged);
        inner_pred = runtime->predicate_not(ctx, converged);
        // See if we've run far enough ahead
        if (inner_converged_tests.size() == inner_runahead)
        {
          Future f = inner_converged_tests.front();
          inner_converged_tests.pop_front();
          if (f.get_result<bool>())
            break;
        }
      }
      // Test for outer convergence
      TestOuterConvergence outer_conv(*this, outer_pred, flux0, flux0po);
      Future outer_converged = 
        outer_conv.dispatch<AndReduction>(ctx, runtime);
      outer_converged_tests.push_back(outer_converged);
      // Update the next predicate
      Predicate converged = runtime->create_predicate(ctx, outer_converged);
      outer_pred = runtime->predicate_not(ctx, converged);
      // See if we've run far enough ahead
      if (outer_converged_tests.size() == outer_runahead)
      {
        Future f = outer_converged_tests.front();
        outer_converged_tests.pop_front();
        if (f.get_result<bool>())
          break;
      }
    }
  }
}

//------------------------------------------------------------------------------
void Snap::output(void)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
void Snap::save_fluxes(const Predicate &pred, 
                       const SnapArray &src, const SnapArray &dst) const
//------------------------------------------------------------------------------
{
  // Build the CopyLauncher
  CopyLauncher launcher(pred);
  launcher.add_copy_requirements(
      RegionRequirement(LogicalRegion::NO_REGION, READ_ONLY, 
                        EXCLUSIVE, src.get_region()),
      RegionRequirement(LogicalRegion::NO_REGION, WRITE_DISCARD,
                        EXCLUSIVE, dst.get_region()));
  const std::set<FieldID> &src_fields = src.get_all_fields();
  RegionRequirement &src_req = launcher.src_requirements.back();
  src_req.privilege_fields = src_fields;
  src_req.instance_fields.insert(src_req.instance_fields.end(),
      src_fields.begin(), src_fields.end());
  const std::set<FieldID> &dst_fields = dst.get_all_fields();
  RegionRequirement &dst_req = launcher.dst_requirements.back();
  dst_req.privilege_fields = dst_fields;
  dst_req.instance_fields.insert(dst_req.instance_fields.end(),
      dst_fields.begin(), dst_fields.end());
  // Iterate over the sub-regions and issue copies for each separately
  for (GenericPointInRectIterator<3> color_it(launch_bounds); 
        color_it; color_it++)   
  {
    DomainPoint dp = DomainPoint::from_point<3>(color_it.p);
    src_req.region = src.get_subregion(dp);
    dst_req.region = dst.get_subregion(dp);
    runtime->issue_copy_operation(ctx, launcher);
  }
}

//------------------------------------------------------------------------------
/*static*/ void Snap::snap_top_level_task(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  printf("Welcome to Legion-SNAP!\n");
  report_arguments();
  Snap snap(ctx, runtime); 
  snap.setup();
  snap.transport_solve();
  snap.output();
}

static void skip_line(FILE *f)
{
  char buffer[80];
  fgets(buffer, 79, f);
}

static void read_int(FILE *f, const char *name, int &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  fscanf(f, format, &result);
}

static void read_bool(FILE *f, const char *name, bool &result)
{
  char format[80];
  sprintf(format,"  %s=%%d", name);
  int temp = 0;
  fscanf(f, format, &temp);
  result = (temp != 0);
}

static void read_double(FILE *f, const char *name, double &result)
{
  char format[80];
  sprintf(format,"  %s=%%lf", name);
  fscanf(f, format, &result);
}

int Snap::num_dims = 1;
int Snap::nx_chunks = 4;
int Snap::ny_chunks = 1;
int Snap::nz_chunks = 1;
int Snap::nx = 4;
double Snap::lx = 1.0;
int Snap::ny = 1;
double Snap::ly = 1.0;
int Snap::nz = 1;
double Snap::lz = 1.0;
int Snap::num_moments = 1;
int Snap::num_angles = 1;
int Snap::num_groups = 1;
double Snap::convergence_eps = 1e-4;
int Snap::max_inner_iters = 5;
int Snap::max_outer_iters = 100;
bool Snap::time_dependent = false;
double Snap::total_sim_time = 0.0;
int Snap::num_steps = 1;
Snap::MaterialLayout Snap::material_layout = HOMOGENEOUS_LAYOUT;
Snap::SourceLayout Snap::source_layout = EVERYWHERE_SOURCE; 
bool Snap::dump_scatter = false;
bool Snap::dump_iteration = false;
int Snap::dump_flux = 0;
bool Snap::flux_fixup = false;
bool Snap::dump_solution = false;
int Snap::dump_kplane = 0;
int Snap::dump_population = 0;
bool Snap::minikba_sweep = false;
bool Snap::single_angle_copy = true;

//------------------------------------------------------------------------------
/*static*/ void Snap::parse_arguments(int argc, char **argv)
//------------------------------------------------------------------------------
{
  if (argc < 2) {
    printf("No input file specified. Exiting.\n");
    exit(1);
  }
  printf("Reading input file %s\n", argv[1]);
  FILE *f = fopen(argv[1], "r");
  int dummy_int = 0;
  skip_line(f);
  skip_line(f);
  read_int(f, "npey", ny_chunks);
  read_int(f, "npez", nz_chunks);
  read_int(f, "ichunk", nx_chunks);
  read_int(f, "nthreads", dummy_int);
  read_int(f, "nnested", dummy_int);
  read_int(f, "ndimen", num_dims);
  read_int(f, "nx", nx);
  read_double(f, "lx", lx);
  read_int(f, "ny", ny);
  read_double(f, "ly", ly);
  read_int(f, "nz", nz);
  read_double(f, "lz", lz);
  read_int(f, "nmom", num_moments);
  read_int(f, "nang", num_angles);
  read_int(f, "ng", num_groups);
  read_double(f, "epsi", convergence_eps);
  read_int(f, "iitm", max_inner_iters);
  read_int(f, "oitm", max_outer_iters);
  read_bool(f, "timedep", time_dependent);
  read_double(f, "tf", total_sim_time);
  read_int(f, "nsteps", num_steps);
  int mat_opt = 0;
  read_int(f, "mat_opt", mat_opt);
  switch (mat_opt)
  {
    case 0:
      {
        material_layout = HOMOGENEOUS_LAYOUT;
        break;
      }
    case 1:
      {
        material_layout = CENTER_LAYOUT;
        break;
      }
    case 2:
      {
        material_layout = CORNER_LAYOUT;
        break;
      }
    default:
      assert(false);
  }
  int src_opt = 0;
  read_int(f, "src_opt", src_opt);
  switch (src_opt)
  {
    case 0:
      {
        source_layout = EVERYWHERE_SOURCE;
        break;
      }
    case 1:
      {
        source_layout = CENTER_SOURCE;
        break;
      }
    case 2:
      {
        source_layout = CORNER_SOURCE;
        break;
      }
    case 3:
      {
        source_layout = MMS_SOURCE;
        break;
      }
    default:
      assert(false);
  }
  read_bool(f, "scatp", dump_scatter);
  read_bool(f, "it_det", dump_iteration);
  read_int(f, "fluxp", dump_flux);
  read_bool(f, "fixup", flux_fixup);
  read_bool(f, "solutp", dump_solution);
  read_int(f, "kplane", dump_kplane);
  read_int(f, "popout", dump_population);
  read_bool(f, "swp_typ", minikba_sweep);
  read_bool(f, "angcpy", single_angle_copy);
  fclose(f);
  // Check all the conditions
  assert((1 <= num_dims) && (num_dims <= 3));
  assert((1 <= nx_chunks) && (nx_chunks <= nx));
  assert((1 <= ny_chunks) && (ny_chunks <= ny));
  assert((1 <= nz_chunks) && (nz_chunks <= nz));
  assert(4 <= nx);
  assert(0.0 < lx);
  assert(4 <= ny);
  assert(0.0 < ly);
  assert(4 <= ny);
  assert(0.0 < lz);
  assert((1 <= num_moments) && (num_moments <= 4));
  assert(1 <= num_angles);
  assert(1 <= num_groups);
  assert((0.0 < convergence_eps) && (convergence_eps < 1e-2));
  assert(1 <= max_inner_iters);
  assert(1 <= max_outer_iters);
  if (time_dependent)
    assert(0.0 <= total_sim_time);
  assert(1 <= num_steps);
}

//------------------------------------------------------------------------------
/*static*/ void Snap::report_arguments(void)
//------------------------------------------------------------------------------
{
  printf("Dimensions: %d\n", num_dims);
  printf("X-Chunks: %d\n", nx_chunks);
  printf("Y-Chunks: %d\n", ny_chunks);
  printf("Z-Chunks: %d\n", nz_chunks);
  printf("nx,ny,nz: %d,%d,%d\n", nx, ny, nz);
  printf("lx,ly,lz: %.8g,%.8g,%.8g\n", lx, ly, lz);
  printf("Moments: %d\n", num_moments);
  printf("Angles: %d\n", num_angles);
  printf("Groups: %d\n", num_groups);
  printf("Convergence: %.8g\n", convergence_eps);
  printf("Max Inner Iterations: %d\n", max_inner_iters);
  printf("Max Outer Iterations: %d\n", max_outer_iters);
  printf("Time Dependent: %s\n", (time_dependent ? "Yes" : "No"));
  printf("Total Simulation Time: %.8g\n", total_sim_time);
  printf("Total Steps: %d\n", num_steps);
  printf("Material Layout: %s\n", ((material_layout == HOMOGENEOUS_LAYOUT) ? 
        "Homogenous Layout" : (material_layout == CENTER_LAYOUT) ? 
        "Center Layout" : "Corner Layout"));
  printf("Source Layout: %s\n", ((source_layout == EVERYWHERE_SOURCE) ? 
        "Everywhere" : (source_layout == CENTER_SOURCE) ? "Center" :
        (source_layout == CORNER_SOURCE) ? "Corner" : "MMS"));
  printf("Dump Scatter: %s\n", dump_scatter ? "Yes" : "No");
  printf("Dump Iteration: %s\n", dump_iteration ? "Yes" : "No");
  printf("Dump Flux: %s\n", (dump_flux == 2) ? "All" : 
      (dump_flux == 1) ? "Scalar" : "No");
  printf("Fixup Flux: %s\n", flux_fixup ? "Yes" : "No");
  printf("Dump Solution: %s\n", dump_solution ? "Yes" : "No");
  printf("Dump Kplane: %d\n", dump_kplane);
  printf("Dump Population: %s\n", (dump_population == 2) ? "All" : 
      (dump_population == 1) ? "Final" : "No");
  printf("Mini-KBA Sweep: %s\n", minikba_sweep ? "Yes" : "No");
  printf("Single Angle Copy: %s\n", single_angle_copy ? "Yes" : "No");
}

//------------------------------------------------------------------------------
/*static*/ void Snap::register_task_variants(void)
//------------------------------------------------------------------------------
{
  TaskVariantRegistrar registrar(SNAP_TOP_LEVEL_TASK_ID, "snap_main_variant");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
  Runtime::preregister_task_variant<snap_top_level_task>(registrar,
                          Snap::task_names[SNAP_TOP_LEVEL_TASK_ID]);
  Runtime::set_top_level_task_id(SNAP_TOP_LEVEL_TASK_ID);
  Runtime::set_registration_callback(mapper_registration);
  // Now register all the task variants
  CalcOuterSource::preregister_all_variants();
  TestOuterConvergence::preregister_all_variants();
  CalcInnerSource::preregister_all_variants();
  TestInnerConvergence::preregister_all_variants();
  // Finally register our reduction operators
  Runtime::register_reduction_op<AndReduction>(AndReduction::REDOP_ID);
}

//------------------------------------------------------------------------------
/*static*/ void Snap::mapper_registration(Machine machine, Runtime *runtime,
                                         const std::set<Processor> &local_procs)
//------------------------------------------------------------------------------
{
  MapperRuntime *mapper_rt = runtime->get_mapper_runtime();
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    runtime->replace_default_mapper(new SnapMapper(mapper_rt, machine, 
                                                   *it, "SNAP Mapper"), *it);
  }
}

//------------------------------------------------------------------------------
Snap::SnapMapper::SnapMapper(MapperRuntime *rt, Machine machine, 
                             Processor local, const char *mapper_name)
  : DefaultMapper(rt, machine, local, mapper_name)
//------------------------------------------------------------------------------
{
}

//------------------------------------------------------------------------------
void Snap::SnapMapper::select_tunable_value(const MapperContext ctx,
                                            const Task& task,
                                            const SelectTunableInput& input,
                                                  SelectTunableOutput& output)
//------------------------------------------------------------------------------
{
  switch (input.tunable_id)
  {
    case OUTER_RUNAHEAD_TUNABLE:
      {
        runtime->pack_tunable<unsigned>(4, output);
        break;
      }
    case INNER_RUNAHEAD_TUNABLE:
      {
        runtime->pack_tunable<unsigned>(8, output);
        break;
      }
    default:
      // Fall back to the default mapper
      DefaultMapper::select_tunable_value(ctx, task, input, output);
  }
}

//------------------------------------------------------------------------------
SnapArray::SnapArray(IndexSpace is, IndexPartition ip, FieldSpace fs,
                     Context c, Runtime *rt)
  : ctx(c), runtime(rt)
//------------------------------------------------------------------------------
{
  lr = runtime->create_logical_region(ctx, is, fs);
  if (ip.exists())
    lp = runtime->get_logical_partition(lr, ip);
  runtime->get_field_space_fields(fs, all_fields);
}

//------------------------------------------------------------------------------
SnapArray::SnapArray(const SnapArray &rhs)
  : ctx(rhs.ctx), runtime(rhs.runtime)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
}

//------------------------------------------------------------------------------
SnapArray::~SnapArray(void)
//------------------------------------------------------------------------------
{
  runtime->destroy_logical_region(ctx, lr);
}

//------------------------------------------------------------------------------
SnapArray& SnapArray::operator=(const SnapArray &rhs)
//------------------------------------------------------------------------------
{
  // should never be called
  assert(false);
  return *this;
}

//------------------------------------------------------------------------------
LogicalRegion SnapArray::get_subregion(const DomainPoint &color) const
//------------------------------------------------------------------------------
{
  assert(lp.exists());
  // See if we already cached the result
  std::map<DomainPoint,LogicalRegion>::const_iterator finder = 
    subregions.find(color);
  if (finder != subregions.end())
    return finder->second;
  LogicalRegion result = runtime->get_logical_subregion_by_color(lp, color);
  // Save the result for later
  subregions[color] = result;
  return result;
}

//------------------------------------------------------------------------------
void SnapArray::initialize(void)
//------------------------------------------------------------------------------
{
  assert(!all_fields.empty());
  // Assume all the fields are the same size
  size_t field_size = runtime->get_field_size(lr.get_field_space(),
                                              *(all_fields.begin()));
  void *buffer = malloc(field_size);
  memset(buffer, 0, field_size);
  FillLauncher launcher(lr, lr, TaskArgument(buffer, field_size));
  launcher.fields = all_fields;
  runtime->fill_fields(ctx, launcher);
  free(buffer);
}

//------------------------------------------------------------------------------
template<typename T>
void SnapArray::initialize(T value)
//------------------------------------------------------------------------------
{
  FillLauncher launcher(lr, lr, TaskArgument(&value, sizeof(value)));
  launcher.fields = all_fields;
  runtime->fill_fields(ctx, launcher);
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::apply<true>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way 
  if (!rhs)
    lhs = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way 
  if (!rhs)
    lhs = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::fold<true>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way
  if (!rhs2)
    rhs1 = false;
}

//------------------------------------------------------------------------------
template<>
/*static*/ void AndReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  // This is monotonic so no need for synchronization either way
  if (!rhs2)
    rhs1 = false;
}

