//===----------------------------------------------------------------------===//
// See the LICENSE file for license and copyright information.
// SPDX-License-Identifier: NCSA AND BSD-3-Clause
//===----------------------------------------------------------------------===//
// Regression tests for stiff thermal relaxation through the public C API.
// These tests use a fine-stepped GS reference; the implicit solver under test
// is never used to construct its own equilibrium or transient reference.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <grackle.h>
#include "grtestutils/utils.hpp"
#include "chemistry_solver_funcs.hpp"

namespace {

#define TEST_FIELDS(F) \
  F(density) F(internal_energy) F(x_velocity) F(y_velocity) F(z_velocity) \
  F(metal_density) F(dust_density) F(e_density) F(HI_density) F(HII_density) \
  F(HeI_density) F(HeII_density) F(HeIII_density) F(HM_density) \
  F(H2I_density) F(H2II_density) F(DI_density) F(DII_density) F(HDI_density) \
  F(DM_density) F(HDII_density) F(HeHII_density) \
  F(CI_density) F(CII_density) F(OI_density) F(OII_density) F(CHI_density) \
  F(CH2I_density) F(COI_density) F(COII_density) F(CO2I_density) \
  F(O2I_density) F(O2II_density) F(OHI_density) F(OHII_density) \
  F(H2OI_density) F(H2OII_density) F(H3OII_density) F(SiI_density) \
  F(SiOI_density) F(SiO2I_density) F(Mg_density) F(Al_density) F(S_density) \
  F(Fe_density) F(MgSiO3_dust_density) F(AC_dust_density) F(SiM_dust_density) \
  F(FeM_dust_density) F(Mg2SiO4_dust_density) F(Fe3O4_dust_density) \
  F(SiO2_dust_density) F(MgO_dust_density) F(FeS_dust_density) \
  F(Al2O3_dust_density) F(ref_org_dust_density) F(vol_org_dust_density) \
  F(H2O_ice_dust_density) F(H2_self_shielding_length) \
  F(H2_custom_shielding_factor) F(isrf_habing) F(temperature_floor) \
  F(specific_heating_rate) F(volumetric_heating_rate)

struct Grid {
  grackle_field_data fields{};
  std::array<int, 3> dims, start, end;
  std::map<std::string, std::vector<gr_float>> values;
  explicit Grid(std::array<int, 3> d = {1, 1, 1}, int ghost = 0)
    : dims(d), start{ghost, ghost, ghost},
      end{d[0]-ghost-1, d[1]-ghost-1, d[2]-ghost-1} {
    gr_initialize_field_data(&fields);
    fields.grid_rank = 3;
    fields.grid_dimension = dims.data();
    fields.grid_start = start.data();
    fields.grid_end = end.data();
    fields.grid_dx = 1.e-3;
    const int n = size();
#define ALLOCATE(name) values[#name] = std::vector<gr_float>(n, 0.); fields.name = values[#name].data();
    TEST_FIELDS(ALLOCATE)
#undef ALLOCATE
  }
  Grid(const Grid&) = delete;
  Grid& operator=(const Grid&) = delete;
  int size() const { return dims[0]*dims[1]*dims[2]; }
  int index(int x, int y, int z) const { return x+dims[0]*(y+dims[1]*z); }
  bool active(int x, int y, int z) const {
    return x>=start[0] && x<=end[0] && y>=start[1] && y<=end[1] &&
           z>=start[2] && z<=end[2];
  }
  void restore(const std::map<std::string, std::vector<gr_float>>& saved) {
    for (const auto& [name, data] : saved)
      std::copy(data.begin(), data.end(), values.at(name).begin());
  }
  void copy_cell(const Grid& other, int from, int to) {
    for (auto& [name, data] : values) data[to] = other.values.at(name)[from];
  }
};

struct Config {
  chemistry_data chemistry{};
  chemistry_data_storage rates{};
  code_units units{};
  bool initialized = false;
  Config(int primordial, bool uv = true, bool dust = false,
         bool metals = false, bool metal_network = false, bool shield = false) {
    set_default_chemistry_parameters(&chemistry);
    chemistry.use_grackle = 1;
    chemistry.with_radiative_cooling = 1;
    chemistry.primordial_chemistry = primordial;
    chemistry.UVbackground = uv;
    chemistry.metal_cooling = metals;
    chemistry.metal_chemistry = metal_network;
    chemistry.dust_chemistry = dust;
    chemistry.use_dust_density_field = dust;
    chemistry.H2_self_shielding = shield ? 2 : 0;
    chemistry.self_shielding_method = shield ? 1 : 0;
    chemistry.cmb_temperature_floor = 1;
    chemistry.solver_method = 1;
    chemistry.max_iterations = 10000;
    chemistry.exit_after_iterations_exceeded = 1;
    if (primordial > 0) chemistry.HydrogenFractionByMass = .76;
    grtest::set_standard_datafile(chemistry, uv ?
        (shield ? "CloudyData_UVB=HM2012_shielded.h5" :
                  "CloudyData_UVB=HM2012.h5") : "CloudyData_noUVB.h5");
    units.comoving_coordinates = 0;
    units.a_value = units.a_units = 1.;
    units.density_units = 1.67262171e-24;
    units.length_units = 3.08567758e24;
    units.time_units = 3.15576e13;  // Myr
    set_velocity_units(&units);
  }
  ~Config() {
    if (initialized) local_free_chemistry_data(&chemistry, &rates);
  }
  bool data_available() const {
    return std::ifstream(chemistry.grackle_data_file).good();
  }
  bool initialize() {
    initialized = local_initialize_chemistry_data(&chemistry, &rates, &units)
                  == GR_SUCCESS;
    return initialized;
  }
  bool solve(Grid& grid, double dt) {
    return local_solve_chemistry(&chemistry, &rates, &units, &grid.fields, dt)
           == GR_SUCCESS;
  }
  bool temperatures(Grid& grid, std::vector<gr_float>& out) {
    out.resize(grid.size());
    return local_calculate_temperature(&chemistry, &rates, &units,
                                        &grid.fields, out.data()) == GR_SUCCESS;
  }
  double temperature(Grid& grid) {
    std::vector<gr_float> out;
    return temperatures(grid, out) ? out[0] : std::numeric_limits<double>::quiet_NaN();
  }
  double cooling_time(Grid& grid) {
    std::vector<gr_float> out(grid.size());
    if (local_calculate_cooling_time(&chemistry, &rates, &units, &grid.fields,
                                     out.data()) != GR_SUCCESS)
      return std::numeric_limits<double>::quiet_NaN();
    return out[0];
  }
  // The derivative is evaluated with all species held fixed.
  double relaxation_time(Grid& grid) {
    const double energy = grid.fields.internal_energy[0], eps = 1.e-3;
    grid.fields.internal_energy[0] = energy*(1.+eps);
    const double plus = grid.fields.internal_energy[0]/cooling_time(grid);
    grid.fields.internal_energy[0] = energy*(1.-eps);
    const double minus = grid.fields.internal_energy[0]/cooling_time(grid);
    grid.fields.internal_energy[0] = energy;
    return 1./std::abs((plus-minus)/(2.*eps*energy));
  }
  void initialize_cell(Grid& g, int i, double nH, double T, double ion = .1) {
    const double X = chemistry.HydrogenFractionByMass;
    auto& v = g.values;
    const double rho = nH/X;
    const double metal = chemistry.metal_cooling ? .01*chemistry.SolarMetalFractionByMass*rho : 0.;
    const double gas = rho-metal;
    const double hydrogen = X*gas, helium = (1.-X)*gas;
    for (auto& [name, data] : v) data[i] = 0.;
    v["density"][i] = rho;
    v["metal_density"][i] = metal;
    v["dust_density"][i] = chemistry.dust_chemistry ? .001*metal : 0.;
    v["HII_density"][i] = ion*hydrogen;
    v["HeII_density"][i] = .1*ion*helium;
    v["HeIII_density"][i] = .01*ion*helium;
    v["HeI_density"][i] = helium-v["HeII_density"][i]-v["HeIII_density"][i];
    const double seed_floor = 1.e-20*rho;
    if (chemistry.primordial_chemistry>1) {
      v["H2I_density"][i] = 1.e-6*hydrogen;
      v["HM_density"][i] = v["H2II_density"][i] = seed_floor;
    }
    if (chemistry.primordial_chemistry>2) {
      v["DI_density"][i] = chemistry.DeuteriumToHydrogenRatio*hydrogen;
      v["DII_density"][i] = v["HDI_density"][i] = seed_floor;
    }
    if (chemistry.primordial_chemistry>3) {
      v["DM_density"][i] = v["HDII_density"][i] = v["HeHII_density"][i] = seed_floor;
    }
    v["HI_density"][i] = hydrogen-v["HII_density"][i]-v["H2I_density"][i]-
      v["HM_density"][i]-v["H2II_density"][i]-v["HDI_density"][i]/3.-
      v["HDII_density"][i]/3.-v["HeHII_density"][i]/5.;
    if (chemistry.metal_chemistry) {
      // A finite, neutral initial metal mixture; the initialization test does
      // not use this approximate mixture as an elemental reference answer.
      v["CI_density"][i] = .2*metal;
      v["OI_density"][i] = .5*metal;
      v["SiI_density"][i] = .1*metal;
      for (const char* name : {"CII_density","OII_density","CHI_density",
           "CH2I_density","COI_density","COII_density","CO2I_density",
           "O2I_density","O2II_density","OHI_density","OHII_density",
           "H2OI_density","H2OII_density","H3OII_density","SiOI_density",
           "SiO2I_density"}) v[name][i] = seed_floor;
    }
    v["e_density"][i] = v["HII_density"][i]+v["HeII_density"][i]/4.+
      v["HeIII_density"][i]/2.+v["H2II_density"][i]/2.-v["HM_density"][i]+
      v["DII_density"][i]/2.-v["DM_density"][i]/2.+v["HDII_density"][i]/3.+
      v["HeHII_density"][i]/5.;
    v["H2_self_shielding_length"][i] = 1.e-3;
    v["H2_custom_shielding_factor"][i] = 1.;
    v["isrf_habing"][i] = chemistry.interstellar_radiation_field;
    v["temperature_floor"][i] = 2.725;
    const double particles = v["HI_density"][i]+v["HII_density"][i]+
      v["e_density"][i]+helium/4.+v["HM_density"][i]+
      (v["H2I_density"][i]+v["H2II_density"][i])/2.;
    v["internal_energy"][i] = T*particles/rho/get_temperature_units(&units)/
                              (chemistry.Gamma-1.);
  }
};

void expect_physical(const Grid& g, const Config& cfg) {
  for (int z=g.start[2]; z<=g.end[2]; ++z)
    for (int y=g.start[1]; y<=g.end[1]; ++y)
      for (int x=g.start[0]; x<=g.end[0]; ++x) {
        const int i = g.index(x,y,z);
        for (const auto& [name, v] : g.values) {
          EXPECT_TRUE(std::isfinite(v[i])) << name << " cell " << i;
          EXPECT_GE(v[i], 0.) << name << " cell " << i;
        }
        EXPECT_GT(g.fields.internal_energy[i], 0.);
        if (cfg.chemistry.primordial_chemistry < 1 || cfg.chemistry.metal_chemistry)
          continue;
        const auto& f = g.fields;
        const double H = f.HI_density[i]+f.HII_density[i]+f.HM_density[i]+
          f.H2I_density[i]+f.H2II_density[i]+f.HDI_density[i]/3.+
          f.HDII_density[i]/3.+f.HeHII_density[i]/5.;
        const double He = f.HeI_density[i]+f.HeII_density[i]+f.HeIII_density[i]+
          4.*f.HeHII_density[i]/5.;
        const double gas = f.density[i]-f.metal_density[i];
        // D and HeH+ share nuclei with the atomic species. The end-of-call
        // projection is sequential, so allow roundoff plus a small trace term.
        const double field_eps=std::numeric_limits<gr_float>::epsilon();
        EXPECT_NEAR(H/gas, cfg.chemistry.HydrogenFractionByMass, std::max(2.e-8,32.*field_eps));
        EXPECT_NEAR(He/gas, 1.-cfg.chemistry.HydrogenFractionByMass, std::max(2.e-8,32.*field_eps));
        const double charge = f.HII_density[i]+f.HeII_density[i]/4.+
          f.HeIII_density[i]/2.+f.H2II_density[i]/2.-f.HM_density[i]+
          f.DII_density[i]/2.-f.DM_density[i]/2.+f.HDII_density[i]/3.+
          f.HeHII_density[i]/5.;
        EXPECT_NEAR(f.e_density[i]/gas, charge/gas, std::max(1.e-12,16.*field_eps));
      }
}

// Explicit GS remains the independent fine-step reference. Re-evaluate tau
// periodically and require both temperature and meaningful abundances to settle.
bool equilibrium(Config& cfg, Grid& g) {
  cfg.chemistry.solver_method = 2;
  auto previous = g.values;
  double dt = .1*std::min(cfg.relaxation_time(g), std::abs(cfg.cooling_time(g)));
  if (!std::isfinite(dt) || dt<=0.) return false;
  for (int step=1; step<=30000; ++step) {
    if (!cfg.solve(g, dt)) return false;
    if (step%100 != 0) continue;
    double change = 0.;
    for (const auto& [name, values] : g.values) {
      if (name=="internal_energy" || (name.find("_density")!=std::string::npos &&
          name!="metal_density" && name!="dust_density")) {
        const double floor = name=="internal_energy" ? 1.e-30 :
                             1.e-8*g.fields.density[0];
        change = std::max(change, std::abs(values[0]-previous[name][0])/
                                   std::max(std::abs(double(values[0])), floor));
      }
    }
    if (step>=500 && change<1.e-7) return true;
    previous = g.values;
    dt = .1*std::min(cfg.relaxation_time(g), std::abs(cfg.cooling_time(g)));
    if (!std::isfinite(dt) || dt<=0.) return false;
  }
  return false;
}

// Test the reaction RHS itself: end-of-call projection must not hide a
// missing product or photoelectron in the implicit system.
TEST(AdaptiveSpeciesRhs, PhotoReactionsConserveNucleiAndCharge) {
  using namespace grackle::impl;
  Config cfg(4, true);
  Grid g;
  g.fields.HI_density[0]=.7; g.fields.HII_density[0]=.2;
  g.fields.HeI_density[0]=.1; g.fields.HeII_density[0]=.02;
  g.fields.HeIII_density[0]=.01; g.fields.e_density[0]=.21;
  g.fields.HM_density[0]=.03; g.fields.H2I_density[0]=.04;
  g.fields.H2II_density[0]=.05; g.fields.DI_density[0]=.006;
  g.fields.DII_density[0]=.003; g.fields.HDI_density[0]=.001;
  g.fields.DM_density[0]=.0001; g.fields.HDII_density[0]=.0002;
  g.fields.HeHII_density[0]=.0003;
  std::array<double,rxn_rate_buf_detail::MAX_LEN> storage{};
  FullRxnRateBuf rates{};
  for(int i=0;i<rxn_rate_buf_detail::MAX_LEN;++i) rates.data[i]=&storage[i];
  const double rhoH=1.;
  const gr_mask_type metal_mask=MASK_FALSE;
  for(int rate : {PhotoRxnLUT::k24,PhotoRxnLUT::k25,PhotoRxnLUT::k26,
                 PhotoRxnLUT::k27,PhotoRxnLUT::k28,PhotoRxnLUT::k29,
                 PhotoRxnLUT::k30,PhotoRxnLUT::kdissHDI}) {
    SCOPED_TRACE(testing::Message() << "photo rate id=" << rate);
    storage.fill(0.);
    FullRxnRateBuf_kph_bufs(&rates)[rate][0]=1.;
    std::array<double,MAX_EVOLVED_SPECIES_FIELDS> derivatives{};
    View<double**> out(derivatives.data(),MAX_EVOLVED_SPECIES_FIELDS,1);
    chemistry::species_density_derivatives_0d(out,MASK_FALSE,&rhoH,&metal_mask,
                                             &cfg.chemistry,&g.fields,rates);
    const auto d=[&](int species){return derivatives[species];};
    const double H=d(SpLUT::HI)+d(SpLUT::HII)+d(SpLUT::HM)+
      d(SpLUT::H2I)+d(SpLUT::H2II)+(d(SpLUT::HDI)+d(SpLUT::HDII))/3.+
      d(SpLUT::HeHII)/5.;
    const double He=d(SpLUT::HeI)+d(SpLUT::HeII)+d(SpLUT::HeIII)+
      4.*d(SpLUT::HeHII)/5.;
    const double D=d(SpLUT::DI)+d(SpLUT::DII)+d(SpLUT::DM)+
      2.*(d(SpLUT::HDI)+d(SpLUT::HDII))/3.;
    const double charge=d(SpLUT::HII)+d(SpLUT::HeII)/4.+
      d(SpLUT::HeIII)/2.+d(SpLUT::H2II)/2.-d(SpLUT::HM)+
      d(SpLUT::DII)/2.-d(SpLUT::DM)/2.+d(SpLUT::HDII)/3.+
      d(SpLUT::HeHII)/5.-d(SpLUT::e);
    EXPECT_NEAR(H,0.,1.e-14);
    EXPECT_NEAR(He,0.,1.e-14);
    EXPECT_NEAR(D,0.,1.e-14);
    EXPECT_NEAR(charge,0.,1.e-14);
  }
}

class AdaptivePhotoEquilibrium : public testing::TestWithParam<int> {};
TEST_P(AdaptivePhotoEquilibrium, LongStepsApproachFineStepEquilibrium) {
  const int primordial = GetParam();
  Config cfg(primordial, true, false, primordial==0);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  cfg.initialize_cell(cell, 0, 1., 1.e4);
  ASSERT_TRUE(equilibrium(cfg, cell)) << "fine-step equilibrium for prim=" << primordial;
  const auto eq = cell.values;
  const double Teq = cfg.temperature(cell), tau = cfg.relaxation_time(cell);
  ASSERT_GT(tau, 0.);
  ASSERT_TRUE(std::isfinite(tau));
  for (double kick : {.7, 1., 1.3}) {
    auto initial = eq;
    initial["internal_energy"][0] *= kick;
    for (double ratio : {10., 1.e3, 1.e5}) {
      SCOPED_TRACE(testing::Message() << "prim=" << primordial << " kick=" << kick
                                    << " dt/tau=" << ratio);
      double expected_T = Teq;
      auto expected = eq;
      if (ratio==10.) {
        cell.restore(initial);
        cfg.chemistry.solver_method = 2;
        for (int step=0; step<500; ++step) ASSERT_TRUE(cfg.solve(cell, .02*tau));
        expected_T = cfg.temperature(cell);
        expected = cell.values;
      }
      cell.restore(initial);
      cfg.chemistry.solver_method = 1;
      // The old explicit path exhausts this cap for a 1e5-tau perturbation.
      cfg.chemistry.max_iterations = 512;
      ASSERT_TRUE(cfg.solve(cell, ratio*tau));
      const double T = cfg.temperature(cell);
      EXPECT_NEAR(T/expected_T, 1., ratio==10. ? .04 : .005);
      if (primordial>0) {
        const double relative_tolerance=ratio==10. ? .04 : .005;
        const double trace_floor=1.e-8*cell.fields.density[0];
        for(const char* name : {"HI_density","HII_density","HeI_density",
                               "HeII_density","HeIII_density","e_density"})
          EXPECT_NEAR(cell.values.at(name)[0],expected.at(name)[0],
                      relative_tolerance*std::abs(expected.at(name)[0])+
                      trace_floor) << name;
        if(primordial>1)
          for(const char* name : {"HM_density","H2I_density","H2II_density"})
            EXPECT_NEAR(cell.values.at(name)[0],expected.at(name)[0],
                        relative_tolerance*std::abs(expected.at(name)[0])+
                        trace_floor) << name;
      }
      expect_physical(cell, cfg);
    }
  }
}
INSTANTIATE_TEST_SUITE_P(Primordial, AdaptivePhotoEquilibrium,
                         testing::Values(0,1,2));

struct SmokeCase {
  int primordial;
  bool uv, dust, metals, metal_network, shielding;
};
class AdaptiveChemistryCoverage : public testing::TestWithParam<SmokeCase> {};
TEST_P(AdaptiveChemistryCoverage, FinitePositiveAcrossSupportedNetworks) {
  const auto p=GetParam();
  Config cfg(p.primordial,p.uv,p.dust,p.metals,p.metal_network,p.shielding);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  cfg.initialize_cell(cell,0,100.,3000.);
  cfg.chemistry.solver_method = p.primordial==0 ? 1 : 3;
  const double dt=.02*std::abs(cfg.cooling_time(cell));
  ASSERT_TRUE(std::isfinite(dt));
  ASSERT_GT(dt,0.);
  ASSERT_TRUE(cfg.solve(cell,dt));
  expect_physical(cell,cfg);
}
INSTANTIATE_TEST_SUITE_P(Networks,AdaptiveChemistryCoverage,testing::Values(
  SmokeCase{0,true,false,true,false,false},
  SmokeCase{1,true,false,false,false,false},
  SmokeCase{2,false,false,false,false,false},
  SmokeCase{3,true,false,false,false,false},
  SmokeCase{4,true,false,false,false,true},
  SmokeCase{2,true,true,true,false,false},
  SmokeCase{4,false,false,true,true,false},
  SmokeCase{4,true,true,true,true,true}));

TEST(AdaptiveSolverApi, InvalidAndZeroTimestepsDoNotMutateFields) {
  Config cfg(2,false);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  cfg.initialize_cell(cell,0,1.,1.e4);
  const auto initial=cell.values;
  for(double dt : {-1.,std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(testing::Message() << "dt=" << dt);
    EXPECT_FALSE(cfg.solve(cell,dt));
    EXPECT_EQ(cell.values,initial);
  }
  EXPECT_TRUE(cfg.solve(cell,0.));
  EXPECT_EQ(cell.values,initial);
}

TEST(AdaptiveSolverApi, FrozenEnergyRemainsUnchanged) {
  Config cfg(1,false);
  cfg.chemistry.with_radiative_cooling=0;
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  cfg.initialize_cell(cell,0,1.,1.e4,.5);
  const double energy=cell.fields.internal_energy[0];
  const double initial_HII=cell.fields.HII_density[0];
  cfg.chemistry.solver_method=1;
  ASSERT_TRUE(cfg.solve(cell,.1));
  EXPECT_EQ(cell.fields.internal_energy[0],energy);
  EXPECT_NE(cell.fields.HII_density[0],initial_HII);
  expect_physical(cell,cfg);
}

TEST(AdaptiveSolverApi, FailedImplicitRhsRestoresComovingDensityUnits) {
  Config cfg(2,false);
  cfg.chemistry.solver_method=3;
  cfg.chemistry.use_specific_heating_rate=1;
  cfg.units.comoving_coordinates=1;
  cfg.units.a_value=.25;
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  cfg.initialize_cell(cell,0,1.,1.e4);
  // Force an invalid RHS after the conversion into proper density units.
  // This checks bounded failure and unit restoration, not whole-call rollback:
  // other cells/subcycles could already have completed in a larger grid.
  cell.fields.specific_heating_rate[0]=std::numeric_limits<gr_float>::quiet_NaN();
  const auto initial=cell.values;
  EXPECT_FALSE(cfg.solve(cell,1.));
  for(const auto& [name,values]:cell.values)
    if(name=="density" || name.find("_density")!=std::string::npos)
      EXPECT_EQ(values,initial.at(name)) << name;
}

TEST(AdaptiveSolverApi, ColdNeutralEvolutionIsIndependentOfCodeUnits) {
  Config standard(1,true,true,true), extreme(1,true,true,true);
  extreme.units.length_units=1.;
  extreme.units.time_units=1.e12;
  set_velocity_units(&extreme.units);
  for(Config* cfg : {&standard,&extreme}) {
    cfg->chemistry.use_dust_density_field=0;
    cfg->chemistry.use_isrf_field=1;
    cfg->chemistry.max_iterations=1000;
  }
  if (!standard.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(standard.initialize());
  ASSERT_TRUE(extreme.initialize());
  Grid first, second;
  const auto initialize_neutral=[](Config& cfg, Grid& cell) {
    cfg.initialize_cell(cell,0,.76,1000.,1.e-20);
    const double old_gas=cell.fields.density[0]-cell.fields.metal_density[0];
    cell.fields.metal_density[0]=cfg.chemistry.SolarMetalFractionByMass*
                                cell.fields.density[0];
    const double new_gas=cell.fields.density[0]-cell.fields.metal_density[0];
    for(const char* name : {"HI_density","HII_density","HeI_density",
                           "HeII_density","HeIII_density","e_density"})
      cell.values.at(name)[0]*=new_gas/old_gas;
    cell.fields.internal_energy[0]=1000./get_temperature_units(&cfg.units);
  };
  initialize_neutral(standard,first);
  initialize_neutral(extreme,second);
  EXPECT_NEAR(standard.temperature(first)/extreme.temperature(second),1.,1.e-6);
  // The second representation has energy ~8e34 while species are O(1).
  // An unscaled Newton matrix stalls for this ordinary physical state.
  const double physical_dt=3.15e13;
  ASSERT_TRUE(standard.solve(first,physical_dt/standard.units.time_units));
  ASSERT_TRUE(extreme.solve(second,physical_dt/extreme.units.time_units));
  EXPECT_NEAR(standard.temperature(first)/extreme.temperature(second),1.,.01);
  for(const char* name : {"HI_density","HII_density","HeI_density",
                         "HeII_density","HeIII_density","e_density"})
    EXPECT_NEAR(first.values.at(name)[0],second.values.at(name)[0],
                .01*std::abs(first.values.at(name)[0])+1.e-8) << name;
  expect_physical(first,standard);
  expect_physical(second,extreme);
}

struct RegimeCase {
  double nH, temperature;
  bool uv;
};
void PrintTo(const RegimeCase& p, std::ostream* out) {
  *out << "nH=" << p.nH << ", T=" << p.temperature << ", UV=" << p.uv;
}
std::vector<RegimeCase> regime_cases() {
  std::vector<RegimeCase> out;
  for (double nH : {1.e-2,1.,1.e8,1.e12})
    for (double T : {10.,1.e4,1.e7})
      for (bool uv : {false,true}) out.push_back({nH,T,uv});
  return out;
}
class AdaptiveThermalRegimes : public testing::TestWithParam<RegimeCase> {};
TEST_P(AdaptiveThermalRegimes, DefaultSolverHandlesDensityAndTemperatureRange) {
  const auto p=GetParam();
  SCOPED_TRACE(testing::Message() << "nH=" << p.nH << " T=" << p.temperature
                                << " UV=" << p.uv);
  Config cfg(2,p.uv);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  cfg.initialize_cell(cell,0,p.nH,p.temperature);
  // Correct the seed energy for the actual temperature-dependent molecular
  // gamma. Preserve the deliberately non-equilibrium species state.
  for (int correction=0; correction<6; ++correction) {
    const double T=cfg.temperature(cell);
    ASSERT_GT(T,0.);
    ASSERT_TRUE(std::isfinite(T));
    cell.fields.internal_energy[0]*=p.temperature/T;
  }
  EXPECT_NEAR(cfg.temperature(cell)/p.temperature,1.,
              std::max(1.e-10,16.*std::numeric_limits<gr_float>::epsilon()));
  const double dt=.01*std::abs(cfg.cooling_time(cell));
  ASSERT_GT(dt,0.);
  ASSERT_TRUE(std::isfinite(dt));
  cfg.chemistry.solver_method=1;
  ASSERT_TRUE(cfg.solve(cell,dt)) << "dt=" << dt;
  expect_physical(cell,cfg);
}
INSTANTIATE_TEST_SUITE_P(DensityTemperature,AdaptiveThermalRegimes,
                         testing::ValuesIn(regime_cases()));

TEST(AdaptiveCoolingOptions, DenseMolecularCoolingOptions) {
  Config cfg(2,false);
  cfg.chemistry.cie_cooling=1;
  cfg.chemistry.h2_optical_depth_approximation=1;
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid cell;
  const double nH=1.e12, target_T=2000.;
  cfg.initialize_cell(cell,0,nH,target_T,1.e-8);
  const double added=.5*nH-cell.fields.H2I_density[0];
  cell.fields.H2I_density[0]+=added;
  cell.fields.HI_density[0]-=added;
  for(int correction=0;correction<12;++correction)
    cell.fields.internal_energy[0]*=target_T/cfg.temperature(cell);
  const double dt=.01*std::abs(cfg.cooling_time(cell));
  ASSERT_GT(dt,0.);
  ASSERT_TRUE(std::isfinite(dt));
  ASSERT_TRUE(cfg.solve(cell,dt));
  expect_physical(cell,cfg);
}

TEST(AdaptiveSolverGrid, MixedCellsAgreeWithIsolatedCallsAndPreserveGhosts) {
  Config cfg(2,true);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  Grid scalar;
  cfg.initialize_cell(scalar,0,1.,1.e4);
  ASSERT_TRUE(equilibrium(cfg,scalar));
  const auto eq=scalar.values;
  const double tau=cfg.relaxation_time(scalar);
  Grid batch({5,4,4},1);
  for(auto& [name,values]:batch.values)
    for(int i=0;i<batch.size();++i) values[i]=17.+.001*i;
  std::vector<int> active;
  for(int z=batch.start[2];z<=batch.end[2];++z)
    for(int y=batch.start[1];y<=batch.end[1];++y)
      for(int x=batch.start[0];x<=batch.end[0];++x) {
        const int i=batch.index(x,y,z);
        scalar.restore(eq);
        const double factor=(active.size()%3==0) ? .7 :
                            (active.size()%3==1) ? 1. : 1.3;
        scalar.fields.internal_energy[0]*=factor;
        batch.copy_cell(scalar,0,i);
        active.push_back(i);
      }
  const auto before=batch.values;
  cfg.chemistry.solver_method=1;
  cfg.chemistry.max_iterations=512;
  ASSERT_TRUE(cfg.solve(batch,1.e4*tau));
  expect_physical(batch,cfg);
  Grid original({5,4,4},1);
  original.restore(before);
  for(int i:active) {
    scalar.copy_cell(original,i,0);
    ASSERT_TRUE(cfg.solve(scalar,1.e4*tau));
    for(const auto& [name,values]:batch.values) {
      const double want=scalar.values.at(name)[0];
      EXPECT_NEAR(values[i],want,
                  std::max(2.e-7,32.*std::numeric_limits<gr_float>::epsilon())*
                  std::max(std::abs(want),1.e-12))
        << name << " active cell " << i;
    }
  }
  for(int z=0;z<batch.dims[2];++z) for(int y=0;y<batch.dims[1];++y)
    for(int x=0;x<batch.dims[0];++x) if(!batch.active(x,y,z)) {
      const int i=batch.index(x,y,z);
      for(const auto& [name,values]:batch.values)
        EXPECT_EQ(values[i],before.at(name)[i]) << name << " ghost cell " << i;
    }
}

// Fully ionized 1e6 K gas with no molecules; 2% of the helium is still
// ionizing, so the cell moves between GS and NR within one call. Returns the
// hybrid's T over a 2000-call GS reference for a call of `ratio` cooling times.
double hot_ionized_hybrid_over_reference(Config& cfg, double ratio) {
  Grid cell;
  cfg.initialize_cell(cell,0,1.,1.e6,0.);
  auto& v=cell.values;
  const double gas=v["density"][0]-v["metal_density"][0];
  const double H=cfg.chemistry.HydrogenFractionByMass*gas, He=gas-H;
  v["HII_density"][0]=.999*H;
  v["H2I_density"][0]=0.;  // molecule-free hot gas; this triggers the bug
  v["HI_density"][0]=H-v["HII_density"][0]-v["H2I_density"][0]-
                     v["HM_density"][0]-v["H2II_density"][0];
  v["HeII_density"][0]=.01*He;
  v["HeIII_density"][0]=.98*He;
  v["HeI_density"][0]=He-v["HeII_density"][0]-v["HeIII_density"][0];
  v["e_density"][0]=v["HII_density"][0]+v["HeII_density"][0]/4.+
    v["HeIII_density"][0]/2.+v["H2II_density"][0]/2.-v["HM_density"][0];
  for (int c=0;c<8;++c) cell.fields.internal_energy[0]*=1.e6/cfg.temperature(cell);
  const auto initial=cell.values;
  const double dt=ratio*std::abs(cfg.cooling_time(cell));
  if (!(std::isfinite(dt) && dt>0.)) return std::numeric_limits<double>::quiet_NaN();
  cfg.chemistry.solver_method=2;  // independent fine-stepped reference
  for (int step=0; step<2000; ++step)
    if (!cfg.solve(cell,dt/2000.)) return std::numeric_limits<double>::quiet_NaN();
  const double T_ref=cfg.temperature(cell);
  cell.restore(initial);
  cfg.chemistry.solver_method=1;
  if (!cfg.solve(cell,dt)) return std::numeric_limits<double>::quiet_NaN();
  expect_physical(cell,cfg);
  return cfg.temperature(cell)/T_ref;
}

// GS must not limit its step with change rates stored before NR substeps
// (that made the hybrid ~10% cold here while either scheme alone was within 1%).
TEST(AdaptiveSolverHybrid, GaussSeidelAfterNewtonUsesFreshRateHistory) {
  Config cfg(2,false);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  EXPECT_NEAR(hot_ionized_hybrid_over_reference(cfg,10.),1.,.02);
}

// Once a cell has used NR it must finish the call on NR: the stiffness test
// fails near the end of every call, and a final GS step was 3% cold here.
TEST(AdaptiveSolverHybrid, CellStaysOnNewtonUntilCallEnds) {
  Config cfg(2,false);
  if (!cfg.data_available()) GTEST_SKIP() << "Cloudy data file unavailable";
  ASSERT_TRUE(cfg.initialize());
  EXPECT_NEAR(hot_ionized_hybrid_over_reference(cfg,100.),1.,.01);
}

#undef TEST_FIELDS
}  // namespace
