/*
  COMP328 ASSIGNMENT
  (c) mkbane, University of Liverpool (2021)


  You are provided with this serial code.  Your assignment is to make
  this code go faster subject to following the formal definition of
  your assignment. You should add necessary error handling and
  your final code should work on any number of parallel processing
  elements.

  This code is a simplistic model of aerosol processes as described in a
  forthcoming book by Topping & Bane (Wiley).
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <mpi.h>

#include <immintrin.h>


double liquid_mass=2.0, gas_mass=0.3, k=0.00001;

int init(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);
double calc_system_energy(double, double*, double*, double*, int);
void output_particles(double*, double*, double*, double*, double*, double*, double*, double*, int);
void calc_centre_mass(double*, double*, double*, double*, double*, double, int);

void swap_ptr(double**, double**);

int numRanks, rankId, rankSimdLoad, lastRankLoad, toSend, rankOffset, rankLimit;

int main(int argc, char* argv[]) {

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

  int i, j;
  int num;     // user defined (argv[1]) total number of gas molecules in simulation
  int time, timesteps; // for time stepping, including user defined (argv[2]) number of timesteps to integrate
  int rc;      // return code
  double *mass, *x, *y, *z, *vx, *vy, *vz;  // 1D array for mass, (x,y,z) position, (vx, vy, vz) velocity
  double dx, dy, dz, d, F, GRAVCONST=0.001, T=300;
  double ax, ay, az;
  double *gas, *liquid, *loss_rate;       // 1D array for each particle's component that will evaporate
  double *old_x, *old_y, *old_z, *old_mass;            // save previous values whilst doing global updates
  double totalMass, systemEnergy;  // for stats

  int *counts, *displacements; // for MPI variable collective comms.

  double start;
  if (rankId == 0)
    start=omp_get_wtime();   // we make use of the simple wall clock timer available in OpenMP

  /* if avail, input size of system */
  if (argc > 1 ) {
    num = atoi(argv[1]);
    timesteps = atoi(argv[2]);
  }
  else {
    num = 20000;
    timesteps = 50;
  }

  if (rankId == 0)
    printf("Initializing for %d particles in x,y,z space...", num);

  /* malloc arrays and pass ref to init(). NOTE: init() uses random numbers */
  mass = (double *) malloc(num * sizeof(double));
  x =  (double *) malloc(num * sizeof(double));
  y =  (double *) malloc(num * sizeof(double));
  z =  (double *) malloc(num * sizeof(double));
  vx = (double *) malloc(num * sizeof(double));
  vy = (double *) malloc(num * sizeof(double));
  vz = (double *) malloc(num * sizeof(double));
  gas = (double *) malloc(num * sizeof(double));
  liquid = (double *) malloc(num * sizeof(double));
  loss_rate = (double *) malloc(num * sizeof(double));
  old_x = (double *) malloc(num * sizeof(double));
  old_y = (double *) malloc(num * sizeof(double));
  old_z = (double *) malloc(num * sizeof(double));
  old_mass = (double *) malloc(num * sizeof(double));
  counts = (int *) malloc(numRanks * sizeof(int));
  displacements = (int *) malloc(numRanks * sizeof(int));

  if (rankId == 0)
  {
    // should check all rc but let's just see if last malloc worked
    if (old_mass == NULL) {
      printf("\n ERROR in malloc for (at least) old_mass - aborting\n");
      return -99;
    }
    else {
      printf("  (malloc-ed)  ");
    }
  }

  // Domain decomposition for MPI processes.
  // Each rank (but last) gets the nearest SIMD multiple of work to do.
  int rankSimdLoad = (num / numRanks) & (-8);
  // Last rank will have more if not perfect SIMD multiple.
  int lastRankLoad = num - rankSimdLoad * (numRanks-1);
  int toSend = (rankId == (numRanks-1)) ? lastRankLoad : rankSimdLoad;
  // Each rank gets the interval [rankOffset, rankLimit) of particles to compute.
  int rankOffset = rankSimdLoad * rankId;
  int rankLimit = (rankId == (numRanks-1)) ? num : (rankOffset + rankSimdLoad);
  // Counts and displacement arrays used for varaible sized comms.
  for (i=0; i<numRanks-1; ++i) counts[i] = rankSimdLoad;
  counts[numRanks - 1] = lastRankLoad;
  for (i=0; i<numRanks; ++i) displacements[i] = rankSimdLoad * i;

  // Initialise on all nodes (computing will be faster than transfer).
  rc = init(mass, x, y, z, vx, vy, vz, gas, liquid, loss_rate, num);

  if (rankId == 0) {
    if (rc != 0) {
      printf("\n ERROR during init() - aborting\n");
      return -99;
    }
    else {
      printf("  INIT COMPLETE\n");
    }
  }

  // Expand constants into AVX-512 form.
  __m512d vec_GRAVCONST = _mm512_set1_pd(GRAVCONST);
  __m512d vec_gas_mass = _mm512_set1_pd(gas_mass);
  __m512d vec_liquid_mass = _mm512_set1_pd(liquid_mass);
  __m512d vec_totalMass = _mm512_set1_pd(0.0);

  // Calculate mass & do a (+=) reduction into totalMass.
  // Each rank gets the interval [rankOffset, rankLimit) of particles to compute.
  // First each MPI rank does AVX3 vec -> double. Then do MPI_REDUCE into one double.
  for (i = rankOffset; i < rankLimit; i += 8) {
    // Load mass elements if i<num, else set to 0.
    __m512i vec_i = _mm512_setr_epi64(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
    __mmask8 i_mask = _mm512_cmp_epi64_mask(vec_i, _mm512_set1_epi64(rankLimit), 1); // OP: 1 is LT
    __m512d vec_gas = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, gas + i);
    __m512d vec_liquid = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, liquid + i);

    // mass[i] = gas[i]*gas_mass + liquid[i]*liquid_mass;
    // Will have 0 for elements i >= num.
    __m512d vec_mass = _mm512_fmadd_pd(vec_gas, vec_gas_mass,
                                       _mm512_mul_pd(vec_liquid, vec_liquid_mass));
    vec_totalMass = _mm512_add_pd(vec_totalMass, vec_mass);
    // Store back to mem.
    _mm512_mask_store_pd(mass + i, i_mask, vec_mass);
  }

  // Wait for threads to join and do AVX->double reduction on a single thread.
  totalMass = _mm512_reduce_add_pd(vec_totalMass);
  double root_totalMass;

  // Wait for all processes to calculate their local totalMass, before reducing into root.
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&totalMass, &root_totalMass, 1,    // for all local totalMass: totalMass += local totalMass
             MPI_DOUBLE, MPI_SUM,               // it's a SUM(double, double) op
             0, MPI_COMM_WORLD);                // receive to root process (rank=0)

  if (rankId == 0) {
    systemEnergy = calc_system_energy(root_totalMass, vx, vy, vz, num);
    printf("Time 0. Total mass=%g\n", root_totalMass);
    printf("Time 0. System energy=%g\n", systemEnergy);

    printf("Now to integrate for %d timesteps\n", timesteps);
  }

  /*
     MAIN TIME STEPPING LOOP

     We 'save' old (entry to timestep loop) values to use on RHS of:

     For each aerosol particle we will: calc forces due to other
     particles & update change in velocity and thus position; then we
     condense some of the gas to liquid which changes the mass of the
     particule so we then determine its new velocity via conservation
     of kinetic energy.

     Having looped over the particles (using 'old' values on the right
     hand side as a crude approximation to real life) we then have
     updated all particles independently.

     We then determine the total mass & the system energy of the
     system.

     The final statement of each time-stepping loop is to decrease the
     temperature of the system, T.

     After completing all time-steps, we output the the time taken and
     the centre of mass of the final system configuration.

  */


  // time=0 was initial conditions
  for (time=1; time<=timesteps; time++) {
    // Each rank has to have all x, y, z and mass values from the last timestep.
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgatherv(x + rankOffset, toSend, MPI_DOUBLE,
                   old_x, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(y + rankOffset, toSend, MPI_DOUBLE,
                   old_y, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(z + rankOffset, toSend, MPI_DOUBLE,
                   old_z, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(mass + rankOffset, toSend, MPI_DOUBLE,
                   old_mass, counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);

    // The exp func has to be done just once per timestep.
    __m512d vec_exp_kT = _mm512_set1_pd(exp(-k*T));

    // LOOP2: update position etc per particle (based on old data)
    // Each rank gets the interval [rankOffset, rankLimit) of particles to compute.
    for(i=rankOffset; i<rankLimit; i += 8) {
      __m512d vec_old_x, vec_old_y, vec_old_z, vec_old_mass;

      // Set i_mask back for element i, if i<num. (1: OP := _MM_CMPINT_LT)
      __m512i vec_i = _mm512_setr_epi64(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
      __mmask8 i_mask = _mm512_cmp_epi64_mask(vec_i, _mm512_set1_epi64(rankLimit), 1);

      // Load into AVX-512 registers. (only i<num elemenets are loaded, else the lane is set to 0.0).
      __m512d vec_x = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, x + i);
      __m512d vec_y = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, y + i);
      __m512d vec_z = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, z + i);
      __m512d vec_vx = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, vx + i);
      __m512d vec_vy = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, vy + i);
      __m512d vec_vz = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, vz + i);
      __m512d vec_mass = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, mass + i);
      __m512d vec_gas = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, gas + i);
      __m512d vec_loss_rate = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, loss_rate + i);
      __m512d vec_liquid = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, liquid + i);

      // calc forces on body i due to particles (j != i)
      for (j=0; j<num; j += 1) {
        // if (i == j), then  don't set j writeback bit.
        __m512i vec_j = _mm512_set1_epi64(j);
        __mmask8 j_mask = _mm512_cmp_epi64_mask(vec_i, vec_j, 4); // 4: OP := _MM_CMPINT_NE

        // Load into AVX-512 registers.
        vec_old_x = _mm512_set1_pd(old_x[j]);
        vec_old_y = _mm512_set1_pd(old_y[j]);
        vec_old_z = _mm512_set1_pd(old_z[j]);
        vec_old_mass = _mm512_set1_pd(old_mass[j]);

        __m512d vec_dx = _mm512_sub_pd(vec_old_x, vec_x);
        __m512d vec_dy = _mm512_sub_pd(vec_old_y, vec_y);
        __m512d vec_dz = _mm512_sub_pd(vec_old_z, vec_z);

        // The expression "a*a + b*b + c*c" can be done using 1 mul and 2 fma ops:
        //  fma(a, a, fma(b, b, mul(c, c))
        __m512d vec_temp_d = _mm512_max_pd(_mm512_fmadd_pd(vec_dz, vec_dz,
                                                    _mm512_fmadd_pd(vec_dy, vec_dy,
                                                    _mm512_mul_pd(vec_dx, vec_dx))),
                                            _mm512_set1_pd(0.0001));
        __m512d vec_d = _mm512_sqrt_pd(vec_temp_d);

        // Note: mass is not factored in, since when calculating ax, ay, az we do F/mass.
        //       The same goes for d*d. Since it is sqrt'ed later, we don't have to multiply here.
        __m512d vec_F = _mm512_div_pd(_mm512_mul_pd(vec_old_mass, vec_GRAVCONST), vec_temp_d);

        // calculate acceleration due to the force, F and add to velocities
        // (approximate velocities in "unit time")
        // Note: elements where (i==j) or (j>num) are not written back thanks to using a mask.
        vec_vx = _mm512_mask3_fmadd_pd(vec_F, _mm512_div_pd(vec_dx, vec_d), vec_vx, j_mask);
        vec_vy = _mm512_mask3_fmadd_pd(vec_F, _mm512_div_pd(vec_dy, vec_d), vec_vy, j_mask);
        vec_vz = _mm512_mask3_fmadd_pd(vec_F, _mm512_div_pd(vec_dz, vec_d), vec_vz, j_mask);
      }

      vec_old_x = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, old_x + i);
      vec_old_y = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, old_y + i);
      vec_old_z = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, old_z + i);
      vec_old_mass = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, old_mass + i);

      // calc new position
      // Note: elements where i>num, are not written back thanks to using a mask.
      vec_x = _mm512_add_pd(vec_old_x, vec_vx); _mm512_mask_store_pd(x + i, i_mask, vec_x);
      vec_y = _mm512_add_pd(vec_old_y, vec_vy); _mm512_mask_store_pd(y + i, i_mask, vec_y);
      vec_z = _mm512_add_pd(vec_old_z, vec_vz); _mm512_mask_store_pd(z + i, i_mask, vec_z);

      // temp-dependent condensation from gas to liquid
      vec_gas = _mm512_mul_pd(vec_gas, _mm512_mul_pd(vec_loss_rate, vec_exp_kT));
      vec_liquid = _mm512_sub_pd(_mm512_set1_pd(1.0), vec_gas);
      vec_mass = _mm512_fmadd_pd(vec_gas, vec_gas_mass, _mm512_mul_pd(vec_liquid, vec_liquid_mass));
      // Store gas, liquid and mass
      _mm512_mask_store_pd(gas + i, i_mask, vec_gas);
      _mm512_mask_store_pd(liquid + i, i_mask, vec_liquid);
      _mm512_mask_store_pd(mass + i, i_mask, vec_mass);

      // "sqrt(old_mass * v_squared / mass) / sqrt(v_squared)"" can be simpliefied to:
      // "sqrt(old_mass / mass)", if numbers rooted are >0 (guranteed for mass & velocities).
      __m512d factor = _mm512_sqrt_pd(_mm512_div_pd(vec_old_mass, vec_mass));

      vec_vx = _mm512_mul_pd(factor, vec_vx);
      vec_vy = _mm512_mul_pd(factor, vec_vy);
      vec_vz = _mm512_mul_pd(factor, vec_vz);

      // Store vx, vy, vz
      _mm512_mask_store_pd(vx + i, i_mask, vec_vx);
      _mm512_mask_store_pd(vy + i, i_mask, vec_vy);
      _mm512_mask_store_pd(vz + i, i_mask, vec_vz);

    } // end of LOOP 2

    // Do a (+) reduction into totalMass.
    // First each MPI rank does AVX3 vec -> double. Then do MPI_REDUCE into one double.
    vec_totalMass = _mm512_set1_pd(0.0);
    for (i = rankOffset; i < rankLimit; i += 8) {
      // Load mass elements if i<num, else set to 0.
      __m512i vec_i = _mm512_setr_epi64(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
      __mmask8 i_mask = _mm512_cmp_epi64_mask(vec_i, _mm512_set1_epi64(rankLimit), 1); // OP: 1 is LT
      __m512d vec_gas = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, gas + i);
      __m512d vec_liquid = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, liquid + i);

      // mass[i] = gas[i]*gas_mass + liquid[i]*liquid_mass;
      // Will have 0 for elements i >= num.
      __m512d vec_mass = _mm512_fmadd_pd(vec_gas, vec_gas_mass,
                                        _mm512_mul_pd(vec_liquid, vec_liquid_mass));
      vec_totalMass = _mm512_add_pd(vec_totalMass, vec_mass);
      // Store back to mem.
      _mm512_mask_store_pd(mass + i, i_mask, vec_mass);
    }
    totalMass = _mm512_reduce_add_pd(vec_totalMass);
    root_totalMass = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&totalMass, &root_totalMass, 1,   // for all local totalMass: totalMass += local totalMass
              MPI_DOUBLE, MPI_SUM,               // it's a SUM(double, double) op
              0, MPI_COMM_WORLD);                // receive to root process (rank=0)

    // Gather velocities into root.
    // In gatherv, the send and receive buffers cannot alias (there is no IN_PLACE flag for gatherv).
    // We reuse old_* buffers here and swap pointers in root later.
    MPI_Gatherv(vx + rankOffset, toSend, MPI_DOUBLE,
                old_x, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vy + rankOffset, toSend, MPI_DOUBLE,
                old_y, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vz + rankOffset, toSend, MPI_DOUBLE,
               old_z, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root calculates system energy, and prints statistics.
    if (rankId == 0) {
      swap_ptr(&vx, &old_x); swap_ptr(&vy, &old_y); swap_ptr(&vz, &old_z);
      systemEnergy = calc_system_energy(root_totalMass, vx, vy, vz, num);
      printf("At end of timestep %d with temp %f the system energy=%g and total aerosol mass=%g\n",
              time, T, systemEnergy, root_totalMass);
    }

    // temperature drops per timestep
    T *= 0.99999;
  } // time steps


  // In gatherv, the send and receive buffers cannot alias (there is no IN_PLACE flag for gatherv).
  // We reuse old_* buffers here and swap pointers in root later.
  MPI_Gatherv(x + rankOffset, toSend, MPI_DOUBLE,
              old_x, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(y + rankOffset, toSend, MPI_DOUBLE,
              old_y, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(z + rankOffset, toSend, MPI_DOUBLE,
              old_z, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(mass + rankOffset, toSend, MPI_DOUBLE,
              old_mass, counts, displacements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  swap_ptr(&x, &old_x); swap_ptr(&y, &old_y); swap_ptr(&z, &old_z); swap_ptr(&mass, &old_mass);

  if (rankId == 0)
  {
    printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n",
           num, timesteps, omp_get_wtime()-start);

    // output a metric (centre of mass) for checking
    double com[3];
    calc_centre_mass(com, x,y,z,mass,root_totalMass,num);
    printf("Centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);
  }

  MPI_Finalize();
} // main


// init() will return 0 only if successful
int init(double *mass, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *gas, double* liquid, double* loss_rate, int num) {
  // random numbers to set initial conditions - do not parallelise or amend order of random number usage
  int i;
  double min_pos = -50.0, mult = +100.0, maxVel = +10.0;
  double recip = 1.0 / (double)RAND_MAX;

  // create all random numbers
  int numToCreate = num * 8;
  double *ranvec;
  ranvec = (double *) malloc(numToCreate * sizeof(double));
  if (ranvec == NULL) {
    printf("\n ERROR in malloc ranvec within init()\n");
    return -99;
  }

  // Don't change order
  for (i=0; i<numToCreate; i++) {
    ranvec[i] = (double) rand();
  }

  // Pull invariants and common subexpressions out of the for loop.
  // Since we're using -O0, the compiler won't do these for us.
  double recipDivBy2 = recip/2.0;
  double recipDivBy25 = recip/25.0;
  double multTimesRecip = mult * recip;
  double maxVelTimes2 = 2.0*maxVel;
  double maxVelTimes2TimesRecip = maxVelTimes2 * recip;

  for (i=0; i<num; i++) {
    // TODO: we could use SIMD here as well, with a gather load into ranvec to preserve order.
    //       But init() doesn't take many cycles, so probably not worth the effort.
    double *thisRanvec = ranvec + (i << 3); // i*8

    x[i] = min_pos + thisRanvec[0] * multTimesRecip;
    y[i] = min_pos + thisRanvec[1] * multTimesRecip;
    z[i] = thisRanvec[2] * multTimesRecip;

    vx[i] = -maxVel + thisRanvec[3] * maxVelTimes2TimesRecip;
    vy[i] = -maxVel + thisRanvec[4] * maxVelTimes2TimesRecip;
    vz[i] = -maxVel + thisRanvec[5] * maxVelTimes2TimesRecip;

    // proportion of aerosol that evaporates
    loss_rate[i] = 1.0 - thisRanvec[7] * recipDivBy25;
    // aerosol is component of gas and (1-comp) of liquid
    gas[i] = .5 + thisRanvec[6] * recipDivBy2;
    liquid[i] = (1.0 - gas[i]);
  }

  // release temp memory for ranvec which is no longer required
  free(ranvec);

  return 0;
} // init


double calc_system_energy(double mass, double *vx, double *vy, double *vz, int num) {
  /*
     energy is sum of 0.5*mass*velocity^2
     where velocity^2 is sum of squares of components
  */
  int i;
  double totalEnergy = 0.0, systemEnergy;
  __m512d vec_totalEnergy = _mm512_set1_pd(0.0);
  for (i=0; i<num; i += 8) {
    // Load mass elements if i<num, else set to 0.
    __m512i vec_i = _mm512_setr_epi64(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
    __mmask8 i_mask = _mm512_cmp_epi64_mask(vec_i, _mm512_set1_epi64(num), 1); // OP: 1 is LT
    __m512d vec_vx = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, vx + i);
    __m512d vec_vy = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, vy + i);
    __m512d vec_vz = _mm512_mask_load_pd(_mm512_set1_pd(0.0), i_mask, vz + i);
    // The expression "d += a*a + b*b + c*c" can be done using 3 fma ops:
    //  fma(a, a, fma(b, b, fma(c, c, d))
    vec_totalEnergy = _mm512_fmadd_pd(vec_vz, vec_vz, _mm512_fmadd_pd(vec_vy, vec_vy,
                                      _mm512_fmadd_pd(vec_vx, vec_vx, vec_totalEnergy)));
  }

  totalEnergy = _mm512_reduce_add_pd(vec_totalEnergy) * 0.5 * mass;
  systemEnergy = totalEnergy / (double) num;
  return systemEnergy;
}


void output_particles(double *x, double *y, double *z, double *vx, double *vy, double *vz, double *gas, double *liquid, int n) {
  int i;
  printf("num \t position (x,y,z) \t velocity (vx, vy, vz) \t mass (gas, liquid)\n");
  for (i=0; i<n; i++) {
    printf("%d \t %f %f %f \t %f %f %f \t %f %f\n", i, x[i], y[i], z[i], vx[i], vy[i], vz[i], gas[i]*gas_mass, liquid[i]*liquid_mass);
  }
}


void calc_centre_mass(double *com, double *x, double *y, double *z, double *mass, double totalMass, int N) {
  int i, axis;
   // calculate the centre of mass, com(x,y,z)
  for (axis=0; axis<2; axis++) {
    com[0] = 0.0;     com[1] = 0.0;     com[2] = 0.0;
    for (i=0; i<N; i++) {
      com[0] += mass[i]*x[i];
      com[1] += mass[i]*y[i];
      com[2] += mass[i]*z[i];
    }
    com[0] /= totalMass;
    com[1] /= totalMass;
    com[2] /= totalMass;
  }
  return;
}

// Swap two pointers.
void swap_ptr(double **a, double **b) {
  double *tmp;
  tmp = *b; *b = *a; *a = tmp;
}
