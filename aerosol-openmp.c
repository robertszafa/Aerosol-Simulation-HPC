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

#include <mkl.h>
#include <immintrin.h>


double liquid_mass=2.0, gas_mass=0.3, k=0.00001;

int init(double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, int);
double calc_system_energy(double, double*, double*, double*, int);
void output_particles(double*, double*, double*, double*, double*, double*, double*, double*, int);
void calc_centre_mass(double*, double*, double*, double*, double*, double, int);

int main(int argc, char* argv[]) {
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

  double start=omp_get_wtime();   // we make use of the simple wall clock timer available in OpenMP

  /* if avail, input size of system */
  if (argc > 1 ) {
    num = atoi(argv[1]);
    timesteps = atoi(argv[2]);
  }
  else {
    num = 20000;
    timesteps = 50;
  }

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

  // should check all rc but let's just see if last malloc worked
  if (old_mass == NULL) {
    printf("\n ERROR in malloc for (at least) old_mass - aborting\n");
    return -99;
  }
  else {
    printf("  (malloc-ed)  ");
  }

  // initialise
  rc = init(mass, x, y, z, vx, vy, vz, gas, liquid, loss_rate, num);
  if (rc != 0) {
    printf("\n ERROR during init() - aborting\n");
    return -99;
  }
  else {
    printf("  INIT COMPLETE\n");
  }
  totalMass = 0.0;
  for (i=0; i<num; i++) {
    mass[i] = gas[i]*gas_mass + liquid[i]*liquid_mass;
    totalMass += mass[i];
  }
  //DEBUG  output_particles(x,y,z, vx,vy,vz, gas, liquid, num);
  systemEnergy = calc_system_energy(totalMass, vx, vy, vz, num);
  printf("Time 0. System energy=%g\n", systemEnergy);

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


  printf("Now to integrate for %d timesteps\n", timesteps);


  // For swapping poniters.
  double *tmp_x, *tmp_y, *tmp_z, *tmp_mass;

  // Expand constants into AVX-512 form.
  double tKExpNegative = exp(-k*T);
  __m512d vec_tKExpNegative = _mm512_set_pd(tKExpNegative, tKExpNegative, tKExpNegative, tKExpNegative,
                                            tKExpNegative, tKExpNegative, tKExpNegative, tKExpNegative);
  __m512d vec_GRAVCONST = _mm512_set_pd(GRAVCONST, GRAVCONST, GRAVCONST, GRAVCONST,
                                        GRAVCONST, GRAVCONST, GRAVCONST, GRAVCONST);
  __m512d vec_gas_mass = _mm512_set_pd(gas_mass, gas_mass, gas_mass, gas_mass,
                                       gas_mass, gas_mass, gas_mass, gas_mass);
  __m512d vec_liquid_mass = _mm512_set_pd(liquid_mass, liquid_mass, liquid_mass, liquid_mass,
                                          liquid_mass, liquid_mass, liquid_mass, liquid_mass);
  __m512d vec_min_d = _mm512_set_pd(0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01);
  __m512d vec_one = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
  __m512i vec_num = _mm512_set_epi64(num, num, num, num, num, num, num, num);


  // time=0 was initial conditions
  // #pragma omp parallel default(none) private(time, i, j) \
  //                                    shared(timesteps, tmp_x, tmp_y, tmp_z, tmp_mass)
  {
    for (time=1; time<=timesteps; time++) {

      // Swap pointers insted of moving data.
      tmp_x = x; old_x = x; x = tmp_x;
      tmp_y = x; old_y = y; y = tmp_y;
      tmp_z = z; old_z = z; z = tmp_z;
      tmp_mass = mass; old_mass = mass; mass = tmp_mass;

      // LOOP2: update position etc per particle (based on old data)
      // #pragma omp for schedule(static, 1)
      for(i=0; i<num; i += 8) {
        __m512d vec_old_x, vec_old_y, vec_old_z, vec_old_mass;

        // Load into AVX-512 registers.
        __m512d vec_x = _mm512_load_pd(x + i);
        __m512d vec_y = _mm512_load_pd(y + i);
        __m512d vec_z = _mm512_load_pd(z + i);
        __m512d vec_vx = _mm512_load_pd(vx + i);
        __m512d vec_vy = _mm512_load_pd(vy + i);
        __m512d vec_vz = _mm512_load_pd(vz + i);
        __m512d vec_mass = _mm512_load_pd(mass + i);
        __m512d vec_gas = _mm512_load_pd(gas + i);
        __m512d vec_loss_rate = _mm512_load_pd(loss_rate + i);
        __m512d vec_liquid = _mm512_load_pd(liquid + i);

        // Use masks to cover edge cases, e.g. i>num, i==j, etc.
        __m512i vec_i = _mm512_set_epi64(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
        // i_writeback = (i<num) ? 1 : 0
        __mmask8 i_writeback = _mm512_cmp_epi64_mask(vec_i, vec_num, 1); // 1: OP := _MM_CMPINT_LT

        // calc forces on body i due to particles (j != i)
        for (j=0; j<num; j += 8) {

          // if (i == j) or (j > num), don't set j writeback bit.
          __m512i vec_j = _mm512_set_epi64(j, j+1, j+2, j+3, j+4, j+5, j+6, j+7);
          __mmask8 is_i_eq_j = _mm512_cmp_epi64_mask(vec_i, vec_j, 4); // 4: OP := _MM_CMPINT_NE
          __mmask8 j_writeback = _mm512_mask_cmp_epi64_mask(is_i_eq_j, vec_j, vec_num, 1); // 1: LT

          // Load into AVX-512 registers.
          vec_old_x = _mm512_load_pd(old_x + j);
          vec_old_y = _mm512_load_pd(old_y + j);
          vec_old_z = _mm512_load_pd(old_z + j);
          vec_old_mass = _mm512_load_pd(old_mass + j);

          __m512d vec_dx = _mm512_sub_pd(vec_old_x, vec_x);
          __m512d vec_dy = _mm512_sub_pd(vec_old_y, vec_y);
          __m512d vec_dz = _mm512_sub_pd(vec_old_z, vec_z);

          // The expression "a*a + b*b + c*c" can be done using 1 mul and 2 fma ops:
          //  fma(a, a, fma(b, b, mul(c, c))
          __m512d vec_temp_d = _mm512_sqrt_pd(_mm512_fmadd_pd(vec_dz, vec_dz,
                                              _mm512_fmadd_pd(vec_dy, vec_dy,
                                              _mm512_mul_pd(vec_dx, vec_dx))));

          __m512d vec_d = _mm512_min_pd(vec_temp_d, vec_min_d);

          // vec_F = vec_GRAVCONST * vec_old_mass / (vec_d * vec_d);
          // Note: mass is not factored in, since when calculating ax, ay, az we do F/mass.
          __m512d vec_F = _mm512_div_pd(_mm512_mul_pd(vec_GRAVCONST, vec_old_mass),
                                        _mm512_mul_pd(vec_d, vec_d));

          // calculate acceleration due to the force, F and add to velocities
          // (approximate velocities in "unit time")
          // Note: elements where (i==j) or (j>num) are not written back thanks to using a mask.
          vec_vx = _mm512_mask3_fmadd_pd(vec_F, _mm512_div_pd(vec_dx, vec_d), vec_vx, j_writeback);
          vec_vy = _mm512_mask3_fmadd_pd(vec_F, _mm512_div_pd(vec_dy, vec_d), vec_vy, j_writeback);
          vec_vz = _mm512_mask3_fmadd_pd(vec_F, _mm512_div_pd(vec_dz, vec_d), vec_vz, j_writeback);
        }

        // calc new position & store back to mem.
        // Note: elements where i>num, are not written back thanks to using a mask.
        vec_x = _mm512_add_pd(vec_old_x, vec_vx); _mm512_mask_store_pd(x + i, i_writeback, vec_x);
        vec_y = _mm512_add_pd(vec_old_y, vec_vy); _mm512_mask_store_pd(y + i, i_writeback, vec_y);
        vec_z = _mm512_add_pd(vec_old_z, vec_vz); _mm512_mask_store_pd(z + i, i_writeback, vec_z);

        // temp-dependent condensation from gas to liquid
        vec_gas = _mm512_mul_pd(vec_gas, _mm512_mul_pd(vec_loss_rate, vec_tKExpNegative));
        vec_liquid = _mm512_add_pd(vec_one, vec_gas);
        vec_mass = _mm512_fmadd_pd(vec_gas, vec_gas_mass,
                                   _mm512_mul_pd(vec_liquid, vec_liquid_mass));

        // Store gas, liquid and mass
        _mm512_mask_store_pd(gas + i, i_writeback, vec_gas);
        _mm512_mask_store_pd(liquid + i, i_writeback, vec_liquid);
        _mm512_mask_store_pd(mass + i, i_writeback, vec_mass);

        // conserve energy means 0.5*m*v*v remains constant
        // v_squared = vx*vx + vy*vy + vz*vz
        __m512d vec_v_squared = _mm512_fmadd_pd(vec_vz, vec_vz,
                                                _mm512_fmadd_pd(vec_vy, vec_vy,

                                                                _mm512_mul_pd(vec_vx, vec_vx)));
        __m512d factor = _mm512_div_pd(_mm512_sqrt_pd(_mm512_div_pd(_mm512_mul_pd(vec_old_mass,
                                                                                  vec_v_squared),
                                                                    vec_mass)),
                                       _mm512_sqrt_pd(vec_v_squared));
        vec_vx = _mm512_mul_pd(factor, vec_vx);
        vec_vy = _mm512_mul_pd(factor, vec_vy);
        vec_vz = _mm512_mul_pd(factor, vec_vz);

        // Store vx, vy, vz
        _mm512_mask_store_pd(vx + i, i_writeback, vec_vx);
        _mm512_mask_store_pd(vy + i, i_writeback, vec_vy);
        _mm512_mask_store_pd(vz + i, i_writeback, vec_vz);

      } // end of LOOP 2

      //    output_particles(x,y,z, vx,vy,vz, gas, liquid, num);
      totalMass = 0.0;
      for (i=0; i<num; i++) {
        totalMass += mass[i];
      }
      systemEnergy = calc_system_energy(totalMass, vx, vy, vz, num);

      // printf("At end of timestep %d with temp %f the system energy=%g and total aerosol mass=%g\n", time, T, systemEnergy, totalMass);
      // temperature drops per timestep
      T *= 0.99999;
    } // time steps
  }

  printf("Time to init+solve %d molecules for %d timesteps is %g seconds\n", num, timesteps, omp_get_wtime()-start);

  // output a metric (centre of mass) for checking
  double com[3];
  calc_centre_mass(com, x,y,z,mass,totalMass,num);
  printf("Centre of mass = (%g,%g,%g)\n", com[0], com[1], com[2]);

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

  // Each thread accesses the half-open range of [i : i+8) within ranvec.
  // We can use a chunk size of 8 with the schedule static.
  #pragma omp parallel for default(none) private(i) \
                                         shared(ranvec, num, min_pos, recipDivBy2, recipDivBy25, \
                                                multTimesRecip, maxVelTimes2, \
                                                maxVelTimes2TimesRecip, maxVel, x, y, z, vx, \
                                                vy, vz, loss_rate, gas, liquid) \
                                         schedule(static, 8)
  for (i=0; i<num; i++) {
  {
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

    /// Threads join here.
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
  for (i=0; i<num; i++) {
    totalEnergy += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
  }
  totalEnergy = 0.5 * mass * totalEnergy;
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
