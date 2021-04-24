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

  // time=0 was initial conditions
  for (time=1; time<=timesteps; time++) {

    // LOOP1: take snapshot to use on RHS when looping for updates
    for (i=0; i<num; i++) {
      old_x[i] = x[i];
      old_y[i] = y[i];
      old_z[i] = z[i];
      old_mass[i] = mass[i];
    }

    double temp_d, temp_z;

    // LOOP2: update position etc per particle (based on old data)
    for(i=0; i<num; i++) {
      // calc forces on body i due to particles (j != i)
      for (j=0; j<num; j++) {
        if (j != i) {
          dx = old_x[j] - x[i];
          dy = old_y[j] - y[i];
	  dz = old_z[j] - z[i];
          temp_d =sqrt(dx*dx + dy*dy + dz*dz);
          d = temp_d>0.01 ? temp_d : 0.01;
          F = GRAVCONST * mass[i] * old_mass[j] / (d*d);
	  // calculate acceleration due to the force, F
          ax = (F/mass[i]) * dx/d;
          ay = (F/mass[i]) * dy/d;
	  az = (F/mass[i]) * dz/d;
	  // approximate velocities in "unit time"
          vx[i] += ax;
          vy[i] += ay;
	  vz[i] += az;
        }
      } 
      // calc new position 
      x[i] = old_x[i] + vx[i];
      y[i] = old_y[i] + vy[i];
      z[i] = old_z[i] + vz[i];
      // temp-dependent condensation from gas to liquid
      gas[i] *= loss_rate[i] * exp(-k*T);
      liquid[i] = 1.0 - gas[i];
      mass[i] = gas[i]*gas_mass + liquid[i]*liquid_mass;
      // conserve energy means 0.5*m*v*v remains constant
      double v_squared = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
      double factor = sqrt(old_mass[i]*v_squared/mass[i])/sqrt(v_squared);
      vx[i] *= factor;
      vy[i] *= factor;
      vz[i] *= factor;
    } // end of LOOP 2
    //    output_particles(x,y,z, vx,vy,vz, gas, liquid, num);
    totalMass = 0.0;
    for (i=0; i<num; i++) {
      totalMass += mass[i];
    }
    systemEnergy = calc_system_energy(totalMass, vx, vy, vz, num);
    printf("At end of timestep %d with temp %f the system energy=%g and total aerosol mass=%g\n", time, T, systemEnergy, totalMass);
    // temperature drops per timestep
    T *= 0.99999;
  } // time steps
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
  double comp;
  double min_pos = -50.0, mult = +100.0, maxVel = +10.0;
  double recip = 1.0 / (double)RAND_MAX;

  // create all random numbers 
  int numToCreate = num*8;
  double *ranvec;
  ranvec = (double *) malloc(numToCreate * sizeof(double));
  if (ranvec == NULL) {
    printf("\n ERROR in malloc ranvec within init()\n");
    return -99;
  } 
  for (i=0; i<numToCreate; i++) {
    ranvec[i] = (double) rand();
  }

  // requirement to access ranvec in same order as if we had just used rand()
  for (i=0; i<num; i++) {
    x[i] = min_pos + mult*ranvec[8*i+0] * recip;  
    y[i] = min_pos + mult*ranvec[8*i+1] * recip;  
    z[i] = 0.0 + mult*ranvec[8*i+2] * recip;   
    vx[i] = -maxVel + 2.0*maxVel*ranvec[8*i+3] * recip;   
    vy[i] = -maxVel + 2.0*maxVel*ranvec[8*i+4] * recip;   
    vz[i] = -maxVel + 2.0*maxVel*ranvec[8*i+5] * recip;   
    // proportion of aerosol that evaporates
    comp = .5 + ranvec[8*i+6]*recip/2.0;
    loss_rate[i] = 1.0 - ranvec[8*i+7]*recip/25.0;
    // aerosol is component of gas and (1-comp) of liquid
    gas[i] = comp;
    liquid[i] = (1.0-comp);
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
