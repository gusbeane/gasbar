/*******************************************************************************
 * This file is part of the GALIC code developed by D. Yurin and V. Springel.
 *
 * Copyright (c) 2014
 * Denis Yurin (denis.yurin@h-its.org) 
 * Volker Springel (volker.springel@h-its.org)
 *******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "allvars.h"
#include "proto.h"

int main(int argc, char **argv)
{
  /* MPI-Initialization */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);
  for(PTask = 0; NTask > (1 << PTask); PTask++);

  mpi_printf("\nThis is GalIC, version %s.\n\n"
	     "Running on %d processors.\n\n" "Code was compiled with settings:\n\n", GALIC_VERSION, NTask);

  if(ThisTask == 0)
    output_compile_time_options();

  /* check code parameters */
  if(argc < 2)
    {
      if(ThisTask == 0)
	{
	  printf("\nParameters are missing.\n");
	  printf("Call with <ParameterFile> [<RestartFlag>]\n");
	  printf("\n");
	  printf("   RestartFlag    Action\n");
	  printf("       0          Start from scratch\n");
	  printf("       1          Try to read gravity field from file\n");
	  printf("\n");
	}
      endrun();
    }

  strcpy(ParameterFile, argv[1]);

  if(argc >= 3)
    RestartFlag = atoi(argv[2]);
  else
    RestartFlag = 0;

  /* read in parameters for this run */
  read_parameter_file(ParameterFile);

  /* do basic code initialization */
  init();

  /* determine structural parameters of the galaxy model */
  structure_determination();

  /* allocate force_grid */
  forcegrid_allocate();

  /* allocate density response grid */
  densitygrid_allocate();

  /* allocate grid for calculating velocity dispersion */
  energygrid_allocate();

  /* calculate force/potential/density grids */
  forcedensitygrid_create();

  /* determine solutions of the Jeans equations in the axisymmetric or spherically symmetric cases */
  calculate_dispfield();

  /* set-up particle positions for galaxy model */
  initialize_particles();

  /* calculate mass maps for energy grid */
  calc_energy_grid_mass_maps();

  /* calculate all orbits up front at initial state */
  calc_all_response_fields();


  /* now prepare the iteration loop */
  int iter = 0, rep;
  log_message(iter);
  output_particles(iter);
  output_density_field(iter);

  do
    {      
      for(rep = 0; rep < All.IndepenentOptimizationsPerStep; rep++)
	{
          init_updates();

          optimize_some_particles();

          commit_updates();
	}

      iter++;

      log_message(iter);
      output_particles(iter);
      output_density_field(iter);
    }
  while(iter <= All.MaximumNumberOfSteps);

  mpi_printf("Maximum number of steps reached.\n");

  /* clean up & finalize MPI */
  endrun();

  return 0;
}


/* this function draws new velocities for a fraction of the particles
 * and checks whether they would improve the fit, assuming the other
 * particles are not changed.
 */
void optimize_some_particles(void)
{
  static int first = 1, n = 0, n_per_cpu;
  int i;

  if(first)
    {
      first = 0;
      n_per_cpu = 1 + (All.FractionToOptimizeIndependendly * All.TotNumPart) / NTask;

      mpi_printf("\nParticles treated independently per CPU: %d  (effective value of FractionToOptimizeIndependendly is %g)\n\n",
            n_per_cpu, ((double) n_per_cpu * NTask) / All.TotNumPart);

      if(n_per_cpu > NumPart)
        terminate("n_per_cpu > NumPart");
    }

  for(i = 0; i < n_per_cpu; i++, n++)
    {
      if(n >= NumPart)
        n = 0;

      optimize(permutation[n].index);
    }
}


/* this function calculates all the local "response fields" of the system, consisting of the
 * 1) time-averaged mass density of the orbits (DG_MassLoc), where "DG" stands for density grid.
 * 2) the energy response in the "radial" component (EG_EgyResponseRLoc), where "EG" stands for energy grid.
 * 3) similarly, energy response in the "tangential" component (EG_EgyResponseTLoc)
 * 4) similarly, energy response in the azimuthal component, relative to the streaming velocity (vphi-vstream)^2, (EG_EgyResponsePLoc)
 * 5) similarly, the energy response in azimuthal direction, vphi^2 (EG_EgyResponseQLoc)
 * 6) Finally, also the mass density is accumulated in the energy grid (EG_MassLoc), so that one can obtain the
 *    velocity dispersion by dividing the energy response fields by this.
 */
void calc_all_response_fields(void)
{
  int i, n, type;
 
  /* loop over the possible particle types */
  for(type = 1; type <= 3; type++)
    {
      if(type == 1 && All.Halo_N == 0)
	continue;

      if(type == 2 && All.Disk_N == 0)
	continue;

      if(type == 3 && All.Bulge_N == 0)
	continue;

      for(i = 0; i < DG_Ngrid; i++)
	DG_MassLoc[type][i] = 0;

      for(i = 0; i < EG_Ngrid; i++)
	{
	  EG_MassLoc[type][i] = 0;
	  EG_EgyResponseRLoc[type][i] = 0;
	  EG_EgyResponseTLoc[type][i] = 0;
          EG_EgyResponsePLoc[type][i] = 0;
          EG_EgyResponseQLoc[type][i] = 0;
	}
    }
 
  double *massOrbit = mymalloc("massOrbit", sizeof(double) * DG_Ngrid);

  for(n = 0; n < NumPart; n++)
    {
      type = P[n].Type;

      produce_orbit_response_field(P[n].Pos, P[n].Vel, P[n].ID, massOrbit, P[n].Mass, P[n].Tint, &P[n].Orbits);

      for(i = 0; i < DG_Ngrid; i++)
        DG_MassLoc[P[n].Type][i] += massOrbit[i];

      calc_disp_components_for_particle(n, P[n].Vel, &P[n].vr2, &P[n].vt2, &P[n].vp2, &P[n].vq2);

      add_to_energy_grid(P[n].Pos, P[n].Mass, P[n].vr2, P[n].vt2, P[n].vp2, P[n].vq2,
          EG_MassLoc[type], EG_EgyResponseRLoc[type], EG_EgyResponseTLoc[type], EG_EgyResponsePLoc[type], EG_EgyResponseQLoc[type]);
    }

  myfree(massOrbit);

  calc_global_fit();
}


/* this function evaluates the global quality of the fit of the model, taking into account the
 * orbit response and the initial velocity dispersions of the particles.
 */
void calc_global_fit(void)
{
  int type;
  static int firstrun[6] = {0, 0, 0, 0, 0, 0};

  for(type = 1; type <= 3; type++)
    {
      /* now sum across all CPUs */
      MPI_Allreduce(DG_MassLoc[type], &DGs_MassResponse[type][STACKOFFSET(DG_MaxLevel, 0, 0)], DG_Ngrid, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(EG_MassLoc[type], &EGs_MassResponse[type][STACKOFFSET(EG_MaxLevel, 0, 0)], EG_Ngrid, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(EG_EgyResponseRLoc[type], &EGs_EgyResponse_r[type][STACKOFFSET(EG_MaxLevel, 0, 0)], EG_Ngrid,
		    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(EG_EgyResponseTLoc[type], &EGs_EgyResponse_t[type][STACKOFFSET(EG_MaxLevel, 0, 0)], EG_Ngrid,
		    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(EG_EgyResponsePLoc[type], &EGs_EgyResponse_p[type][STACKOFFSET(EG_MaxLevel, 0, 0)], EG_Ngrid,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(EG_EgyResponseQLoc[type], &EGs_EgyResponse_q[type][STACKOFFSET(EG_MaxLevel, 0, 0)], EG_Ngrid,
                    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      /* hierarchically smooth the results */
      smooth_stack(DGs_MassResponse[type], DG_MaxLevel);
      smooth_stack(EGs_MassResponse[type], EG_MaxLevel);
      smooth_stack(EGs_EgyResponse_r[type], EG_MaxLevel);
      smooth_stack(EGs_EgyResponse_t[type], EG_MaxLevel);
      smooth_stack(EGs_EgyResponse_p[type], EG_MaxLevel);
      smooth_stack(EGs_EgyResponse_q[type], EG_MaxLevel);

      /* calculate the goodness of fit of the density response */
      S[type] = calc_stack_difference(DGs_MassResponse[type], DGs_MassTarget[type], 0, 0, 0, DG_MaxLevel,
				      DGs_MassTarget[type], NULL,
				      All.MinParticlesPerBinForDensityMeasurement * MType[type] /
				      NType[type], DGs_Distance, 0);

      S[type] *= 1.0 / DGs_MassTarget[type][0];


      /* now evalute the difference between the fields for the velocity dispersions */
      Sdisp_r[type] =
	calc_stack_difference(EGs_EgyResponse_r[type], EGs_EgyTarget_r[type], 0, 0, 0, EG_MaxLevel,
			      EGs_MassResponse[type], EGs_MassTarget[type],
			      All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

      Sdisp_t[type] =
	calc_stack_difference(EGs_EgyResponse_t[type], EGs_EgyTarget_t[type], 0, 0, 0, EG_MaxLevel,
			      EGs_MassResponse[type], EGs_MassTarget[type],
			      All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

      Sdisp_p[type] =
        calc_stack_difference(EGs_EgyResponse_p[type], EGs_EgyTarget_p[type], 0, 0, 0, EG_MaxLevel,
			      EGs_MassResponse[type], EGs_MassTarget[type],
                              All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

      Sdisp_q[type] =
        calc_stack_difference(EGs_EgyResponse_q[type], EGs_EgyTarget_q[type], 0, 0, 0, EG_MaxLevel,
                              EGs_MassResponse[type], EGs_MassTarget[type],
                              All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);


      int typeOfVelocityStructure;
      
      if(type == 1)               /* a halo particle */
	typeOfVelocityStructure = All.TypeOfHaloVelocityStructure;
      else if(type == 2)          /* disk */
	typeOfVelocityStructure = All.TypeOfDiskVelocityStructure;
      else if(type == 3)          /* bulge */
	typeOfVelocityStructure = All.TypeOfBulgeVelocityStructure;
      else
	terminate("unknown type");

      /* depending on the type of velocity structure model that is selected, we either average 3 or 4 velocity dispersion constraints,
       * and weight this then roughly equally as the density response.
       */
      if(typeOfVelocityStructure == 0 || typeOfVelocityStructure == 1)
        {
          Srelfac_count[type] = 3.0;
          Sdisp_q[type] = 0;
        }
      else
        Srelfac_count[type] = 4.0;

      /* the relative weighting factor is determined on the first iteration such that density response and velocity structure
       * constraint are weighted equally. The relative weighting factor is then kept fixed as the optimization progresses.
       */
      if(firstrun[type] == 0)
	{
	  Srelfac[type] = S[type] / ((Sdisp_r[type] + Sdisp_t[type] + Sdisp_p[type] + Sdisp_q[type])/ Srelfac_count[type]);
	  firstrun[type] = 1;
	}

      Sdisp_r[type] *= Srelfac[type];
      Sdisp_t[type] *= Srelfac[type];
      Sdisp_p[type] *= Srelfac[type];
      Sdisp_q[type] *= Srelfac[type];
    }
}


/* this function determines a new velocity guess for particle n, and checks whether adopting this would
 * improve the overall fit.
 */
void optimize(int n)
{
  int i, k, orbits, orbits_try;
  double vdir_new;
  double vbase[3], dir[3], sigma;
  double dir_rnd = gsl_rng_uniform(random_generator);
  double F_base, F_try;
  int typeOfVelocityStructure, type = P[n].Type;

  if(type == 1)               /* a halo particle */
    typeOfVelocityStructure = All.TypeOfHaloVelocityStructure;
  else if(type == 2)          /* disk */
    typeOfVelocityStructure = All.TypeOfDiskVelocityStructure;
  else if(type == 3)          /* bulge */
    typeOfVelocityStructure = All.TypeOfBulgeVelocityStructure;
  else
    terminate("unknown type");

  double phi = atan2(P[n].Pos[1], P[n].Pos[0]);
  double vstr = 0;

  if(typeOfVelocityStructure == 2)
    {    
      if(dir_rnd < 0.333333)
	{
	  /* phi - direction */
	  dir[0] = -sin(phi);
	  dir[1] = cos(phi);
	  dir[2] = 0;

	  sigma = sqrt(P[n].vp2_target);
	  vstr = get_vstream(P[n].Pos, P[n].Type);
	}
      else if(dir_rnd < 0.666666)
	{
	  /* R - direction */
	  dir[0] = cos(phi);
	  dir[1] = sin(phi);
	  dir[2] = 0;
	  
	  sigma = sqrt(P[n].vr2_target);
	}
      else
	{
	  /* z - direction */
	  dir[0] = 0;
	  dir[1] = 0;
	  dir[2] = 1;

	  sigma= sqrt(P[n].vt2_target);
	}
    }
  else
    {
      double phi = atan2(P[n].Pos[1], P[n].Pos[0]);
      double theta = acos(P[n].Pos[2] / sqrt(P[n].Pos[0] * P[n].Pos[0] + P[n].Pos[1] * P[n].Pos[1] + P[n].Pos[2] * P[n].Pos[2]));

      if(dir_rnd < 0.333333)
	{
	  /* phi-direction */
          dir[0] = -sin(phi);
          dir[1] = cos(phi);
          dir[2] = 0;

	  sigma = sqrt(P[n].vp2_target);
	  vstr = get_vstream(P[n].Pos, P[n].Type);
	}	  
      else if(dir_rnd < 0.666666)
	{
	  /* radial r-direction */
          dir[0] = sin(theta) * cos(phi);
          dir[1] = sin(theta) * sin(phi);
          dir[2] = cos(theta);

	  sigma = sqrt(P[n].vr2_target);
	}
      else
	{
	  /* theta-direction */
          dir[0] = -cos(theta) * cos(phi);
          dir[1] = -cos(theta) * sin(phi);
          dir[2] = sin(theta);

	  sigma= sqrt(P[n].vt2_target);
	}
    }


  double vdir = P[n].Vel[0] * dir[0] + P[n].Vel[1] * dir[1] + P[n].Vel[2] * dir[2];
  

  /* We only change one of the velocity components, in direction dir[].
   * The rest of the velocity vector is computed in vbase[], and will not be changed.
   */
  vbase[0] = P[n].Vel[0] - vdir * dir[0];
  vbase[1] = P[n].Vel[1] - vdir * dir[1];
  vbase[2] = P[n].Vel[2] - vdir * dir[2];
  
  double vbase2 = (vbase[0] * vbase[0] + vbase[1] * vbase[1] + vbase[2] * vbase[2]);

  int iter =0;
  do
    {
      vdir_new = vstr + gsl_ran_gaussian(random_generator, sigma);

      iter++;
      if(iter > 100000)
	terminate("iter > 100000");
    }     
  while( sqrt(vdir_new * vdir_new + vbase2) >= All.MaxVelInUnitsVesc * P[n].Vesc);


  /* in order to avoid that directional biases through binning or grid alignment effects can
   * lead to subtle biases, we always evaluate an orbit twice, where the relevant velocity component
   * is changed in sign. Vel[] and vel2[] correspond to the sign-changed pair of the original velocity,
   * while vel_try[] and vel_try2[] are the new velocities.
   */

  double vel_try[3], vel_try2[3];
  
  for(k = 0; k < 3; k++)
    {
      vel_try[k] = vbase[k] + vdir_new * dir[k];
      vel_try2[k] = vbase[k] - vdir_new * dir[k];
    }
  
  /* velocity with reversed component */
  double vel2[3];
  vel2[0] = vbase[0] - vdir * dir[0];
  vel2[1] = vbase[1] - vdir * dir[1];
  vel2[2] = vbase[2] - vdir * dir[2];


  double *massOrbit_old = mymalloc("massOrbit_old", sizeof(double) * DG_Ngrid);
  double *massOrbit_new = mymalloc("massOrbit_new", sizeof(double) * DG_Ngrid);

  produce_orbit_response_field(P[n].Pos, P[n].Vel, P[n].ID, massOrbit_old, P[n].Mass, P[n].Tint, &P[n].Orbits);
  produce_orbit_response_field(P[n].Pos, vel_try, P[n].ID, massOrbit_new, P[n].Mass, P[n].Tint, &orbits_try);

  if(vstr > 0)
    {
      F_base = eval_fit(n, P[n].Vel, massOrbit_old, massOrbit_old);
      F_try = eval_fit(n, vel_try, massOrbit_new, massOrbit_old);
    }
  else
    {
      double *massOrbit_old2 = mymalloc("massOrbit_old2", sizeof(double) * DG_Ngrid);
      double *massOrbit_new2 = mymalloc("massOrbit_new2", sizeof(double) * DG_Ngrid);

      produce_orbit_response_field(P[n].Pos, vel2, P[n].ID, massOrbit_old2, P[n].Mass, P[n].Tint, &orbits);
      produce_orbit_response_field(P[n].Pos, vel_try2, P[n].ID, massOrbit_new2, P[n].Mass, P[n].Tint, &orbits);

      F_base = eval_fit(n, P[n].Vel, massOrbit_old, massOrbit_old) + eval_fit(n, vel2, massOrbit_old2, massOrbit_old);
      F_try = eval_fit(n, vel_try, massOrbit_new, massOrbit_old) + eval_fit(n, vel_try2, massOrbit_new2, massOrbit_old);

      myfree(massOrbit_new2);
      myfree(massOrbit_old2);
    }

  Tries[type]++;

  if(F_try < F_base)
    {
      for(k = 0; k < 3; k++)
	P[n].Vel[k] = vel_try[k];

      P[n].Orbits = orbits_try;

      for(i = 0; i < DG_Ngrid; i++)
        DG_MassLoc_delta[type][i] += massOrbit_new[i] - massOrbit_old[i];

      double vr2, vt2, vp2, vq2;
      calc_disp_components_for_particle(n, vel_try, &vr2, &vt2, &vp2, &vq2);

      double vr2_diff = (vr2 - P[n].vr2);
      double vt2_diff = (vt2 - P[n].vt2);
      double vp2_diff = (vp2 - P[n].vp2);
      double vq2_diff = (vq2 - P[n].vq2);

      add_to_energy_grid(P[n].Pos, P[n].Mass, vr2_diff, vt2_diff, vp2_diff, vq2_diff,
            NULL,
            EG_EgyResponseRLoc_delta[type], EG_EgyResponseTLoc_delta[type], EG_EgyResponsePLoc_delta[type], EG_EgyResponseQLoc_delta[type]);

      P[n].vr2 = vr2;
      P[n].vt2 = vt2;
      P[n].vp2 = vp2;
      P[n].vq2 = vq2;

      Changes[type]++;
    }

  myfree(massOrbit_new);
  myfree(massOrbit_old);
  
  Noptimized++;
}


/* this function updated the local response fields with the calculated
 * changes due to adopting modified velocities for some of the particles.
 */
void commit_updates(void)
{
  int i, type;

  for(type = 1; type <= 3; type++)
    {
      if(type == 1 && All.Halo_N == 0)
        continue;

      if(type == 2 && All.Disk_N == 0)
        continue;

      if(type == 3 && All.Bulge_N == 0)
        continue;

      for(i = 0; i < DG_Ngrid; i++)
        DG_MassLoc[type][i] += DG_MassLoc_delta[type][i];

      for(i = 0; i < EG_Ngrid; i++)
        {
          EG_EgyResponseRLoc[type][i] += EG_EgyResponseRLoc_delta[type][i];
          EG_EgyResponseTLoc[type][i] += EG_EgyResponseTLoc_delta[type][i];
          EG_EgyResponsePLoc[type][i] += EG_EgyResponsePLoc_delta[type][i];
          EG_EgyResponseQLoc[type][i] += EG_EgyResponseQLoc_delta[type][i];
        }
    }

  calc_global_fit();
}


/* clear the update fields
 */
void init_updates(void)
{
  int i, type;

  for(type = 1; type <= 3; type++)
    {
      if(type == 1 && All.Halo_N == 0)
        continue;

      if(type == 2 && All.Disk_N == 0)
        continue;

      if(type == 3 && All.Bulge_N == 0)
        continue;

      for(i = 0; i < DG_Ngrid; i++)
        DG_MassLoc_delta[type][i] = 0;

      for(i = 0; i < EG_Ngrid; i++)
        {
          EG_EgyResponseRLoc_delta[type][i] = 0;
          EG_EgyResponseTLoc_delta[type][i] = 0;
          EG_EgyResponsePLoc_delta[type][i] = 0;
          EG_EgyResponseQLoc_delta[type][i] = 0;
        }
    }
}


/* add a particle with the given squared velocity components to the energy grids.
 */
void add_to_energy_grid(double *pos, double mass, double vr2, double vt2, double vp2, double vq2,
                       double *egyMass, double *egyResponse_r, double *egyResponse_t, double *egyResponse_p, double *egyResponse_q)
{
  int iR, iz;
  double fR, fz;

  energygrid_get_cell(pos, &iR, &iz, &fR, &fz);

  vr2 *= mass;
  vt2 *= mass;
  vp2 *= mass;
  vq2 *= mass;

  if(egyMass)
    {
      egyMass[OFFSET(EG_MaxLevel, iz, iR)] +=  (1 - fR) * (1 - fz) * mass;
      egyMass[OFFSET(EG_MaxLevel, iz, iR + 1)] +=  (fR) * (1 - fz) * mass;
      egyMass[OFFSET(EG_MaxLevel, iz + 1, iR)] +=  (1 - fR) * (fz) * mass;
      egyMass[OFFSET(EG_MaxLevel, iz + 1, iR + 1)] +=  (fR) * (fz) * mass;
    }

  egyResponse_r[OFFSET(EG_MaxLevel, iz, iR)] +=  (1 - fR) * (1 - fz) * vr2;
  egyResponse_r[OFFSET(EG_MaxLevel, iz, iR + 1)] +=  (fR) * (1 - fz) * vr2;
  egyResponse_r[OFFSET(EG_MaxLevel, iz + 1, iR)] +=  (1 - fR) * (fz) * vr2;
  egyResponse_r[OFFSET(EG_MaxLevel, iz + 1, iR + 1)] +=  (fR) * (fz) * vr2;

  egyResponse_t[OFFSET(EG_MaxLevel, iz, iR)] +=  (1 - fR) * (1 - fz) * vt2;
  egyResponse_t[OFFSET(EG_MaxLevel, iz, iR + 1)] +=  (fR) * (1 - fz) * vt2;
  egyResponse_t[OFFSET(EG_MaxLevel, iz + 1, iR)] +=  (1 - fR) * (fz) * vt2;
  egyResponse_t[OFFSET(EG_MaxLevel, iz + 1, iR + 1)] +=  (fR) * (fz) * vt2;

  egyResponse_p[OFFSET(EG_MaxLevel, iz, iR)] +=  (1 - fR) * (1 - fz) * vp2;
  egyResponse_p[OFFSET(EG_MaxLevel, iz, iR + 1)] +=  (fR) * (1 - fz) * vp2;
  egyResponse_p[OFFSET(EG_MaxLevel, iz + 1, iR)] +=  (1 - fR) * (fz) * vp2;
  egyResponse_p[OFFSET(EG_MaxLevel, iz + 1, iR + 1)] +=  (fR) * (fz) * vp2;

  egyResponse_q[OFFSET(EG_MaxLevel, iz, iR)] +=  (1 - fR) * (1 - fz) * vq2;
  egyResponse_q[OFFSET(EG_MaxLevel, iz, iR + 1)] +=  (fR) * (1 - fz) * vq2;
  egyResponse_q[OFFSET(EG_MaxLevel, iz + 1, iR)] +=  (1 - fR) * (fz) * vq2;
  egyResponse_q[OFFSET(EG_MaxLevel, iz + 1, iR + 1)] +=  (fR) * (fz) * vq2;
}


/* calc the global quality of the fit if for particle n the velocity vel[] is adopted.
 */
double eval_fit(int n, double vel[3], double *massOrbit_new, double *massOrbit_old)
{
  int i, type;
  double value = 0, value_r = 0, value_t = 0, value_p = 0, value_q = 0;

  type = P[n].Type;

  double *updated_DGs_MassResponse = mymalloc("updated_DGs_MassResponse", DG_Nstack * sizeof(double));

  for(i = 0; i < DG_Ngrid; i++)
    updated_DGs_MassResponse[STACKOFFSET(DG_MaxLevel, 0, 0) + i] = 
      DGs_MassResponse[type][STACKOFFSET(DG_MaxLevel, 0, 0) + i] + massOrbit_new[i] - massOrbit_old[i];

  smooth_stack(updated_DGs_MassResponse, DG_MaxLevel);

  value = calc_stack_difference(updated_DGs_MassResponse, DGs_MassTarget[type], 0, 0, 0, DG_MaxLevel,
				DGs_MassTarget[type], NULL, 
				All.MinParticlesPerBinForDensityMeasurement * MType[type] / NType[type], DGs_Distance, 0);

  value *= 1.0 / MType[type];

  myfree(updated_DGs_MassResponse);



  double mass = P[n].Mass, vr2, vt2, vp2, vq2;

  calc_disp_components_for_particle(n, vel, &vr2, &vt2, &vp2, &vq2);

  double vr2_diff = (vr2 - P[n].vr2);
  double vt2_diff = (vt2 - P[n].vt2);
  double vp2_diff = (vp2 - P[n].vp2);
  double vq2_diff = (vq2 - P[n].vq2);

  double *updated_EGs_EgyResponse_r = mymalloc("updated_EGs_EgyResponse_r", EG_Nstack * sizeof(double));
  double *updated_EGs_EgyResponse_t = mymalloc("updated_EGs_EgyResponse_r", EG_Nstack * sizeof(double));
  double *updated_EGs_EgyResponse_p = mymalloc("updated_EGs_EgyResponse_p", EG_Nstack * sizeof(double));
  double *updated_EGs_EgyResponse_q = mymalloc("updated_EGs_EgyResponse_q", EG_Nstack * sizeof(double));

  int off = STACKOFFSET(EG_MaxLevel, 0, 0);

  for(i = 0; i < EG_Ngrid; i++)
    {
      updated_EGs_EgyResponse_r[off + i] = EGs_EgyResponse_r[type][off + i];
      updated_EGs_EgyResponse_t[off + i] = EGs_EgyResponse_t[type][off + i];
      updated_EGs_EgyResponse_p[off + i] = EGs_EgyResponse_p[type][off + i];
      updated_EGs_EgyResponse_q[off + i] = EGs_EgyResponse_q[type][off + i];
    }

  add_to_energy_grid(P[n].Pos, mass, vr2_diff, vt2_diff, vp2_diff, vq2_diff,
		     NULL, 
		     updated_EGs_EgyResponse_r + off, updated_EGs_EgyResponse_t + off, 
		     updated_EGs_EgyResponse_p + off, updated_EGs_EgyResponse_q + off);

  smooth_stack(updated_EGs_EgyResponse_r, EG_MaxLevel);
  smooth_stack(updated_EGs_EgyResponse_t, EG_MaxLevel);
  smooth_stack(updated_EGs_EgyResponse_p, EG_MaxLevel);
  smooth_stack(updated_EGs_EgyResponse_q, EG_MaxLevel);


  value_r = calc_stack_difference(updated_EGs_EgyResponse_r, EGs_EgyTarget_r[type], 0, 0, 0, EG_MaxLevel,
				  EGs_MassResponse[type], EGs_MassTarget[type],
				  All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

  value_t = calc_stack_difference(updated_EGs_EgyResponse_t, EGs_EgyTarget_t[type], 0, 0, 0, EG_MaxLevel,
				  EGs_MassResponse[type], EGs_MassTarget[type],
				  All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

  value_p = calc_stack_difference(updated_EGs_EgyResponse_p, EGs_EgyTarget_p[type], 0, 0, 0, EG_MaxLevel,
				  EGs_MassResponse[type], EGs_MassTarget[type],
                                  All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

  value_q = calc_stack_difference(updated_EGs_EgyResponse_q, EGs_EgyTarget_q[type], 0, 0, 0, EG_MaxLevel,
                                  EGs_MassResponse[type], EGs_MassTarget[type],
                                  All.MinParticlesPerBinForDispersionMeasurement * MType[type] / NType[type], NULL, 1);

  myfree(updated_EGs_EgyResponse_q);
  myfree(updated_EGs_EgyResponse_p);
  myfree(updated_EGs_EgyResponse_t);
  myfree(updated_EGs_EgyResponse_r);

  int typeOfVelocityStructure;

  if(type == 1)               /* a halo particle */
    typeOfVelocityStructure = All.TypeOfHaloVelocityStructure;
  else if(type == 2)          /* disk */
    typeOfVelocityStructure = All.TypeOfDiskVelocityStructure;
  else if(type == 3)          /* bulge */
    typeOfVelocityStructure = All.TypeOfBulgeVelocityStructure;
  else
    terminate("unknown type");

  if(typeOfVelocityStructure == 0 || typeOfVelocityStructure == 1)
    value_q = 0;

  return value + Srelfac[type] * (value_r + value_t + value_p + value_q) / Srelfac_count[type];
}


/* ouput some log-info about how the iterations are progressing.
 */
void log_message(int iter)
{
  int n, k, type;
  long long totNoptimized, tottries, totchanges;

  sumup_large_ints(1, &Noptimized, &totNoptimized);

  for(type = 1; type <= 3; type++)
    {
      sumup_large_ints(1, &Tries[type], &tottries);
      sumup_large_ints(1, &Changes[type], &totchanges);

      Tries[type] = 0;
      Changes[type] = 0;

      double vavg = 0, vavgtot, norbits = 0;
      int count = 0, counttot;

      for(n = 0; n < NumPart; n++)
	{
	  if(P[n].Type == type)
	    {
	      for(k = 0; k < 3; k++)
		vavg += P[n].Vel[k] * P[n].Vel[k];

	      count++;

	      norbits += P[n].Orbits;
	    }
	}

      MPI_Allreduce(&vavg, &vavgtot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(&count, &counttot, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      MPI_Allreduce(&norbits, &Totorbits[type], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      if(counttot > 0)
	{
	  vavg = sqrt(vavgtot / counttot);

	  mpi_printf
	    ("iter=%5d: opt-frac=%10.7g   type=%d  vavg=%10.7g   (average number of orbits %7.3g)      S=%10.7g    [Sdisp_r=%10.7g | Sdisp_t=%10.7g | Sdisp_p=%10.7g | Sdisp_q=%10.7g]    Sall=%10.7g  Chg-Frac=%10.7g\n",
	     iter, 
	     ((double)totNoptimized) / All.TotNumPart,
	     type, vavg, (double) Totorbits[type] / counttot, S[type], 
	     Sdisp_r[type],
	     Sdisp_t[type],
	     Sdisp_p[type],
	     Sdisp_q[type],
	     S[type] + (Sdisp_r[type] + Sdisp_t[type] + Sdisp_p[type] + Sdisp_q[type]) / Srelfac_count[type], 
	     ((double)totchanges) / (tottries+1.0e-60));

	  if(ThisTask == 0)
	    {
	      fprintf(FdFit[type], "%10.7g  %10.7g  %10.7g     %10.7g %10.7g %10.7g %10.7g     %10.7g    %10.7g\n",
		      ((double)totNoptimized) / All.TotNumPart,
		      S[type], vavg,
		      Sdisp_r[type],
		      Sdisp_t[type],
		      Sdisp_p[type],
		      Sdisp_q[type],
		      S[type] + (Sdisp_r[type] + Sdisp_t[type] + Sdisp_p[type] + Sdisp_q[type]) / Srelfac_count[type],
		      ((double)totchanges) / (tottries+1.0e-60));
	      fflush(FdFit[type]);
	    }
	}
    }
}

