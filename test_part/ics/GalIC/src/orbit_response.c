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


/* returns the timestep for the particle with the giving velocity and acceleration
 */
double get_timestep(double *pos, double *vel, double *acc, int icell)
{
  // double r = sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
  double v = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
  double aa = sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);

  double torbit = All.V200 / aa;
  double tcross = DG_CellSize[icell] / v;

  return dmin(All.TimeStepFactorOrbit * torbit, All.TimeStepFactorCellCross * tcross);
}


/* calculate the density response of a single particle starting from pos[]/vel[],
 * averaged over time 'timespan'. If timespan=0, the routine determines an
 * appropriate time itself.
 */
double produce_orbit_response_field(double *pos, double *vel, int id, double *mfield, double mass,
				    double timespan, int *orbitstaken)
{
  int i, norbit, icell, flag = 0, iR, iz;
  double x[3], v[3], a[3], dt, tall, radsign_previous = 0, radsign, fR, fz;

  for(i = 0; i < 3; i++)
    {
      x[i] = pos[i];
      v[i] = vel[i];
    }

  for(i = 0; i < DG_Ngrid; i++)
    mfield[i] = 0;

  norbit = 0;
  tall = 0;


  forcegrid_get_acceleration(x, a);

  densitygrid_get_cell(x, &iR, &iz, &fR, &fz);
  icell = iz * DG_Nbin + iR;

  int Norbits = 100000000;

  double E0 = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + forcegrid_get_potential(x);
  int steps = 0;

  do
    {
      dt = get_timestep(x, v, a, icell);

      if(timespan > 0)
	if(dt + tall > timespan)
	  {
	    dt = timespan - tall;
	    flag = 1;
	  }

      mfield[iz * DG_Nbin + iR] += 0.5 * dt * (1 - fR) * (1 - fz);
      mfield[iz * DG_Nbin + (iR + 1)] += 0.5 * dt * (fR) * (1 - fz);
      mfield[(iz + 1) * DG_Nbin + iR] += 0.5 * dt * (1 - fR) * (fz);
      mfield[(iz + 1) * DG_Nbin + (iR + 1)] += 0.5 * dt * (fR) * (fz);

      for(i = 0; i < 3; i++)
	v[i] += 0.5 * dt * a[i];

      for(i = 0; i < 3; i++)
	x[i] += dt * v[i];

      forcegrid_get_acceleration(x, a);

      for(i = 0; i < 3; i++)
	v[i] += 0.5 * dt * a[i];

      densitygrid_get_cell(x, &iR, &iz, &fR, &fz);
      icell = iz * DG_Nbin + iR;

      mfield[iz * DG_Nbin + iR] += 0.5 * dt * (1 - fR) * (1 - fz);
      mfield[iz * DG_Nbin + (iR + 1)] += 0.5 * dt * (fR) * (1 - fz);
      mfield[(iz + 1) * DG_Nbin + iR] += 0.5 * dt * (1 - fR) * (fz);
      mfield[(iz + 1) * DG_Nbin + (iR + 1)] += 0.5 * dt * (fR) * (fz);

      tall += dt;

      radsign = v[0] * x[0] + v[1] * x[1] + v[2] * x[2];

      if(radsign > 0 && radsign_previous < 0)
	norbit++;

      radsign_previous = radsign;

      steps++;
      if(steps > 100000000)
	{
	  printf("too many steps...  pos=(%g|%g|%g)  vel=(%g|%g|%g)  dt=%g\n",
		 pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], dt);
	  double E1 = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + forcegrid_get_potential(x);
	  printf("steps=%d:  rel error = %g\n", steps, fabs(E1 - E0) / fabs(E0));
	  exit(1);
	}
    }
  while((timespan == 0 && norbit < Norbits) || (timespan != 0 && flag == 0));

  double E1 = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + forcegrid_get_potential(x);

  double rel_egy_error = fabs((E1 - E0) / E0);

  if(rel_egy_error > 0.5)
    {
      mpi_printf("relative energy error= %g  orbits=%d   steps=%d  pos(=%g|%g|%g) vel=(%g|%g|%g)\n", rel_egy_error, norbit, steps,
		 pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
      /*
         terminate("error seems large, we better stop:  pos=(%g|%g|%g)  vel=(%g|%g|%g) id=%d  v=%g  vesc=%g",
         pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], id, 
         sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]), 
         forcegrid_get_escape_speed(pos));
       */
    }

  double fac = mass / tall;

  for(i = 0; i < DG_Ngrid; i++)
    mfield[i] *= fac;

  *orbitstaken = norbit;

  return tall;
}
