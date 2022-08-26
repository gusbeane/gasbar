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

#include "allvars.h"
#include "proto.h"


double get_density_of_type(double *pos, int type)
{
  if(type == 1)
    return halo_get_density(pos);
  else if(type == 2)
    return disk_get_density(pos);
  else if(type == 3)
    return bulge_get_density(pos);
  else
    terminate("unknown type");

  return 0;
}

double get_beta_of_type(double *pos, int type)
{
  double beta = 0;

  if(type == 1)
    {
      beta = All.HaloBetaParameter;
      if(beta >= 1)
        {
          /* this signals that we adopt a beta that depends on the local density slope */
          double r = sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
          double dlogrhodr = -1.0 - 3.0 * r / (r + All.Halo_A);
          beta = -0.15 - 0.20 * dlogrhodr;
        }
    }
  else if(type == 2)
    {
      /* for the disk component, we only support beta = 0 */
      beta = 0;
    }
  else if(type == 3)
    {
      beta = All.BulgeBetaParameter;
      if(beta >= 1)
        {
          /* this signals that we adopt a beta that depends on the local density slope */
          double r = sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
          double dlogrhodr = -1.0 - 3.0 * r / (r + All.Bulge_A);
          beta = -0.15 - 0.20 * dlogrhodr;
        }
    }
  else
    terminate("unknown type");

  return beta;
}



void get_disp_rtp(double *pos, int type, double *disp_r, double *disp_t, double *disp_p, double *disp_q)
{
  int typeOfVelocityStructure = 0;

  if(type == 1)               /* a halo particle */
    typeOfVelocityStructure = All.TypeOfHaloVelocityStructure;
  else if(type == 2)          /* disk */
    typeOfVelocityStructure = All.TypeOfDiskVelocityStructure;
  else if(type == 3)          /* bulge */
    typeOfVelocityStructure = All.TypeOfBulgeVelocityStructure;
  else
    terminate("unknown type");

  if(typeOfVelocityStructure == 0)	/* spherical, isotropic case */
    {
      *disp_r = get_radial_disp_spherical(pos, type);
      *disp_t = *disp_r;
      *disp_p = *disp_r;
      *disp_q = *disp_r;
    }
  else if(typeOfVelocityStructure == 1)	/* spherical, anisotropic case */
    {
      *disp_r = get_radial_disp_spherical(pos, type);
      *disp_t = (1 - get_beta_of_type(pos, type)) * (*disp_r);
      *disp_p = *disp_t;
      *disp_q = *disp_t;
    }
  else if(typeOfVelocityStructure == 2)
    {
      *disp_t = get_z_disp_cylindrical(pos, type);
      *disp_r = *disp_t;
      *disp_p = get_phi_disp(pos, type);

      double vstr = get_vstream(pos, type);
      *disp_q = (*disp_p) + vstr * vstr;
    }
  else if(typeOfVelocityStructure == 3)
    {
      *disp_r = get_r_disp_tilted(pos, type);
      *disp_t = get_theta_disp_tilted(pos, type);
      *disp_p = get_phi_disp(pos, type);

      double vstr = get_vstream(pos, type);
       *disp_q = (*disp_p) + vstr * vstr;
     }
  else
    terminate("unknown velocity structure");
}


double get_vstream(double *pos, int type)
{
  int iz, iR;
  double fR, fz;

  forcegrid_get_cell(pos, &iR, &iz, &fR, &fz);

  double vstr =
    FG_Vstream[type][iz * FG_Nbin + iR] * (1 - fR) * (1 - fz) +
    FG_Vstream[type][(iz + 1) * FG_Nbin + iR] * (1 - fR) * (fz) +
    FG_Vstream[type][iz * FG_Nbin + (iR + 1)] * (fR) * (1 - fz) +
    FG_Vstream[type][(iz + 1) * FG_Nbin + (iR + 1)] * (fR) * (fz);

  return vstr;
}

double get_z_disp_cylindrical(double *pos, int type)
{
  int iz, iR;
  double fR, fz;

  forcegrid_get_cell(pos, &iR, &iz, &fR, &fz);

  double disp = FG_DispZ[type][iz * FG_Nbin + iR] * (1 - fR) * (1 - fz) +
    FG_DispZ[type][(iz + 1) * FG_Nbin + iR] * (1 - fR) * (fz) +
    FG_DispZ[type][iz * FG_Nbin + (iR + 1)] * (fR) * (1 - fz) +
    FG_DispZ[type][(iz + 1) * FG_Nbin + (iR + 1)] * (fR) * (fz);

  return disp;
}


double get_phi_disp(double *pos, int type)
{
  int iz, iR;
  double fR, fz;

  forcegrid_get_cell(pos, &iR, &iz, &fR, &fz);

  double disp = FG_DispPhi[type][iz * FG_Nbin + iR] * (1 - fR) * (1 - fz) +
      FG_DispPhi[type][(iz + 1) * FG_Nbin + iR] * (1 - fR) * (fz) +
      FG_DispPhi[type][iz * FG_Nbin + (iR + 1)] * (fR) * (1 - fz) +
      FG_DispPhi[type][(iz + 1) * FG_Nbin + (iR + 1)] * (fR) * (fz);

  return disp;
}

double get_r_disp_tilted(double *pos, int type)
{
  int iz, iR;
  double fR, fz;

  forcegrid_get_cell(pos, &iR, &iz, &fR, &fz);

  double disp = FG_tilted_vR2_prime[type][iz * FG_Nbin + iR] * (1 - fR) * (1 - fz) +
      FG_tilted_vR2_prime[type][(iz + 1) * FG_Nbin + iR] * (1 - fR) * (fz) +
      FG_tilted_vR2_prime[type][iz * FG_Nbin + (iR + 1)] * (fR) * (1 - fz) +
      FG_tilted_vR2_prime[type][(iz + 1) * FG_Nbin + (iR + 1)] * (fR) * (fz);

  return disp;
}

double get_theta_disp_tilted(double *pos, int type)
{
  int iz, iR;
  double fR, fz;

  forcegrid_get_cell(pos, &iR, &iz, &fR, &fz);

  double disp = FG_tilted_vz2_prime[type][iz * FG_Nbin + iR] * (1 - fR) * (1 - fz) +
      FG_tilted_vz2_prime[type][(iz + 1) * FG_Nbin + iR] * (1 - fR) * (fz) +
      FG_tilted_vz2_prime[type][iz * FG_Nbin + (iR + 1)] * (fR) * (1 - fz) +
      FG_tilted_vz2_prime[type][(iz + 1) * FG_Nbin + (iR + 1)] * (fR) * (fz);

  return disp;
}



/* this function decomposes the velocity vector vel[0,1,2] assigned to particle n into
 * the relevant 'radial' and 'tangential' velocity components squared
 */
void calc_disp_components_for_particle(int n, double *vel, double *vr2, double *vt2, double *vp2, double *vq2)
{
  int type = P[n].Type;
  int typeOfVelocityStructure;

  if(type == 1)               /* a halo particle */
    typeOfVelocityStructure = All.TypeOfHaloVelocityStructure;
  else if(type == 2)          /* disk */
    typeOfVelocityStructure = All.TypeOfDiskVelocityStructure;
  else if(type == 3)          /* bulge */
    typeOfVelocityStructure = All.TypeOfBulgeVelocityStructure;
  else
    terminate("unknown type");

  if(typeOfVelocityStructure == 0 || typeOfVelocityStructure == 1 || typeOfVelocityStructure == 3)
    {
      double phi = atan2(P[n].Pos[1], P[n].Pos[0]);
      double theta = acos(P[n].Pos[2] / sqrt(P[n].Pos[0] * P[n].Pos[0] + P[n].Pos[1] * P[n].Pos[1] + P[n].Pos[2] * P[n].Pos[2]));
      double er[3], ePhi[3], eTheta[3];

      er[0] = sin(theta) * cos(phi);
      er[1] = sin(theta) * sin(phi);
      er[2] = cos(theta);

      ePhi[0] = -sin(phi);
      ePhi[1] = cos(phi);
      ePhi[2] = 0;

      eTheta[0] = -cos(theta) * cos(phi);
      eTheta[1] = -cos(theta) * sin(phi);
      eTheta[2] = sin(theta);

      double vr = vel[0] * er[0] + vel[1] * er[1] + vel[2] * er[2];
      double vphi = vel[0] * ePhi[0] + vel[1] * ePhi[1] + vel[2] * ePhi[2];
      double vtheta = vel[0] * eTheta[0] + vel[1] * eTheta[1] + vel[2] * eTheta[2];

      double vstr = 0;

      if(typeOfVelocityStructure == 1 || typeOfVelocityStructure == 3)
	vstr = get_vstream(P[n].Pos, type);

      *vr2 = vr * vr;
      *vt2 = vtheta * vtheta;
      *vp2 = (vphi - vstr) * (vphi - vstr);
      *vq2 = vphi * vphi;
    }
  else if(typeOfVelocityStructure == 2)
    {
      double phi = atan2(P[n].Pos[1], P[n].Pos[0]);
      double eR[3], ePhi[3], eZ[3];

      eR[0] = cos(phi);
      eR[1] = sin(phi);
      eR[2] = 0;

      ePhi[0] = -sin(phi);
      ePhi[1] = cos(phi);
      ePhi[2] = 0;

      eZ[0] = 0;
      eZ[1] = 0;
      eZ[2] = 1;

      double vR = vel[0] * eR[0] + vel[1] * eR[1] + vel[2] * eR[2];
      double vphi = vel[0] * ePhi[0] + vel[1] * ePhi[1] + vel[2] * ePhi[2];
      double vZ = vel[0] * eZ[0] + vel[1] * eZ[1] + vel[2] * eZ[2];

      double vstr = get_vstream(P[n].Pos, type);

      *vr2 = vR * vR;
      *vt2 = vZ * vZ;
      *vp2 = (vphi - vstr) * (vphi - vstr);
      *vq2 = vphi * vphi;
    }
 }


double get_radial_disp_spherical(double *pos, int type)
{
  double r = sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
  double br = (log(r / FG_Rmin + 1.0) / log(FG_Fac));
  int binr;
  double fR;

  if(br < 0)
    br = 0;

  binr = (int) br;
  fR = br - binr;

  if(binr < 0)
    terminate("binr=%d\n", binr);

  if(binr >= FG_Nbin - 1)
    {
      binr = FG_Nbin - 2;
      fR = 1;
    }

  double disp = FG_Disp_r[type][binr] * (1 - fR) + FG_Disp_r[type][binr + 1] * (fR);

  return disp;
}



struct int_parameters
{
  double R;
  int type;
};



double disp_integ(double z, void *param)
{
  double pos[3], acc[3];
  struct int_parameters *par;

  par = param;

  pos[0] = par->R;
  pos[1] = 0;
  pos[2] = z;

  forcegrid_get_acceleration(pos, acc);

  return -acc[2] * get_density_of_type(pos, par->type);
}


double integrate_axisymmetric_jeans(double zstart, double zend, double R, int type)
{
  int steps = 50, i;
  double dz = (zend - zstart) / steps;
  double sum = 0;

  for(i = 0; i < steps; i++)
    {
      double z0 = zstart + i * dz;
      double z1 = zstart + (i + 1) * dz;
      double pos0[3] = {R, 0, z0};
      double pos1[3] = {R, 0, z1};
      double acc0[3], acc1[3];
      
      forcegrid_get_acceleration(pos0, acc0);
      double y0 = -acc0[2] * get_density_of_type(pos0, type);

      forcegrid_get_acceleration(pos1, acc1);
      double y1 = -acc1[2] * get_density_of_type(pos1, type);
      
      sum += 0.5 * (y0 + y1) * dz;
    }

  return sum;
}


double integrate_spherical_jeans_beta(int type, double ystart, double rstart, double rend)
{
  int steps = 50, i;
  double dr = (rend - rstart) / steps;

  for(i = 0; i < steps; i++)
    {
      double r0 = rstart + i * dr;
      double r1 = rstart + (i + 1) * dr;
      double pos0[3] = {r0, 0, 0};
      double pos1[3] = {r1, 0, 0};

      double beta0 = get_beta_of_type(pos0, type);
      double beta1 = get_beta_of_type(pos1, type);

      double pos[3], acc[3];
      pos[0] = r0;
      pos[1] = 0;
      pos[2] = 0;
      double dens0 = get_density_of_type(pos, type);
      forcegrid_get_acceleration(pos, acc);
      double acc0 = acc[0];


      pos[0] = r1;
      pos[1] = 0;
      pos[2] = 0;
      double dens1 = get_density_of_type(pos, type);
      forcegrid_get_acceleration(pos, acc);
      double acc1 = acc[0];

       if(r1 > fabs(dr) && r0 > fabs(dr))
        {
          double ypred = ystart + dr * (dens0 * acc0 - 2 * beta0 / r0 * ystart);
          ystart += dr * 0.5 * ((dens0 * acc0 - 2 * beta0 / r0 * ystart) + (dens1 * acc1 - 2 * beta1 / r1 * ypred));
        }
      else if(r0 > fabs(dr))
        {
          ystart += dr * (dens0 * acc0 - 2 * beta0 / r0 * ystart);
        }
    }

  return ystart;
}


void calculate_dispfield(void)
{
  int i, j, k, type;
  struct int_parameters par;

#define AA(i,j) ((i) * FG_Nbin + (j))    /* (zindex, Rindex) */

  mpi_printf("\nCalculating velocity dispersion fields...\n");


  /* purely radial case first (only useful for TypeOf-VelocityStructure = 0 or 1) */
  for(type = 1; type <= 3; type++)
    {
      if(type == 1 && All.Halo_N == 0)
        continue;

      if(type == 2 && All.Disk_N == 0)
        continue;

      if(type == 3 && All.Bulge_N == 0)
        continue;

      double r1 = FG_Rmin * (pow(FG_Fac, FG_Nbin) - 1.0);
      double r2 = FG_Rmin * (pow(FG_Fac, FG_Nbin + 2) - 1.0);

      double y = integrate_spherical_jeans_beta(type, 0.0, r2, r1);

      for(j = FG_Nbin - 1; j >= 0; j--)
        {
          r1 = FG_Rmin * (pow(FG_Fac, j) - 1.0);
          r2 = FG_Rmin * (pow(FG_Fac, j + 1) - 1.0);

          y = integrate_spherical_jeans_beta(type, y, r2, r1);

          FG_Disp_r[type][j] = y;
        }

      for(j = FG_Nbin - 1; j >= 0; j--)
        {
          r1 = FG_Rmin * (pow(FG_Fac, j) - 1.0);

          double pos[3];
          pos[0] = r1;
          pos[1] = 0;
          pos[2] = 0;
          double dens = get_density_of_type(pos, type);

          if(dens > 0)
            FG_Disp_r[type][j] /= dens;
          else
            FG_Disp_r[type][j]  = 0;
        }

      /* now we output the result */
      if(ThisTask == 0)
        {
          char buf[1000];
          sprintf(buf, "%s/sigma_r_%d.txt", All.OutputDir, type);
          FILE *fd = fopen(buf, "w");
          fprintf(fd, "%d\n", FG_Nbin);
          for(j = 0; j < FG_Nbin; j++)
            {
              r1 = FG_Rmin * (pow(FG_Fac, j) - 1.0);
              double pos[3];
              pos[0] = r1;
              pos[1] = 0;
              pos[2] = 0;
              fprintf(fd, "%g    %g    %g\n", r1, FG_Disp_r[type][j], get_beta_of_type(pos, type));
            }
          fclose(fd);
        }
    }


  /* now do the simple axisymmetric f(E,Lz) case  (useful for TypeOf-VelocityStructure = 2) */
  for(type = 1; type <= 3; type++)
    {
      if(type == 1 && All.Halo_N == 0)
	continue;

      if(type == 2 && All.Disk_N == 0)
	continue;

      if(type == 3 && All.Bulge_N == 0)
	continue;

      double kParameter = 0;

      if(type == 1)
        kParameter = All.HaloStreamingVelocityParameter;
      else if(type == 2)
        kParameter = All.DiskStreamingVelocityParameter;
      else
        kParameter = All.BulgeStreamingVelocityParameter;

      for(j = 0; j < FG_Nbin; j++)
	{
	  double R = FG_Rmin * (pow(FG_Fac, j) - 1.0);
	  double z1, z2;

	  par.R = R;
	  par.type = type;

	  k = FG_Nbin;
	  z1 = FG_Rmin * (pow(FG_Fac, k) - 1.0);

	  double integ = integrate_axisymmetric_jeans(z1, FG_Rmin * (pow(FG_Fac, 2 * FG_Nbin) - 1.0), R, type);

	  for(k = FG_Nbin - 1; k >= 0; k--)
	    {
	      i = k * FG_Nbin + j;	/* r,z */

	      z1 = FG_Rmin * (pow(FG_Fac, k) - 1.0);
	      z2 = FG_Rmin * (pow(FG_Fac, k + 1) - 1.0);
				  
	      integ += integrate_axisymmetric_jeans(z1, z2, R, type);

	      double pos[3];
	      pos[0] = R;
	      pos[1] = 0;
	      pos[2] = z1;
	      double dens = get_density_of_type(pos, type);

	      if(dens > 0)
		{
		  FG_DispZ[type][i] = integ / dens;
		}
	      else
		FG_DispZ[type][i] = 0;
	    }
	}


      /* now calculate streaming velocity through axisymmetric Jeans equations */
      for(k = FG_Nbin - 1; k >= 0; k--)
	for(j = 0; j < FG_Nbin; j++)
	  {
	    double z = FG_Rmin * (pow(FG_Fac, k) - 1.0);
	    double R = FG_Rmin * (pow(FG_Fac, j) - 1.0);
	    double pos[3], acc[3], R1, R2;
	    int i1, i2;

	    pos[0] = R;
	    pos[1] = 0;
	    pos[2] = z;

	    forcegrid_get_acceleration(pos, acc);

	    i = k * FG_Nbin + j;	/* r,z */

	    if(j > 1 && j < FG_Nbin - 1)
	      {
		R1 = FG_Rmin * (pow(FG_Fac, j - 1) - 1.0);
		i1 = k * FG_Nbin + j - 1;

		R2 = FG_Rmin * (pow(FG_Fac, j + 1) - 1.0);
		i2 = k * FG_Nbin + j + 1;
	      }
	    else if(j == 1)
	      {
		R1 = FG_Rmin * (pow(FG_Fac, j) - 1.0);
		i1 = k * FG_Nbin + j;

		R2 = FG_Rmin * (pow(FG_Fac, j + 1) - 1.0);
		i2 = k * FG_Nbin + j + 1;
	      }
	    else if(j == 0)
	      {
		R1 = FG_Rmin * (pow(FG_Fac, j + 1) - 1.0);
		i1 = k * FG_Nbin + j + 1;

		R2 = FG_Rmin * (pow(FG_Fac, j + 2) - 1.0);
		i2 = k * FG_Nbin + j + 2;
	      }
	    else
	      {
		R1 = FG_Rmin * (pow(FG_Fac, j - 1) - 1.0);
		i1 = k * FG_Nbin + j - 1;

		R2 = FG_Rmin * (pow(FG_Fac, j) - 1.0);
		i2 = k * FG_Nbin + j;
	      }

	      pos[0] = R1;
	      double dens1 = get_density_of_type(pos, type);

	      pos[0] = R2;
              double dens2 = get_density_of_type(pos, type);

	      double dlogDensSigma_dlogR = 0;

	      if(dens1 * FG_DispZ[type][i1] > 0 && dens2 * FG_DispZ[type][i2] > 0)
		dlogDensSigma_dlogR = log( (dens2 * FG_DispZ[type][i2]) / (dens1 * FG_DispZ[type][i1])) / log(R2/R1);

	      double Vphi2 = FG_DispZ[type][i] + R * (-acc[0]) + FG_DispZ[type][i] * dlogDensSigma_dlogR;

	      if(Vphi2 > 0)
		{
		  double vstr = 0;

		  if(kParameter >= 0)
		    {
		      if(Vphi2 >= FG_DispZ[type][i])
		        vstr = kParameter * sqrt(Vphi2 - FG_DispZ[type][i]);
		    }
		  else
		    {
		      vstr = -kParameter * sqrt(Vphi2);
		      
		      if(kParameter < -1)
		        terminate("illegal parameter kParameter=%g", kParameter);
		    }

		  FG_DispPhi[type][i] = Vphi2 - vstr * vstr;
		  FG_Vstream[type][i] = vstr;
		}
	      else
		{
		  FG_DispPhi[type][i] = 0;
		  FG_Vstream[type][i] = 0;
		}
	  }
      
      if(ThisTask == 0)
	{
	  double *tmpR = mymalloc("tmpR", FG_Ngrid * sizeof(double));
	  double *tmpz = mymalloc("tmpz", FG_Ngrid * sizeof(double));

	  for(k = 0; k < FG_Nbin; k++)
	    {
	      double z = FG_Rmin * (pow(FG_Fac, k) - 1.0);
	      for(j = 0; j < FG_Nbin; j++)
		{
		  double R = FG_Rmin * (pow(FG_Fac, j) - 1.0);
		  i = k * FG_Nbin + j;	/* z,r */
		  tmpR[i] = R;
		  tmpz[i] = z;
		}
	    }

	  char buf[1000];
	  sprintf(buf, "%s/sigma_%d.dat", All.OutputDir, type);
	  FILE *fd = fopen(buf, "w");
	  fwrite(&FG_Nbin, sizeof(int), 1, fd);
	  fwrite(FG_DispZ[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_DispPhi[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_Vstream[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(tmpR, sizeof(double), FG_Ngrid, fd);
	  fwrite(tmpz, sizeof(double), FG_Ngrid, fd);
	  fclose(fd);

	  myfree(tmpz);
	  myfree(tmpR);
	}
    }

  /* now do the more difficult  axisymmetic f(E,Lz,I3) case  (useful for TypeOf-VelocityStructure = 3) */
  for(type = 1; type <= 3; type++)
    {
      if(type == 1 && All.Halo_N == 0)
        continue;

      if(type == 2 && All.Disk_N == 0)
        continue;

      if(type == 3 && All.Bulge_N == 0)
        continue;

      if(type == 1 && All.TypeOfHaloVelocityStructure != 3)
        continue;

      if(type == 2 && All.TypeOfDiskVelocityStructure != 3)
        continue;

      if(type == 3 && All.TypeOfBulgeVelocityStructure != 3)
        continue;

      if(type == 1 && All.HaloDispersionRoverZratio == 0)
        terminate("invalid HaloDispersionRoverZratio=%g", All.HaloDispersionRoverZratio);

      if(type == 2 && All.DiskDispersionRoverZratio == 0)
        terminate("invalid DiskDispersionRoverZratio=%g", All.DiskDispersionRoverZratio);

      if(type == 3 && All.BulgeDispersionRoverZratio == 0)
        terminate("invalid BulgeDispersionRoverZratio=%g", All.BulgeDispersionRoverZratio);

      double kParameter = 0;

      if(type == 1)
        kParameter = All.HaloStreamingVelocityParameter;
      else if(type == 2)
        kParameter = All.DiskStreamingVelocityParameter;
      else  if(type == 3)
        kParameter = All.BulgeStreamingVelocityParameter;

      /* grid for solution */
      double *FG_q = mymalloc("FG_q", FG_Ngrid * sizeof(double));

      /* auxiliary vectors */
      double *qprev = mymalloc("qprev", FG_Nbin * sizeof(double));


      for(k = FG_Nbin-1; k >= 0; k--)
	{
	  mpi_printf("method of lines, row %d out of %d for type=%d\n", k, FG_Nbin, type);

	  if(k== FG_Nbin-1)  
	    {
	      for(j = 0; j < FG_Nbin; j++)
		{
		  FG_q[AA(k,j)] = 0;
		}
	    }
	  else
	    {
	      double z0 = FG_Rmin * (pow(FG_Fac, k + 1) - 1.0);
              double z1 = FG_Rmin * (pow(FG_Fac, k) - 1.0);

	      double dz = (z1 - z0);
	      int nsteps = 100, st;
	
	      for(j = 0; j < FG_Nbin; j++)
		qprev[j] = FG_q[AA(k+1,j)];
	
	      for(st = 0; st < nsteps; st++)
		{
		  double z = z0 + (dz / nsteps)*st;

		  for(j = 0; j < FG_Nbin; j++)
		    {
		      double R1 = FG_Rmin * (pow(FG_Fac, j) - 1.0);
		      double h1 = h_factor(R1, z, type);

		      double pos[3], acc[3];
		      pos[0] = R1;
		      pos[1] = 0;
		      pos[2] = z;
		      double dens = get_density_of_type(pos, type);
		      forcegrid_get_acceleration(pos, acc);
		      double p = - dens * acc[2];

		      double dqdR;

		      if(h1 > 0)
			{
			  if(j == FG_Nbin - 1)
			    dqdR = 0;
			  else
			    {
			      double R2 = FG_Rmin * (pow(FG_Fac, j+1) - 1.0);
			      double h2 = h_factor(R2, z, type);
			      
			      double dR = R2 - R1;
			      
			      dqdR = (h2*qprev[j+1] - h1*qprev[j]) / (dR)  +  qprev[j] * h_over_R(R1, z, type);
			    }
			  
			  FG_q[AA(k,j)] = qprev[j]  + ( - p - dqdR) * (dz / nsteps);
			}
		      else
			{
			  if(j == 0)
			    {
			      double R2 = FG_Rmin * (pow(FG_Fac, j+1) - 1.0);
			      double h2 = h_factor(R2, z, type);
			      
			      double dR = R2 - R1;
			      
			      dqdR = (h2*qprev[j+1] - h1*qprev[j]) / (dR)  +  qprev[j] * h_over_R(R1, z, type);
			    }
			  else
			    {
			      double R2 = FG_Rmin * (pow(FG_Fac, j-1) - 1.0);
			      double h2 = h_factor(R2, z, type);
			      
			      double dR = R2 - R1;
		      
			      dqdR = (h2*qprev[j-1] - h1*qprev[j]) / (dR)  +  qprev[j] * h_over_R(R1, z, type);
			    }
			  
			  FG_q[AA(k,j)] = qprev[j]  + ( - p - dqdR) * (dz / nsteps);
			}
		    }
 
		  for(j = 0; j < FG_Nbin; j++)
		    qprev[j] = FG_q[AA(k,j)];
		}
	    }
	}

      for(k = 0; k < FG_Nbin; k++)
	for(j = 0; j < FG_Nbin; j++)
	  {
	    double R = FG_Rmin * (pow(FG_Fac, j) - 1.0);
	    double z = FG_Rmin * (pow(FG_Fac, k) - 1.0);
	    double pos[3];
	    pos[0] = R;
	    pos[1] = 0;
	    pos[2] = z;
	    double dens = get_density_of_type(pos, type);

	    if(dens > 0)
	      FG_tilted_vz2[type][AA(k,j)] =  FG_q[AA(k,j)] / dens;
	    else
	      FG_tilted_vz2[type][AA(k,j)] = 0;

	    double f = 0;
	    if(type == 1)
	      f = All.HaloDispersionRoverZratio;
	    else if(type == 2)
	      f = All.DiskDispersionRoverZratio;
	    else if(type == 3)
	      f = All.BulgeDispersionRoverZratio;
	    else
	      terminate("not allowed");

	    if(j > 0)
	      {
		double alpha = atan(z / R);

		double vrvz = FG_tilted_vz2[type][AA(k,j)] *  ((f - 1) / 2 * tan (2 * alpha))/
		  ( pow(cos(alpha),2) - f*pow(sin(alpha),2) + (1.0+f)/2.0 * sin(2*alpha) * tan(2*alpha));

		double vr2 = FG_tilted_vz2[type][AA(k,j)] *
		  ( f*pow(cos(alpha),2) - pow(sin(alpha),2) + (1.0+f)/2.0 * sin(2*alpha) * tan(2*alpha)) /
		  ( pow(cos(alpha),2) - f*pow(sin(alpha),2) + (1.0+f)/2.0 * sin(2*alpha) * tan(2*alpha));

		double vr2_prime = vr2 * pow(cos(alpha),2) + 2 * vrvz * sin(alpha)*cos(alpha) + FG_tilted_vz2[type][AA(k,j)] * pow(sin(alpha),2);
		double vz2_prime = vr2 * pow(sin(alpha),2) - 2 * vrvz * sin(alpha)*cos(alpha) + FG_tilted_vz2[type][AA(k,j)] * pow(cos(alpha),2);

		FG_tilted_vR2[type][AA(k,j)] = vr2;
		FG_tilted_vz2_prime[type][AA(k,j)] = vz2_prime;
		FG_tilted_vR2_prime[type][AA(k,j)] = vr2_prime;
	      }
	    else
	      {
		FG_tilted_vR2[type][AA(k,j)] = FG_tilted_vz2[type][AA(k,j)] / f;
		FG_tilted_vz2_prime[type][AA(k,j)] = FG_tilted_vR2[type][AA(k,j)];
		FG_tilted_vR2_prime[type][AA(k,j)] = FG_tilted_vz2[type][AA(k,j)];
	      }

	  }


      /* now calculate streaming velocity through axisymmetric Jeans equations */
      for(k = FG_Nbin - 1; k >= 0; k--)
	for(j = 0; j < FG_Nbin; j++)
	  {
	    double z = FG_Rmin * (pow(FG_Fac, k) - 1.0);
	    double R = FG_Rmin * (pow(FG_Fac, j) - 1.0);
	    double pos[3], acc[3], R2;
	    int i2;

	    pos[0] = R;
	    pos[1] = 0;
	    pos[2] = z;

	    double dens = get_density_of_type(pos, type);
	    forcegrid_get_acceleration(pos, acc);

	    i = k * FG_Nbin + j;	/* r,z */

	    if(j < FG_Nbin - 1)
	      {
		R2 = FG_Rmin * (pow(FG_Fac, j + 1) - 1.0);
		i2 = k * FG_Nbin + j + 1;
	      }
	    else
	      {
		R2 = FG_Rmin * (pow(FG_Fac, j - 1) - 1.0);
		i2 = k * FG_Nbin + j - 1;
	      }

	      pos[0] = R2;
              double dens2 = get_density_of_type(pos, type);

	      
	      double Vphi2 = 0;

	      if(dens > 0)
		Vphi2 = FG_tilted_vR2[type][i] + R * (-acc[0]) +
		  R / dens * (dens2 * FG_tilted_vR2[type][i2] - dens * FG_tilted_vR2[type][i]) / (R2 -  R) +
		  R / dens * (dens2 * h_factor(R2, z, type) * FG_tilted_vz2[type][i2] - dens * h_factor(R, z, type) * FG_tilted_vz2[type][i]) / (R2 -  R);

	      if(Vphi2 > 0)
		{
		  double vstr = 0;

		  if(kParameter >= 0)
		    {
		      if(Vphi2 >= FG_tilted_vR2[type][i])
		        vstr = kParameter * sqrt((Vphi2 - FG_tilted_vR2[type][i]));
		    }
		  else
		    {
		      vstr = -kParameter * sqrt(Vphi2);

		      if(kParameter < -1)
		        terminate("illegal parameter kParameter=%g", kParameter);
		    }

		  FG_DispPhi[type][i] = Vphi2 - vstr * vstr;
		  FG_Vstream[type][i] = vstr;
		}
	      else
		{
		  FG_DispPhi[type][i] = 0;
		  FG_Vstream[type][i] = 0;
		}
	}

      if(ThisTask == 0)
        {
	  double *tmpR = mymalloc("tmpR", FG_Ngrid * sizeof(double));
	  double *tmpz = mymalloc("tmpz", FG_Ngrid * sizeof(double));

	  for(k = 0; k < FG_Nbin; k++)
	    {
	      double z = FG_Rmin * (pow(FG_Fac, k) - 1.0);
	      for(j = 0; j < FG_Nbin; j++)
		{
		  double R = FG_Rmin * (pow(FG_Fac, j) - 1.0);
		  i = k * FG_Nbin + j;	/* z,r */
		  tmpR[i] = R;
		  tmpz[i] = z;
		}
	    }

          char buf[1000];
          sprintf(buf, "%s/pde_%d.dat", All.OutputDir, type);
          FILE *fd = fopen(buf, "w");
          fwrite(&FG_Nbin, sizeof(int), 1, fd);
          fwrite(FG_q, sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_tilted_vz2[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_tilted_vR2[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_tilted_vz2_prime[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_tilted_vR2_prime[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(tmpR, sizeof(double), FG_Ngrid, fd);
	  fwrite(tmpz, sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_DispZ[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_DispPhi[type], sizeof(double), FG_Ngrid, fd);
	  fwrite(FG_Vstream[type], sizeof(double), FG_Ngrid, fd);
	  fclose(fd);

	  myfree(tmpz);
	  myfree(tmpR);
        }

      myfree(qprev);
      myfree(FG_q);
    }

  mpi_printf("done.\n\n");
}

double h_factor(double R, double z, int type)
{
  double f = 0, fac;

  if(type == 1)
    f = All.HaloDispersionRoverZratio;
  else if(type == 2)
    f = All.DiskDispersionRoverZratio;
  else if(type == 3)
    f = All.BulgeDispersionRoverZratio;
  else
    terminate("not allowed");

  if(R <= 1.0e-12 * z || R == 0)
    fac = 0;
  else
    {
      double alpha = atan(z / R);

      fac = ((f - 1) / 2 * tan (2 * alpha))/
          ( pow(cos(alpha),2) - f*pow(sin(alpha),2) + (1.0+f)/2.0 * sin(2*alpha) * tan(2*alpha));
    }

  return fac;
}



double h_over_R(double R, double z, int type)
{
  double f = 0, fac;

  if(type == 1)
    f = All.HaloDispersionRoverZratio;
  else if(type == 2)
    f = All.DiskDispersionRoverZratio;
  else if(type == 3)
    f = All.BulgeDispersionRoverZratio;
  else
    terminate("not allowed");

  if(z == 0)
    terminate("z = 0 not allowed");

  if(R <= 1.0e-12 * z || R == 0)
    fac = (f - 1) / f;
  else
    {
      double alpha = atan(z / R);

      fac = ((f - 1) / 2 * tan(alpha) * tan (2 * alpha))/
          ( pow(cos(alpha),2) - f*pow(sin(alpha),2) + (1.0+f)/2.0 * sin(2*alpha) * tan(2*alpha));
    }

  return fac / z;
}

