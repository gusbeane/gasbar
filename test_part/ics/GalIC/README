------------------------------------------------------------------------
GALIC v1.0  - A code for the creation of galaxy inititial conditions 
------------------------------------------------------------------------

  Copyright (c) 2014 by Denis Yurin and Volker Springel

  Heidelberg Institute for Theoretical Studies (HITS)
  Schloss-Wolfsbrunnenweg 35, 69118 Heidelberg, Germany,
  
  Heidelberg University, Zentrum fuer Astronomy, ARI
  Moenchhofstrasse 12-14, 69120 Heidelberg, Germany

  Code web-site:     http://www.h-its.org/tap/galic

  GALIC is an implementation of a new iterative method to construct
  steady state composite halo-disk-bulge galaxy models with prescribed
  density distribution and velocity anisotropy.

  The method is described in full in the paper

  Yurin D., Springel, V., 2014, MNRAS, in press
  (see also the preprin at http://arxiv.org/abs/1402.1623) 

  Users of the code are kindly asked to cite the paper if they make
  use of the code. The code is released "as is", without any guarantees
  or warrantees.

------------
Dependencies
------------

  GalIC needs the following non-standard libraries for compilation:

  mpi  - the ‘Message Passing Interface’ (http://www-unix.mcs.anl.gov/mpi/mpich)

  gsl  - the GNU scientiﬁc library. This open-source package can be
         obtained at http://www.gnu.org/software/gsl

  hdf5 - the ‘Hierarchical Data Format’ (available at
         http://hdf.ncsa.uiuc.edu/HDF5).  This library is optional and
         only needed when one wants to read or write snapshot ﬁles in
         HDF format.

-----------
Compilation
-----------

  Please first copy "Template-Makefile.systype" file to
  "Makefile.systype" and uncomment your system if a suitable one is
  already predefined. If your system is not listed, then you should
  define a corresponding section in the "Makefile", and activate it in
  "Makefile.systype".

  Next, please copy the "Template-Config.sh" file to "Config.sh" and
  uncomment the compile-time options specified there according to your
  needs.

  Once the above steps are completed, it should be possible to compile
  the code by simply executing "make".

-----
Usage
-----

  To start GalIC, run the executable with a command of the form

      mpirun -np 12 ./GalIC myparameterfile.param

  This example will run GalIC using 12 MPI processors, and with
  parameters as speciﬁed in the parameter file
  "myparameterfile.param", which is passed as an argument to the
  program.  The number of MPI ranks that is used is arbitrary, and it
  is also possible to run the code in serial.

-------------
Parameterfile
-------------

  The parameterfile is a simple text file that defines in each line a
  parameter-value pair. The sequence of the parameters is arbitrary,
  and comment lines are also allowed. The code will complain about
  missing or duplicate parameters. For creating a parameterfile, it is
  best to refer to one of the example files ("Model_H1.param",
  "Model_H2.param", etc.) included in the code description. These
  correspond to the models considered in the method paper (as listed
  in Table 1 of the paper) and contain comments that explain the
  parameters briefly.
   
  As an important step, we note that the total mass of the
  halo-disk-bulge system needs to be specified. GalIC assumes that
  this mass is the total mass enclosed within the virial radius of an
  equivalent NFW halo. To determine the halo mass distribution one
  should set the parameter V200 (the circular velocity at the virial
  radius) and CC (the concentration parameter) - see for details
  http://arxiv.org/abs/astro-ph/0411108. Another important parameter
  is the number of particles you want to use for the different
  components of the galaxy model, which are specified through N_HALO,
  N_DISK, N_BULGE. The mass of the disk and bulge has to be specified
  as fraction of the total mass via the MD and MB parameters. Both
  halo and bulge are parametrized as Hernquist spheres. The scale
  length of the bulge is determined by 'BulgeSize' and must be given
  in units of halo scale length, which GalIC computes automatically
  from the V200 and CC parameters. To parametrize the disk an
  exponential profile in the radial direction is used. One needs to
  specify the disk spin fraction JD (which essentially defines disk
  scale length) and thickness of the disk through 'DiskHeight'.

  Finally, the type of velocity structure of each component is
  selected with the parameters: TypeOfHaloVelocityStructure,
  TypeOfDiskVelocityStructure, and TypeOfBulgeVelocityStructure. For
  each component one can use four values:

  0 - the velocity ellipsoid is everywhere isotropic, and its radial
  variation is spherically symmetrical;

  1 - the same as 0, but now the specific ratio between radial
  velocity dispersion and the velocity dispersion that is
  perpendicular to the radial one can be specified by
  HaloBetaParameter and BulgeBetaParameter;

  2 - this case represents systems where the velocity ellipsoid is
  isotropic in the meridional plane but it has different shape and may
  have a first moment in the azimuthal direction, as specified by
  HaloStreamingVelocityParameter, DiskStreamingVelocityParameter. We
  refer to these systems as axisymmetric systems with two integrals of
  motion, total energy E and vertical component of angular momentum
  Lz. The streaming parameter either refers directly to "k" as given
  in the paper, or it gives k in units of kmax if a negative value is
  adopted.

  3 - this case is a combination of cases 1 and 2, it allows to choose
  the specific ratio between radial velocity dispersion <vr^2> and the
  one that is perpendicular to the orbital plane <vz^2> via the
  HaloDispersionRoverZratio, DiskDispersionRoverZratio and
  BulgeDispersionRoverZratio parameters, and also to specify a
  streaming velocity via HaloStreamingVelocityParameter, or
  DiskStreamingVelocityParameter. This case corresponds to
  axisymmetric systems with three integrals of motion, E, Lz and a
  non-classical one I3.

  All other parameters are related to the procedure of finding a
  solution, and typically need not be altered. You can find short
  comments on them in the example parameter files provided with the
  code. These 20 parameter files were used to create the initial
  condition described in the paper.

------
Output
------

  As GalIC progresses, it regularly dumps "snaphot files", which can
  be used as initial conditions files. The latest snapshot represents
  the last optimization state, and represents the best initial
  conditions produced by the code thus far. GalIC supports the three
  file formats of the GADGET code for its output snapshot files
  (i.e. the venerable 'type1' format, the slightly improved 'type2'
  format, and an HDF5 format).  The default file format in the example
  parametersfiles is the plain type1 format. A documentation of this
  file format can be found in
  http://www.mpa-garching.mpg.de/gadget/users-guide.pdf

-------
Restart 
-------

  The general usage of GALIC is in fact quite similar to the GADGET
  code. However, restarts are currently not yet supported.



