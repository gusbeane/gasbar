#
#/*******************************************************************************
# * This file is part of the GALIC code developed by D. Yurin and V. Springel.
# *
# * Copyright (c) 2014
# * Denis Yurin (denis.yurin@h-its.org) 
# * Volker Springel (volker.springel@h-its.org)
# *******************************************************************************/
#
# You might be looking for the compile-time Makefile options of the code...
#
# They have moved to a separate file.
#
# To build the code, do the following:
#
#  (1) Copy the file "Template-Config.sh"  to  "Config.sh"
#
#        cp Template-Config.sh Config.sh 
#
#  (2) Edit "Config.sh" as needed for your application
#
#  (3) Run "make"
#
#
#
#
#  New compile-time options should be added to the 
#  file "Template-Config.sh" only. Usually, the should be added
#  there in the disabled/default version.
#
#  "Config.sh" should not be checked in to the repository
#
#  Note: It is possible to override the default name of the 
#  Config.sh file, if desired, as well as the name of the
#  executable. For example:
#
#   make CONFIG=MyNewConf.sh EXEC=GalIC_new
# 
#-----------------------------------------------------------------
#
# You might also be looking for the target system SYSTYPE option
#
# It has also moved to a separate file.
#
# To build the code, do the following:
#
# (A) set the SYSTYPE variable in your .bashrc (or similar file):
#
#        e.g. export SYSTYPE=Magny
# or
#
# (B) set SYSTYPE in Makefile.systype 
#     This file has priority over your shell variable:
#
#    (1) Copy the file "Template-Makefile.systype"  to  "Makefile.systype"
#
#        cp Template-Makefile.systype Makefile.systype 
#
#    (2) Uncomment your system in  "Makefile.systype".
#
# If you add an ifeq for a new system below, also add that systype to
# Template-Makefile.systype

EXEC   = GalIC
CONFIG   = Config.sh
BUILD_DIR = build
SRC_DIR = src

###################
#determine SYSTYPE#
###################
ifdef SYSTYPE
SYSTYPE := "$(SYSTYPE)"
-include Makefile.systype
else
include Makefile.systype
endif

MAKEFILES = Makefile config-makefile
ifeq ($(wildcard Makefile.systype), Makefile.systype)
MAKEFILES += Makefile.systype
endif



PERL	 = /usr/bin/perl
RESULT     := $(shell CONFIG=$(CONFIG) PERL=$(PERL) BUILD_DIR=$(BUILD_DIR) make -f config-makefile)
CONFIGVARS := $(shell cat $(BUILD_DIR)/galicconfig.h)



MPICHLIB = -lmpich
GMPLIB   = -lgmp
GSLLIB   = -lgsl -lgslcblas
MATHLIB  = -lm

ifeq ($(SYSTYPE),"cannon")
CC			= mpicc
OPTIMIZE    = -ggdb -O3 -Wall -Wno-format-security -Wno-unknown-pragmas
ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS)))
OPTIMIZE +=  -fopenmp
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
CC       =  gcc  $(MPI_COMPILE_FLAGS)      # to replace clang with gcc (mpicc uses clang for some reason)
endif
GSL_INCL =
GSL_LIBS =
FFTW_INCL=
FFTW_LIBS=
MPICHLIB =
HDF5INCL =  -DH5_USE_16_API
HDF5LIB  =  -lhdf5 -lz
endif

ifeq ($(SYSTYPE),"Darwin")
CC       =  mpicc   # sets the C-compiler
OPTIMIZE =   -ggdb -O3 -Wall -Wno-format-security -Wno-unknown-pragmas
ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS)))
OPTIMIZE +=  -fopenmp
MPI_COMPILE_FLAGS = $(shell mpicc --showme:compile)
CC       =  gcc  $(MPI_COMPILE_FLAGS)      # to replace clang with gcc (mpicc uses clang for some reason)
endif
GSL_INCL = -I/sw/include -I/opt/local/include 
GSL_LIBS = -L/sw/lib     -L/opt/local/lib
FFTW_INCL= -I/sw/include -I/opt/local/include
FFTW_LIBS= -L/sw/lib     -L/opt/local/lib
MPICHLIB = -lmpi 
HDF5INCL = -I/sw/lib     -I/opt/local/include -DH5_USE_16_API #-DUSE_SSE
HDF5LIB  = -L/sw/lib -L/opt/local/lib  -lhdf5 -lz
endif

ifeq ($(SYSTYPE),"Darwin-mpich")
CC       = mpicc   # sets the C-compiler
LINKER   = mpic++
OPTIMIZE = -m64 -ggdb -O3 -Wall -Wextra -Wno-format-security -Wno-unknown-pragmas
ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS)))
OPTIMIZE+= -fopenmp
endif
GSL_INCL = -I/opt/local/include 
GSL_LIBS = -L/opt/local/lib
FFTW_INCL= -I/opt/local/include
FFTW_LIBS= -L/opt/local/lib
HDF5INCL = -I/opt/local/include -DH5_USE_16_API
HDF5LIB  = -L/opt/local/lib  -lhdf5 -lz
CUDA_INCL= -I/Developer/NVIDIA/CUDA-5.0/include
CUDA_LIBS= -Xlinker -rpath /Developer/NVIDIA/CUDA-5.0/lib -L/Developer/NVIDIA/CUDA-5.0/lib -lcudart -lnvToolsExt -framework CUDA
NVCC     = /Developer/NVIDIA/CUDA-5.0/bin/nvcc
CUDA_OPTIMIZE = -g -G -O3 -m64 --ptxas-options=-v -Xptxas="-v" --maxrregcount=32 -arch=sm_30 $(filter -I%,$(shell mpicc -show))
endif


# modules for Magny
# module add mvapich2/gcc/64/1.6-qlc

ifeq ($(SYSTYPE),"Magny") 
CC       =   mpicc
OPTIMIZE =   -g -Wall -m64   -O3 -msse3 
ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS)))
OPTIMIZE +=  -fopenmp
else
OPTIMIZE +=  -Wno-unknown-pragmas
endif
GSL_INCL =  -I/hits/tap/sw/libs/gsl-1.15/include
GSL_LIBS =  -L/hits/tap/sw/libs/gsl-1.15/lib  -Xlinker -R -Xlinker /hits/tap/sw/libs/gsl-1.15/lib
FFTW_INCL=  -I/hits/tap/sw/libs/fftw-2.1.5/include
FFTW_LIBS=  -L/hits/tap/sw/libs/fftw-2.1.5/lib  -Xlinker -R -Xlinker /hits/tap/sw/libs/fftw-2.1.5/lib
GMP_INCL =  -I/hits/tap/sw/libs/gmp-5.0.5/include
GMP_LIBS =  -L/hits/tap/sw/libs/gmp-5.0.5/lib  -Xlinker -R -Xlinker /hits/tap/sw/libs/gmp-5.0.5/lib
MPICHLIB =
HDF5INCL =  -I/hits/tap/sw/libs/hdf5-1.8.10/include -DH5_USE_16_API
HDF5LIB  =  -L/hits/tap/sw/libs/hdf5-1.8.10/lib -lhdf5 -Xlinker -R -Xlinker /hits/tap/sw/libs/hdf5-1.8.10/lib
#OPT      +=  -DNOCALLSOFSYSTEM
#OPT      +=  -DIMPOSE_PINNING
#OPT      +=  -DUSE_SSE
endif


# modules for Nehalem cluster
# module add mvapich2/gcc/64/1.6
ifeq ($(SYSTYPE),"Nehalem") 
CC       =   mpicc
ifeq (SOFTDOUBLEDOUBLE,$(findstring SOFTDOUBLEDOUBLE,$(CONFIGVARS))) 
CC       =   mpicxx
endif
OPTIMIZE =  -O3 -msse3 -g -Wall -m64
GSL_INCL =  -I/hits/tap/sw/libs/include
GSL_LIBS =  -L/hits/tap/sw/libs/lib  -Xlinker -R -Xlinker /hits/tap/sw/libs/lib
FFTW_INCL=  -I/hits/tap/sw/libs/include
FFTW_LIBS=  -L/hits/tap/sw/libs/lib  -Xlinker -R -Xlinker /hits/tap/sw/libs/lib
MPICHLIB =
HDF5INCL =  -I/hits/tap/sw/nehalem/include -DH5_USE_16_API
#HDF5LIB  =  -L/hits/tap/sw/nehalem/lib -Xlinker -R -Xlinker /hits/tap/sw/nehalem/lib -lhdf5  
HDF5LIB  =  /hits/tap/sw/nehalem/lib/libhdf5.a -lz
#OPT      +=  -DNOCALLSOFSYSTEM
#OPT      +=  -DIMPOSE_PINNING
#OPT      +=  -DUSE_SSE
endif




# modules for Magny-Intel
# module load intel/compiler
# module load mvapich2/intel/64/1.6-qlc

ifeq ($(SYSTYPE),"Magny-Intel") 
CC       =   mpicc
OPTIMIZE =   -O2 -g -Wall -m64
ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS)))
OPTIMIZE +=  -openmp
else
OPTIMIZE +=  -Wno-unknown-pragmas
endif
GSL_INCL =  -I/hits/tap/sw/libs/include
GSL_LIBS =  -L/hits/tap/sw/libs/lib  -Xlinker -R -Xlinker /hits/tap/sw/libs/lib
FFTW_INCL=
FFTW_LIBS=
MPICHLIB =
MATHLIB  = -limf -lm
HDF5INCL = -DH5_USE_16_API
HDF5LIB  = -lhdf5
#OPT      +=  -DNOCALLSOFSYSTEM
OPT      +=  -DIMPOSE_PINNING
endif




ifndef LINKER
LINKER = $(CC)
endif





##########################################
#determine the needed object/header files#
##########################################

SUBDIRS = . 

OBJS =   main.o allocate.o  allvars.o  disk.o   grid.o  bulge.o  set_particles.o parallel_sort.o \
	     halo.o  init.o  io.o  mymalloc.o  orbit_response.o  parameters.o  structure.o  system.o  disp_fields.o \
	     forcetree/gravtree.o forcetree/forcetree.o forcetree/forcetree_walk.o domain/peano.o domain/pqueue.o \
	     domain/domain.o domain/domain_balance.o domain/domain_counttogo.o  domain/domain_exchange.o \
	     domain/domain_rearrange.o domain/domain_sort_kernels.o domain/domain_toplevel.o domain/domain_vars.o domain/domain_box.o


INCL += allvars.h proto.h

SUBDIRS += forcetree domain

################################
#determine the needed libraries#
################################


ifneq (HAVE_HDF5,$(findstring HAVE_HDF5,$(CONFIGVARS)))
HDF5LIB  = 
endif

ifeq (NUM_THREADS,$(findstring NUM_THREADS,$(CONFIGVARS)))
THREAD_LIB = 
endif


##########################
#combine compiler options#
##########################

CFLAGS = $(OPTIMIZE) $(OPT) $(HDF5INCL) $(GSL_INCL) $(FFTW_INCL) $(ODE_INCL) $(GMP_INCL) $(MKL_INCL) $(CUDA_INCL) -I$(BUILD_DIR)

CFLAGS_CUDA = $(CUDA_OPTIMIZE) $(OPT) $(GSL_INCL) $(FFTW_INCL) $(HDF5INCL) $(ODE_INCL) $(GMP_INCL) $(MKL_INCL) $(CUDA_INCL) -I$(BUILD_DIR)

LIBS = $(MATHLIB) $(HDF5LIB) $(MPICHLIB) $(GSL_LIBS) $(GSLLIB) $(FFTW_LIB) $(GMP_LIBS) $(GMPLIB) $(ODE_LIB) $(MKL_LIBS) $(THREAD_LIB) $(CUDA_LIBS)


SUBDIRS := $(addprefix $(BUILD_DIR)/,$(SUBDIRS))
OBJS := $(addprefix $(BUILD_DIR)/,$(OBJS)) $(BUILD_DIR)/compile_time_info.o
INCL := $(addprefix $(SRC_DIR)/,$(INCL)) $(BUILD_DIR)/galicconfig.h


################
#create subdirs#
################
RESULT := $(shell mkdir -p $(SUBDIRS)  )



#############
#build rules#
#############

all: $(EXEC)

$(EXEC): $(OBJS)
	$(LINKER) $(OPTIMIZE) $(OBJS) $(LIBS) -o $(EXEC)

clean:
	rm -f $(OBJS) $(EXEC) lib$(LIBRARY).a
	rm -f $(BUILD_DIR)/compile_time_info.c $(BUILD_DIR)/galicconfig.h

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(INCL) $(MAKEFILES)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/compile_time_info.o: $(BUILD_DIR)/compile_time_info.c $(MAKEFILES)
	$(CC) $(CFLAGS) -c $< -o $@
 
