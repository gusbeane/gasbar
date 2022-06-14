EXEC   = ComputeBarTorques 

OBJS   = src/main.o src/globvars.o \
	src/forcetree.o \
	src/nrsrc/bessi0.o   src/nrsrc/bessj1.o   \
	src/nrsrc/erff.o     src/nrsrc/polint.o   src/nrsrc/spline.o \
	src/nrsrc/bessi1.o   src/nrsrc/bessk0.o   src/nrsrc/gasdev.o  \
	src/nrsrc/qromb.o    src/nrsrc/splint.o  \
	src/nrsrc/bessj0.o   src/nrsrc/bessk1.o   \
	src/nrsrc/nrutil.o   src/nrsrc/ran1.o     src/nrsrc/trapzd.o \
	src/nrsrc/gammp.o    src/nrsrc/gcf.o      src/nrsrc/gser.o  \
	src/nrsrc/gammln.o   src/nrsrc/indexx.o   src/nrsrc/zriddr.o


INCL   = src/prototypes.h src/globvars.h src/nrsrc/nrutil.h src/nrsrc/nr.h

OPTIMIZE = -O2 -Wall

CFLAGS = $(OPTIMIZE) $(OPT)

LIBS   =  -lm -lhdf5 

CC     =  gcc


$(EXEC): $(OBJS) 
	cc $(CFLAGS) $(OBJS) -o $(EXEC)  $(LIBS)

$(OBJS): $(INCL) 

$(OBJS): $(INCL) 


clean:
	rm -f $(OBJS) $(EXEC)












