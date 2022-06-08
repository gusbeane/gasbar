EXEC   = ComputeFourier 

OBJS   = src/main.o

INCL   =

OPTIMIZE = -Wall -g -ggdb

CFLAGS = $(OPTIMIZE) $(OPT)

LIBS   =  -lm -lhdf5 

CC     =  gcc


$(EXEC): $(OBJS) 
	cc $(CFLAGS) $(OBJS) -o $(EXEC)  $(LIBS)

$(OBJS): $(INCL) 

$(OBJS): $(INCL) 


clean:
	rm -f $(OBJS) $(EXEC)












