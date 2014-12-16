# -*- makefile -*-

OBJDIR = obj
BINDIR = bin

EXESUFFIX=
EXESRC = test.cpp
TESTSRC =  
PROG_SRC = test.cpp

# source files.
SRC_ALL = $(shell ls *.cpp) ##(shell find . -name '[a-zA-Z]*'.cpp)
FILTER_OUT = $(foreach v,$(2),$(if $(findstring $(v),$(1)),,$(v)))
SRC = $(filter-out $(TESTSRC),$(call FILTER_OUT, $(EXESRC), $(SRC_ALL)))

#HFILES = $(patsubst %.cpp,%.h,$(SRC))
HFILES = $(shell ls *.hpp)

OBJ = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRC))
GDBOBJS = $(patsubst %.cpp,$(OBJDIR)/%-gdb.o,$(SRC))
PROFOBJS = $(patsubst %.cpp,$(OBJDIR)/%-gprof.o,$(SRC))

PROG_EXE= $(patsubst %.cpp,$(BINDIR)/%,$(PROG_SRC))
GDBEXE = $(patsubst %.cpp,$(BINDIR)/%-gdb,$(PROG_SRC))
PROFEXE = $(patsubst %.cpp,$(BINDIR)/%-gprof,$(PROG_SRC))

# include directories
INCLUDES = -Ieigen  #-I./boost/include -I./glpk/build/include

# C++ compiler flags (-g -O2 -Wall)
CCFLAGS = -Wall -static  -Werror -Wno-deprecated -Werror=sign-compare -std=c++0x

# library paths
LIBINCLUDE = #-L./boost/lib -L./glpk/build/lib
LIBFLAGS = $(LIBINCLUDE) #-lglpk -lboost_program_options 

# compiler
CC = g++ $(CCFLAGS) -O3
CCGDB = g++ -O0 -ggdb $(CCFLAGS)
CCGPROF = g++ -O0 -g -pg $(CCFLAGS)


.PHONY: depend clean
.SUFFIXES: .cpp
.PRECIOUS: $(OBJDIR)/%.o

default: $(PROG_EXE)

gdb: $(GDBEXE)

gprof: $(PROFEXE)

$(OBJDIR)/%.o: %.cpp
	$(CC) $(INCLUDES) $(LIBFLAGS) -c $< -o $@

$(OBJDIR)/%-gdb.o: %.cpp
	$(CCGDB) $(INCLUDES) $(LIBFLAGS) -c $< -o $@

$(OBJDIR)/%-gprof.o: %.cpp
	$(CCGPROF) $(INCLUDES) $(LIBFLAGS) -c $< -o $@

$(BINDIR)/%: %.cpp $(OBJ)
	$(CC) $(INCLUDES) $(OBJ) $(LIBFLAGS) $< -o $@

$(BINDIR)/%-gdb: %.cpp $(GDBOBJS)
	$(CCGDB) $(INCLUDES) $(GDBOBJS) $(LIBFLAGS) $< -o $@

$(BINDIR)/%-gprof: %.cpp $(PROFOBJS)
	$(CCGPROF) $(INCLUDES) $(PROFOBJS) $(LIBFLAGS) $< -o $@

depend: make.depend $(HFILES) $(SRC_ALL)

make.depend:
	$(CC) $(INCLUDES) -MM $(SRC_ALL) > make.depend

clean:
	@rm -f $(OBJ) $(GDBOBJS) $(PROFOBJS) $(PROG_EXE) $(GDBEXE) $(PROFEXE) makefile.bak *~


include make.depend

