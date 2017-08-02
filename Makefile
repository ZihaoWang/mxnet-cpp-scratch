CXX=g++

# flags configured by CMake
ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate -lmxnet
else
  LIB_FLAGS = -lopenblas -llapack -lmxnet
endif

OPT = -O3
DEBUGFLAGS = -rdynamic -g
CXXFLAGS = $(OPT) -std=c++1y -MMD -Wall $(DEBUGFLAGS)

all: main

DEPS = 
OBJ = utils.o logger.o mlp_gpu.o

-include *.d

%.o: %.c $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LIB_FLAGS)

main: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)


.PHONY: clean

clean:
	rm -f *.o *.d main

