CXX=g++

# flags configured by CMake
ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate -lmxnet
else
  LIB_FLAGS = -lopenblas -llapack -lmxnet
endif

SRC_DIR = src

OPT = -O3
DEBUGFLAGS = -rdynamic -g
CXXFLAGS = $(OPT) -std=c++1y -MMD -Wall $(DEBUGFLAGS)

all: main

OBJ = utils.o hyp_container.o logger.o thread_pool.o mlp.o

-include *.d

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIB_FLAGS)

main: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)


.PHONY: clean

clean:
	rm -f *.o *.d main

