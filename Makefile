CXX=g++

# flags configured by CMake
ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate -lmxnet
else
  LIB_FLAGS = -lopenblas -llapack -lmxnet
endif

SRC_DIR = src
AUX_DIR = $(SRC_DIR)/aux

OPT = -O3
DEBUGFLAGS = -rdynamic -g
CXXFLAGS = $(OPT) -std=c++1y -MMD -Wall $(DEBUGFLAGS) -I$(AUX_DIR)

all: main

OBJ = utils.o hyp_container.o logger.o thread_pool.o lenet.o

-include *.d

%.o: $(AUX_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIB_FLAGS)

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIB_FLAGS)

main: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)


.PHONY: clean

clean:
	rm -f *.o *.d main

