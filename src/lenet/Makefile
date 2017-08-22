CXX=g++

# flags configured by CMake
ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate -lmxnet
else
  LIB_FLAGS = -lopenblas -llapack -lmxnet
endif

AUX_DIR = ../aux

OPT = -O3
DEBUGFLAGS = -rdynamic -g
CXXFLAGS = $(OPT) -std=c++1y -MMD -Wall $(DEBUGFLAGS) -I$(AUX_DIR)

all: main

AUX_OBJ = $(AUX_DIR)/utils.o $(AUX_DIR)/hyp_container.o $(AUX_DIR)/logger.o $(AUX_DIR)/thread_pool.o
OBJ = lenet.o

-include *.d

$(AUX_DIR)/%.o: $(AUX_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIB_FLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIB_FLAGS)

main: $(AUX_OBJ) $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIB_FLAGS)


.PHONY: clean

clean:
	rm -f *.o *.d main

