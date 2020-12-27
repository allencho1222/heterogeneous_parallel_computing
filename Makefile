NVCC=nvcc
LIBS= -lOpenCL

GRAPH_SRC_DIR := ./graph_builder
GRAPH_OBJ_DIR := ./graph_objs

GRAPH_SRC_FILES := $(wildcard $(GRAPH_SRC_DIR)/*.cpp)
GRAPH_OBJ_FILES := $(patsubst $(GRAPH_SRC_DIR)/%.cpp, $(GRAPH_OBJ_DIR)/%.o, $(GRAPH_SRC_FILES))

main: main.cpp $(GRAPH_OBJ_FILES)
	$(NVCC) -std=c++14 $^ $(LIBS) -o main

$(GRAPH_OBJ_DIR)/%.o: $(GRAPH_SRC_DIR)/%.cpp
	$(NVCC) -std=c++14 -c -o $@ $<

clean:
	rm -rf main $(GRAPH_OBJ_DIR)/*.o
