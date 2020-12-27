NVCC=nvcc

LIB_CUB_DIR := ./cub

GRAPH_SRC_DIR := ../graph_builder
GRAPH_OBJ_DIR := ../graph_objs

GRAPH_SRC_FILES := $(wildcard $(GRAPH_SRC_DIR)/*.cpp)
GRAPH_OBJ_FILES := $(patsubst $(GRAPH_SRC_DIR)/%.cpp, $(GRAPH_OBJ_DIR)/%.o, $(GRAPH_SRC_FILES))

all: pr bfs

pr: $(GRAPH_OBJ_FILES)
	$(NVCC) -std=c++14 -Xcompiler -fopenmp -I$(LIB_CUB_DIR) $^ pr.cu -o $@

bfs: $(GRAPH_OBJ_FILES)
	$(NVCC) -std=c++14 -Xcompiler -fopenmp -I$(LIB_CUB_DIR) $^ bfs.cu -o $@

$(GRAPH_OBJ_DIR)/%.o: $(GRAPH_SRC_DIR)/%.cpp
	$(NVCC) -std=c++14 -c -o $@ $<

clean:
	rm -rf pr bfs $(GRAPH_OBJ_DIR)/*.o
