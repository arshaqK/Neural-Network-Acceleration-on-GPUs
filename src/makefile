# CC = gcc
# CFLAGS = -Wall -O2

# EXE = nn.exe
# SRC = nn.c

# all: $(EXE) run

# $(EXE): $(SRC)
# 	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

# run: $(EXE)
# 	./$(EXE)

# clean:
# 	rm -f $(EXE)
# Compiler definitions
# Compiler flags
NVCC = nvcc
GCC = gcc

# Compiler flags for CUDA and C
NVCCFLAGS = -c -I V3/
GCCFLAGS = -c -I V3/
LDFLAGS = -lcudart

# Source and object files
CUDA_SRC = V3/kernel.cu
C_SRC = nn.c
CUDA_OBJ = kernel.o
C_OBJ = nn.o
EXEC = mnist_nn

# Default target
all: $(EXEC)

# Link object files to create executable
$(EXEC): $(CUDA_OBJ) $(C_OBJ)
	$(NVCC) $(C_OBJ) $(CUDA_OBJ) -o $(EXEC) $(LDFLAGS)

# Compile CUDA source
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRC) -o $(CUDA_OBJ)

# Compile C source with nvcc (since it includes cuda_runtime.h)
$(C_OBJ): $(C_SRC)
	$(NVCC) $(NVCCFLAGS) $(C_SRC) -o $(C_OBJ)

# Run the executable
run: $(EXEC)
	./$(EXEC)

# Clean generated files
clean:
	rm -f $(CUDA_OBJ) $(C_OBJ) $(EXEC)

# Phony targets
.PHONY: all run clean
