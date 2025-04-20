# Compiler settings
NVCC = nvcc
CC = gcc
CFLAGS = -O3 -Wall
NVCC_FLAGS = -O3 -arch=sm_86 # Adjust the architecture as needed for your GPU

# Source files
CPU_SOURCES = main.c timing.c cpu_functions.c
GPU_SOURCES = gpu_functions.cu

# Object files
OBJECTS = $(CPU_SOURCES:.c=.o) $(GPU_SOURCES:.cu=.o)

# Target executable
TARGET = neural_network

# Default target
all: $(TARGET)

# Link the program
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET) -lcudart -lm

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
