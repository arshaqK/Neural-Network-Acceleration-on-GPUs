#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9


#ifdef __cplusplus
extern "C" {
#endif


// Timing structure
typedef struct {
    const char* name;
    double total_time;
} TimingInfo;

// Neural network structure for CPU
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Neural network structure for GPU
typedef struct {
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_target;
    double* d_d_hidden;
    double* d_d_output;
} GpuNeuralNetwork;

// Timing functions
int create_timer(const char* name);
void start_timing(clock_t* start_time);
double end_timing(clock_t start_time, int timer_id);
void print_timing_results(double total_time);

// CPU functions
double** allocateMatrix(int rows, int cols);
void freeMatrix(double** mat, int rows);
NeuralNetwork* createNetwork();
void forward_cpu(NeuralNetwork* net, double* input, double* hidden, double* output);
void freeNetwork(NeuralNetwork* net);

// GPU functions
GpuNeuralNetwork* createGpuNetwork(NeuralNetwork* cpu_net);
void forward_gpu(GpuNeuralNetwork* gpu_net, double* input);
void backward_gpu(GpuNeuralNetwork* gpu_net, double* target);
void copyWeightsFromGPU(NeuralNetwork* cpu_net, GpuNeuralNetwork* gpu_net);
void freeGpuNetwork(GpuNeuralNetwork* gpu_net);

// MNIST dataset functions
double** loadMNISTImages(const char* filename, int numImages);
double** loadMNISTLabels(const char* filename, int numLabels);

// CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernels declarations
__global__ void relu_kernel(double* x, int size);
__global__ void forward_hidden_kernel(double* input, double* W1, double* b1, double* hidden, int input_size, int hidden_size);
__global__ void forward_output_kernel(double* hidden, double* W2, double* b2, double* output, int hidden_size, int output_size);
__global__ void softmax_kernel(double* x, int size);
__global__ void compute_output_gradient_kernel(double* output, double* target, double* d_output, int size);
__global__ void compute_hidden_gradient_kernel(double* W2, double* d_output, double* hidden, double* d_hidden, int hidden_size, int output_size);
__global__ void update_output_weights_kernel(double* W2, double* d_output, double* hidden, double learning_rate, int hidden_size, int output_size);
__global__ void update_hidden_weights_kernel(double* W1, double* d_hidden, double* input, double learning_rate, int input_size, int hidden_size);
__global__ void update_output_bias_kernel(double* b2, double* d_output, double learning_rate, int output_size);
__global__ void update_hidden_bias_kernel(double* b1, double* d_hidden, double learning_rate, int hidden_size);

// External variables
extern TimingInfo timing_info[];
extern int num_timers;

#ifdef __cplusplus
}
#endif


#endif // NEURAL_NETWORK_H
