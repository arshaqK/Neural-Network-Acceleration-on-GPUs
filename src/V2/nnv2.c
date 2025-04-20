// basic version of v2 for arshaq to optimize

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

// Timing structure
typedef struct {
    const char* name;
    double total_time;
} TimingInfo;

// Global timing variables
TimingInfo timing_info[20];
int num_timers = 0;

// Create a new timer
int create_timer(const char* name) {
    int id = num_timers++;
    timing_info[id].name = name;
    timing_info[id].total_time = 0.0;
    return id;
}

// Direct timing measurement
void start_timing(clock_t* start_time) {
    *start_time = clock();
}

double end_timing(clock_t start_time, int timer_id) {
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    timing_info[timer_id].total_time += elapsed;
    return elapsed;
}

// Print timing results
void print_timing_results(double total_time) {
    printf("\n----- TIMING RESULTS -----\n");
    printf("%-30s %-12s %-12s\n", "Component", "Time (s)", "Percentage");
    printf("----------------------------------------------------------\n");
    
    for (int i = 0; i < num_timers; i++) {
        printf("%-30s %-12.3f %-12.2f%%\n", 
               timing_info[i].name, 
               timing_info[i].total_time, 
               (timing_info[i].total_time / total_time) * 100);
    }
    printf("----------------------------------------------------------\n");
    printf("%-30s %-12.3f %-12.2f%%\n", "Total Time", total_time, 100.0);
    printf("----------------------------------------------------------\n");
}

// CUDA error checking
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

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

// CUDA kernels
__global__ void relu_kernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0) ? x[idx] : 0;
    }
}

__global__ void forward_hidden_kernel(double* input, double* W1, double* b1, double* hidden, int input_size, int hidden_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_idx < hidden_size) {
        double sum = b1[h_idx];
        for (int i = 0; i < input_size; i++) {
            sum += W1[h_idx * input_size + i] * input[i];
        }
        hidden[h_idx] = sum;
    }
}

__global__ void forward_output_kernel(double* hidden, double* W2, double* b2, double* output, int hidden_size, int output_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (o_idx < output_size) {
        double sum = b2[o_idx];
        for (int i = 0; i < hidden_size; i++) {
            sum += W2[o_idx * hidden_size + i] * hidden[i];
        }
        output[o_idx] = sum;
    }
}

__global__ void softmax_kernel(double* x, int size) {
    // Find max value for numerical stability
    double max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp(x - max) and sum
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

__global__ void compute_output_gradient_kernel(double* output, double* target, double* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = output[idx] - target[idx];
    }
}

__global__ void compute_hidden_gradient_kernel(double* W2, double* d_output, double* hidden, double* d_hidden, 
                                              int hidden_size, int output_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h_idx < hidden_size) {
        double sum = 0;
        for (int j = 0; j < output_size; j++) {
            sum += W2[j * hidden_size + h_idx] * d_output[j];
        }
        d_hidden[h_idx] = sum * (hidden[h_idx] > 0);
    }
}

__global__ void update_output_weights_kernel(double* W2, double* d_output, double* hidden, double learning_rate,
                                            int hidden_size, int output_size) {
    int o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (o_idx < output_size && h_idx < hidden_size) {
        W2[o_idx * hidden_size + h_idx] -= learning_rate * d_output[o_idx] * hidden[h_idx];
    }
}

__global__ void update_hidden_weights_kernel(double* W1, double* d_hidden, double* input, double learning_rate,
                                           int input_size, int hidden_size) {
    int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h_idx < hidden_size && i_idx < input_size) {
        W1[h_idx * input_size + i_idx] -= learning_rate * d_hidden[h_idx] * input[i_idx];
    }
}

__global__ void update_output_bias_kernel(double* b2, double* d_output, double learning_rate, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        b2[idx] -= learning_rate * d_output[idx];
    }
}

__global__ void update_hidden_bias_kernel(double* b1, double* d_hidden, double learning_rate, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        b1[idx] -= learning_rate * d_hidden[idx];
    }
}

// Initialize neural network on CPU
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Initialize GPU neural network
GpuNeuralNetwork* createGpuNetwork(NeuralNetwork* cpu_net) {
    GpuNeuralNetwork* gpu_net = (GpuNeuralNetwork*)malloc(sizeof(GpuNeuralNetwork));
    
    // Flatten CPU matrices for GPU
    double* flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            flat_W1[i * INPUT_SIZE + j] = cpu_net->W1[i][j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            flat_W2[i * HIDDEN_SIZE + j] = cpu_net->W2[i][j];
        }
    }
    
    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_b2, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&gpu_net->d_d_output, OUTPUT_SIZE * sizeof(double)));
    
    // Copy data to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_net->d_W1, flat_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_net->d_W2, flat_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_net->d_b1, cpu_net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_net->d_b2, cpu_net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(flat_W1);
    free(flat_W2);
    
    return gpu_net;
}

// Copy weights back from GPU to CPU
void copyWeightsFromGPU(NeuralNetwork* cpu_net, GpuNeuralNetwork* gpu_net) {
    double* flat_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* flat_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W1, gpu_net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(flat_W2, gpu_net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpu_net->b1, gpu_net->d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(cpu_net->b2, gpu_net->d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            cpu_net->W1[i][j] = flat_W1[i * INPUT_SIZE + j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            cpu_net->W2[i][j] = flat_W2[i * HIDDEN_SIZE + j];
        }
    }
    
    free(flat_W1);
    free(flat_W2);
}

// GPU forward pass
void forward_gpu(GpuNeuralNetwork* gpu_net, double* input) {
    int blockSize = 128;
    
    // Copy input to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_net->d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Hidden layer computation
    dim3 hidden_blocks((HIDDEN_SIZE + blockSize - 1) / blockSize);
    forward_hidden_kernel<<<hidden_blocks, blockSize>>>(
        gpu_net->d_input, gpu_net->d_W1, gpu_net->d_b1, gpu_net->d_hidden, INPUT_SIZE, HIDDEN_SIZE
    );
    
    // Apply ReLU activation
    relu_kernel<<<hidden_blocks, blockSize>>>(gpu_net->d_hidden, HIDDEN_SIZE);
    
    // Output layer computation
    dim3 output_blocks((OUTPUT_SIZE + blockSize - 1) / blockSize);
    forward_output_kernel<<<output_blocks, blockSize>>>(
        gpu_net->d_hidden, gpu_net->d_W2, gpu_net->d_b2, gpu_net->d_output, HIDDEN_SIZE, OUTPUT_SIZE
    );
    
    // Apply softmax activation
    softmax_kernel<<<1, 1>>>(gpu_net->d_output, OUTPUT_SIZE);
    
    // Ensure all kernels finished
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// GPU backward pass
void backward_gpu(GpuNeuralNetwork* gpu_net, double* target) {
    int blockSize = 128;
    
    // Copy target to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_net->d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    
    // Compute output layer gradient
    dim3 output_blocks((OUTPUT_SIZE + blockSize - 1) / blockSize);
    compute_output_gradient_kernel<<<output_blocks, blockSize>>>(
        gpu_net->d_output, gpu_net->d_target, gpu_net->d_d_output, OUTPUT_SIZE
    );
    
    // Compute hidden layer gradient
    dim3 hidden_blocks((HIDDEN_SIZE + blockSize - 1) / blockSize);
    compute_hidden_gradient_kernel<<<hidden_blocks, blockSize>>>(
        gpu_net->d_W2, gpu_net->d_d_output, gpu_net->d_hidden, gpu_net->d_d_hidden, HIDDEN_SIZE, OUTPUT_SIZE
    );
    
    // Update weights and biases
    dim3 block_dim_w2(16, 16);
    dim3 grid_dim_w2((OUTPUT_SIZE + block_dim_w2.x - 1) / block_dim_w2.x,
                     (HIDDEN_SIZE + block_dim_w2.y - 1) / block_dim_w2.y);
    update_output_weights_kernel<<<grid_dim_w2, block_dim_w2>>>(
        gpu_net->d_W2, gpu_net->d_d_output, gpu_net->d_hidden, LEARNING_RATE, HIDDEN_SIZE, OUTPUT_SIZE
    );
    
    dim3 block_dim_w1(16, 16);
    dim3 grid_dim_w1((HIDDEN_SIZE + block_dim_w1.x - 1) / block_dim_w1.x,
                     (INPUT_SIZE + block_dim_w1.y - 1) / block_dim_w1.y);
    update_hidden_weights_kernel<<<grid_dim_w1, block_dim_w1>>>(
        gpu_net->d_W1, gpu_net->d_d_hidden, gpu_net->d_input, LEARNING_RATE, INPUT_SIZE, HIDDEN_SIZE
    );
    
    update_output_bias_kernel<<<output_blocks, blockSize>>>(
        gpu_net->d_b2, gpu_net->d_d_output, LEARNING_RATE, OUTPUT_SIZE
    );
    
    update_hidden_bias_kernel<<<hidden_blocks, blockSize>>>(
        gpu_net->d_b1, gpu_net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE
    );
    
    // Ensure all kernels finished
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// Free GPU network memory
void freeGpuNetwork(GpuNeuralNetwork* gpu_net) {
    cudaFree(gpu_net->d_W1);
    cudaFree(gpu_net->d_W2);
    cudaFree(gpu_net->d_b1);
    cudaFree(gpu_net->d_b2);
    cudaFree(gpu_net->d_input);
    cudaFree(gpu_net->d_hidden);
    cudaFree(gpu_net->d_output);
    cudaFree(gpu_net->d_target);
    cudaFree(gpu_net->d_d_hidden);
    cudaFree(gpu_net->d_d_output);
    free(gpu_net);
}

// CPU forward pass for evaluation
void forward_cpu(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Hidden layer calculation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    
    // Apply ReLU activation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;
    }

    // Output layer calculation
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    
    // Apply softmax activation
    double max_val = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) max_val = output[i];
    }
    
    double sum = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = exp(output[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] /= sum;
    }
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network Performance Profiling (CUDA V2)\n\n");
    
    // Initialize timers
    int timer_total = create_timer("Total Execution");
    int timer_load_train_images = create_timer("Load Training Images");
    int timer_load_train_labels = create_timer("Load Training Labels");
    int timer_load_test_images = create_timer("Load Test Images");
    int timer_load_test_labels = create_timer("Load Test Labels");
    int timer_network_creation = create_timer("Network Creation");
    int timer_gpu_init = create_timer("GPU Initialization");
    int timer_training_total = create_timer("Training (Total)");
    int timer_forward_pass = create_timer("Forward Pass (Total)");
    int timer_backward_pass = create_timer("Backward Pass (Total)");
    int timer_evaluation = create_timer("Test Evaluation");
    int timer_cleanup = create_timer("Memory Cleanup");
    
    clock_t total_start = clock();
    
    // Load training data
    clock_t start_time;
    
    start_time = clock();
    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    end_timing(start_time, timer_load_train_images);
    
    start_time = clock();
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    end_timing(start_time, timer_load_train_labels);
    
    start_time = clock();
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    end_timing(start_time, timer_load_test_images);
    
    start_time = clock();
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);
    end_timing(start_time, timer_load_test_labels);
    
    // Create network on CPU
    start_time = clock();
    NeuralNetwork* net = createNetwork();
    end_timing(start_time, timer_network_creation);
    
    // Initialize GPU network
    start_time = clock();
    GpuNeuralNetwork* gpu_net = createGpuNetwork(net);
    end_timing(start_time, timer_gpu_init);
    
    // Training
    double* output_cpu = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    clock_t training_start = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        int correct = 0;
        clock_t epoch_start = clock();
        
        for (int i = 0; i < 60000; i++) {
            // Forward pass timing
            clock_t forward_start = clock();
            forward_gpu(gpu_net, train_images[i]);
            CHECK_CUDA_ERROR(cudaMemcpy(output_cpu, gpu_net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));
            timing_info[timer_forward_pass].total_time += (double)(clock() - forward_start) / CLOCKS_PER_SEC;
            
            // Backward pass timing
            clock_t backward_start = clock();
            backward_gpu(gpu_net, train_labels[i]);
            timing_info[timer_backward_pass].total_time += (double)(clock() - backward_start) / CLOCKS_PER_SEC;
            
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= train_labels[i][k] * log(output_cpu[k] + 1e-10);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output_cpu[j] > output_cpu[pred]) pred = j;
                if (train_labels[i][j] > train_labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        double epoch_time = (double)(clock() - epoch_start) / CLOCKS_PER_SEC;
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / 60000, (correct / 60000.0) * 100, epoch_time);
    }
    
    end_timing(training_start, timer_training_total);
    
    // Copy trained weights back to CPU for evaluation
    copyWeightsFromGPU(net, gpu_net);
    
    // Test evaluation
    start_time = clock();
    int correct = 0;
    double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
    
    for (int i = 0; i < 10000; i++) {
        forward_cpu(net, test_images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (test_labels[i][j] > test_labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    end_timing(start_time, timer_evaluation);
    printf("Test Accuracy: %.2f%%\n", (correct / 10000.0) * 100);
    
    // Memory cleanup
    start_time = clock();
    freeNetwork(net);
    freeGpuNetwork(gpu_net);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    free(output_cpu);
    end_timing(start_time, timer_cleanup);
    
    // Calculate total time
    double total_time = (double)(clock() - total_start) / CLOCKS_PER_SEC;
    timing_info[timer_total].total_time = total_time;
    
    // Print timing results
    print_timing_results(total_time);
    
    return 0;
}