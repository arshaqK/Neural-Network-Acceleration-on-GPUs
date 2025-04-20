#include "neural_network.h"

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
