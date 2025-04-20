
//this is the main kernel file for the v3 implementation that will focus on optimizations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "connector.h"

//using all the previously defined constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 8
#define BATCH_SIZE 32
#define TILE_SIZE 32
// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device memory for neural network
struct CudaNetwork {
    float *d_W1, *d_W2;
    float *d_b1, *d_b2;
    
    // Batch processing memory
    float *d_batch_input, *d_batch_hidden, *d_batch_output, *d_batch_target;
    float *d_batch_d_hidden, *d_batch_d_output;
};

//device memory structure

//kernel definitions start here.

//matrix multiplication using shared memory for better accesses.

// Improved matrixMultiply kernel with better boundary checks
__global__ void matrixMultiply(float* A, float* B, float* C, float* bias, int Arows, int Acols, int Bcols, int batch_size) {
    // Shared memory declarations
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    // Early exit for out-of-bounds threads
    if (batch >= batch_size) return;

    float* batchB = B + batch * Acols * Bcols;
    float* batchC = C + batch * Arows * Bcols;

    // Initialize accumulator with bias (only valid rows)
    float sum = (row < Arows) ? bias[row] : 0.0f;

    // Process input matrix in tiles
    for (int i = 0; i < Acols; i += 32) {
        // Load tiles with proper boundary checks
        if (threadIdx.y < 32 && threadIdx.x < 32) {
            tileA[threadIdx.y][threadIdx.x] = (row < Arows && i + threadIdx.x < Acols) ?
                                             A[row * Acols + i + threadIdx.x] : 0.0f;
            
            tileB[threadIdx.y][threadIdx.x] = (i + threadIdx.y < Acols && col < Bcols) ?
                                             batchB[(i + threadIdx.y) * Bcols + col] : 0.0f;
        }
        __syncthreads();

        // Compute partial dot product with bounds check
        if (row < Arows && col < Bcols) {
            for (int j = 0; j < 32 && (i + j) < Acols; j++) {
                sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Store result with bounds check
    if (row < Arows && col < Bcols) {
        batchC[row * Bcols + col] = sum;
    }
}

// Improved transposed matrix multiplication
__global__ void matrixMultiplyTranspose(float* A, float* B, float* C, int Arows, int Acols, int batch_size) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];


    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row in result (corresponds to columns of A)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column in result (single column)
    int batch = blockIdx.z;
    
    // Early exit for out-of-bounds threads
    if (batch >= batch_size) return;

    float* batchB = B + batch * Arows; // B is Arows x 1 per batch
    float* batchC = C + batch * Acols; // C is Acols x 1 per batch

    float sum = 0.0f;

    // Process in tiles
    for (int i = 0; i < Arows; i += 32) {
        if (threadIdx.y < 32 && threadIdx.x < 32) {
            // Load transposed data from A with bounds check
            tileA[threadIdx.y][threadIdx.x] = (row < Acols && i + threadIdx.x < Arows) ?
                                             A[(i + threadIdx.x) * Acols + row] : 0.0f;
            
            // Load data from B
            tileB[threadIdx.y][threadIdx.x] = (i + threadIdx.y < Arows) ?
                                             batchB[i + threadIdx.y] : 0.0f;
        }
        
        __syncthreads();

        // Compute partial dot product
        if (row < Acols) {
            for (int j = 0; j < 32 && (i + j) < Arows; j++) {
                sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
            }
        }
        
        __syncthreads();
    }

    // Store result with bounds check
    if (row < Acols && col == 0) {
        batchC[row] = sum;
    }
}

// Improved ReLU activation with better vectorization
__global__ void relu(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes up to 4 elements
    int baseIdx = idx * 4;
    
    // Process elements with bounds checking
    if (baseIdx < size) {
        // Process first element
        x[baseIdx] = fmaxf(0.0f, x[baseIdx]);
        
        // Process additional elements with bounds checking
        if (baseIdx + 1 < size) x[baseIdx + 1] = fmaxf(0.0f, x[baseIdx + 1]);
        if (baseIdx + 2 < size) x[baseIdx + 2] = fmaxf(0.0f, x[baseIdx + 2]);
        if (baseIdx + 3 < size) x[baseIdx + 3] = fmaxf(0.0f, x[baseIdx + 3]);
    }
}

// Improved ReLU derivative with bounds checking
__global__ void reluDerivativeKernel(float* x, float* dx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int baseIdx = idx * 4;
    
    if (baseIdx < size) {
        // Element-wise multiplication by derivative of ReLU
        dx[baseIdx] *= (x[baseIdx] > 0.0f) ? 1.0f : 0.0f;
        
        // Process additional elements with bounds checking
        if (baseIdx + 1 < size) 
            dx[baseIdx + 1] *= (x[baseIdx + 1] > 0.0f) ? 1.0f : 0.0f;
        if (baseIdx + 2 < size) 
            dx[baseIdx + 2] *= (x[baseIdx + 2] > 0.0f) ? 1.0f : 0.0f;
        if (baseIdx + 3 < size) 
            dx[baseIdx + 3] *= (x[baseIdx + 3] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Improved softmax with better numerical stability
__global__ void softmax(float* x, int batch_size, int size) {
    extern __shared__ float shared[];
    float* sData = shared;
    float* sMax = &shared[blockDim.x];
    float* sSum = &shared[blockDim.x + 1];
    
    int batchIndex = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batchIndex >= batch_size) return;
    
    // Offset for current batch
    float* batchX = x + batchIndex * size;
    
    // Find maximum value for numerical stability
    float maxVal = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        maxVal = fmaxf(maxVal, batchX[i]);
    }
    
    // Reduce to find batch max
    sData[tid] = maxVal;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sData[tid] = fmaxf(sData[tid], sData[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sMax[0] = sData[0];
    }
    __syncthreads();
    
    maxVal = sMax[0];
    
    // Compute exp(x - max) and sum
    float partialSum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float val = expf(batchX[i] - maxVal);
        batchX[i] = val;  // Store exp values temporarily
        partialSum += val;
    }
    
    // Reduce to find sum
    sData[tid] = partialSum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sData[tid] += sData[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        sSum[0] = sData[0];
    }
    __syncthreads();
    
    float sum = sSum[0];
    
    // Normalize with sum
    for (int i = tid; i < size; i += blockDim.x) {
        batchX[i] /= sum;
        
        // Add epsilon for numerical stability
        batchX[i] = fmaxf(batchX[i], 1e-15f);
    }
}

// Improved output gradient calculation
__global__ void OutputGradient(float* output, float* target, float* gradient, int batch_size, int size) {
    extern __shared__ float shared[];
    float* outS = shared;
    float* targetS = &shared[size];
    
    int batchIndex = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batchIndex >= batch_size) return;
    
    int offset = batchIndex * size;
    
    // Load data into shared memory with bounds checking
    for (int i = tid; i < size; i += blockDim.x) {
        outS[i] = output[offset + i];
        targetS[i] = target[offset + i];
    }
    __syncthreads();
    
    // Calculate gradient (output - target) with proper bounds checking
    for (int i = tid; i < size; i += blockDim.x) {
        gradient[offset + i] = outS[i] - targetS[i];
    }
}

// Improved weight update kernel
__global__ void weightUpdate(float* weights, float* inputs, float* gradients,
                            int batch_size, int output_dim, int input_dim, float lr) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int inCol = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds early
    if (outRow >= output_dim || inCol >= input_dim) return;
    
    // Calculate weight index
    int weightIdx = outRow * input_dim + inCol;
    float weightDelta = 0.0f;
    
    // Process all batches to calculate weight update
    for (int b = 0; b < batch_size; b++) {
        float input_val = inputs[b * input_dim + inCol];
        float grad_val = gradients[b * output_dim + outRow];
        weightDelta += grad_val * input_val;
    }
    
    // Apply weight update with proper scaling
    weights[weightIdx] -= lr * (weightDelta / batch_size);
}

// Improved bias update
__global__ void biasUpdate(float* biases, float* gradients,
                          int batch_size, int dim, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= dim) return;
    
    float bias_delta = 0.0f;
    
    // Sum gradients across batch
    for (int b = 0; b < batch_size; b++) {
        bias_delta += gradients[b * dim + idx];
    }
    
    // Update bias
    biases[idx] -= lr * (bias_delta / batch_size);
}

// Improved cross-entropy loss calculation
__global__ void crossEntropyLoss(float* output, float* target, float* loss, int batch_size, int size) {
    extern __shared__ float s_data[];
    float* s_partial = s_data;  // Use entire shared memory for partial sums
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * size;
    
    // Compute partial loss with bounds checking and numerical stability
    float partial_loss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float out_val = fmaxf(output[offset + i], 1e-15f);  // Avoid log(0)
        float target_val = target[offset + i];
        partial_loss -= target_val * logf(out_val);
    }
    
    // Store partial sum
    s_partial[tid] = partial_loss;
    __syncthreads();
    
    // Reduce to get total loss for this batch
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final loss for the batch
    if (tid == 0) {
        loss[batch_idx] = s_partial[0];
    }
}

// Improved accuracy kernel
__global__ void accuracyKernel(float* output, float* target, int* correct,
                              int batch_size, int output_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int offset = batch_idx * output_size;
    int pred_class = 0;
    int true_class = 0;
    
    // Find predicted class (max output value)
    for (int i = 1; i < output_size; i++) {
        if (output[offset + i] > output[offset + pred_class]) {
            pred_class = i;
        }
    }
    
    // Find true class (one-hot encoded)
    for (int i = 0; i < output_size; i++) {
        if (target[offset + i] > 0.5f) {
            true_class = i;
            break;
        }
    }
    
    // Record if prediction was correct
    correct[batch_idx] = (pred_class == true_class) ? 1 : 0;
}

// main functions that utilize the kernel and will be called in the c file which is the main file.
// Create CUDA network from host network
CudaNetwork* createCudaNetwork(double** W1, double** W2, double* b1, double* b2) {
    CudaNetwork* cuda_net = (CudaNetwork*)malloc(sizeof(CudaNetwork));
    
    // Allocate device memory for weights and biases
    CUDA_CHECK(cudaMalloc(&cuda_net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_b2, OUTPUT_SIZE * sizeof(float)));
    
    // Allocate device memory for batch processing
    CUDA_CHECK(cudaMalloc(&cuda_net->d_batch_input, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_batch_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_batch_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_batch_target, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_batch_d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cuda_net->d_batch_d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
    // Convert host double matrices to float arrays for CUDA
    float* h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* h_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Copy and convert data from double to float
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_b1[i] = (float)b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_W1[i*INPUT_SIZE + j] = (float)W1[i][j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_b2[i] = (float)b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_W2[i*HIDDEN_SIZE + j] = (float)W2[i][j];
        }
    }
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(cuda_net->d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_net->d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_net->d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuda_net->d_b2, h_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free temporary host memory
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);
    
    return cuda_net;
}

// Forward pass
// Improved forward pass implementation
void cudaForward(CudaNetwork* net, int batch_size) {
    // First layer: input -> hidden
    dim3 blockDim1(32, 16);
    dim3 gridDim1(
        (1 + blockDim1.x - 1) / blockDim1.x,
        (HIDDEN_SIZE + blockDim1.y - 1) / blockDim1.y,
        batch_size
    );
    
    matrixMultiply<<<gridDim1, blockDim1>>>(
        net->d_W1, net->d_batch_input, net->d_batch_hidden, net->d_b1,
        HIDDEN_SIZE, INPUT_SIZE, 1, batch_size);
    
    // Apply ReLU activation
    int block_size = 256;
    int grid_size = ((batch_size * HIDDEN_SIZE + 4 - 1) / 4 + block_size - 1) / block_size;
    relu<<<grid_size, block_size>>>(net->d_batch_hidden, batch_size * HIDDEN_SIZE);
    
    // Second layer: hidden -> output
    dim3 blockDim2(32, 16);
    dim3 gridDim2(
        (1 + blockDim2.x - 1) / blockDim2.x,
        (OUTPUT_SIZE + blockDim2.y - 1) / blockDim2.y,
        batch_size
    );
    
    matrixMultiply<<<gridDim2, blockDim2>>>(
        net->d_W2, net->d_batch_hidden, net->d_batch_output, net->d_b2,
        OUTPUT_SIZE, HIDDEN_SIZE, 1, batch_size);
    
    // Apply softmax
    block_size = 256;
    grid_size = batch_size;
    size_t shared_mem = (block_size + 2) * sizeof(float); // For data, max, sum
    softmax<<<grid_size, block_size, shared_mem>>>(net->d_batch_output, batch_size, OUTPUT_SIZE);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in forward pass: %s\n", cudaGetErrorString(err));
    }
}

// Improved backward pass implementation
void cudaBackward(CudaNetwork* net, int batch_size) {
    // Calculate output gradients
    int block_size = 256;
    int grid_size = batch_size;
    size_t shared_mem = 2 * OUTPUT_SIZE * sizeof(float); // For output and target data
    
    OutputGradient<<<grid_size, block_size, shared_mem>>>(
        net->d_batch_output, net->d_batch_target, net->d_batch_d_output, 
        batch_size, OUTPUT_SIZE);
    
    // Calculate hidden layer gradients using transposed weights
    dim3 blockDim1(16, 16);
    dim3 gridDim1(
        1, // Only one column in result
        (HIDDEN_SIZE + blockDim1.y - 1) / blockDim1.y,
        batch_size
    );
    
    matrixMultiplyTranspose<<<gridDim1, blockDim1>>>(
        net->d_W2, net->d_batch_d_output, net->d_batch_d_hidden,
        OUTPUT_SIZE, HIDDEN_SIZE, batch_size);
    
    // Apply ReLU derivative
    block_size = 256;
    grid_size = ((batch_size * HIDDEN_SIZE + 4 - 1) / 4 + block_size - 1) / block_size;
    
    reluDerivativeKernel<<<grid_size, block_size>>>(
        net->d_batch_hidden, net->d_batch_d_hidden, batch_size * HIDDEN_SIZE);
    
    // Update weights - simplified approach without tiling for reliability
    dim3 blockW2(16, 16);
    dim3 gridW2(
        (HIDDEN_SIZE + blockW2.x - 1) / blockW2.x,
        (OUTPUT_SIZE + blockW2.y - 1) / blockW2.y
    );
    
    weightUpdate<<<gridW2, blockW2>>>(
        net->d_W2, net->d_batch_hidden, net->d_batch_d_output,
        batch_size, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    
    dim3 blockW1(16, 16);
    dim3 gridW1(
        (INPUT_SIZE + blockW1.x - 1) / blockW1.x,
        (HIDDEN_SIZE + blockW1.y - 1) / blockW1.y
    );
    
    weightUpdate<<<gridW1, blockW1>>>(
        net->d_W1, net->d_batch_input, net->d_batch_d_hidden,
        batch_size, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    
    // Update biases
    block_size = 256;
    int b2_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
    biasUpdate<<<b2_blocks, block_size>>>(
        net->d_b2, net->d_batch_d_output, batch_size, OUTPUT_SIZE, LEARNING_RATE);
    
    int b1_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
    biasUpdate<<<b1_blocks, block_size>>>(
        net->d_b1, net->d_batch_d_hidden, batch_size, HIDDEN_SIZE, LEARNING_RATE);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in backward pass: %s\n", cudaGetErrorString(err));
    }
}

// Improved training function
void trainWithCuda(CudaNetwork* cuda_net, double** images, double** labels, int numImages) {
    // Host memory allocations
    printf("\nEpochs: %d", EPOCHS);
    printf("\nBatch size: %d\n", BATCH_SIZE);
    float* h_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* h_batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float* h_loss = (float*)malloc(BATCH_SIZE * sizeof(float));
    int* h_correct = (int*)malloc(BATCH_SIZE * sizeof(int));

    // Device memory for metrics
    float* d_loss;
    int* d_correct;
    CUDA_CHECK(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct, BATCH_SIZE * sizeof(int)));

    // GPU Timing events
    cudaEvent_t total_start, total_end, epoch_start, epoch_end;
    CUDA_CHECK(cudaEventCreate(&total_start));
    CUDA_CHECK(cudaEventCreate(&total_end));
    CUDA_CHECK(cudaEventCreate(&epoch_start));
    CUDA_CHECK(cudaEventCreate(&epoch_end));
    
    CUDA_CHECK(cudaEventRecord(total_start));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        CUDA_CHECK(cudaEventRecord(epoch_start));

        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * BATCH_SIZE;
            int current_batch_size = (batch_start + BATCH_SIZE <= numImages) ?
                                     BATCH_SIZE : (numImages - batch_start);

            // Prepare batch
            for (int i = 0; i < current_batch_size; i++) {
                int img_idx = batch_start + i;
                for (int j = 0; j < INPUT_SIZE; j++) {
                    h_batch_input[i * INPUT_SIZE + j] = (float)images[img_idx][j];
                }
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    h_batch_target[i * OUTPUT_SIZE + j] = (float)labels[img_idx][j];
                }
            }

            // Copy to device
            CUDA_CHECK(cudaMemcpy(cuda_net->d_batch_input, h_batch_input,
                                  current_batch_size * INPUT_SIZE * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(cuda_net->d_batch_target, h_batch_target,
                                  current_batch_size * OUTPUT_SIZE * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // Forward and backward pass
            cudaForward(cuda_net, current_batch_size);
            cudaBackward(cuda_net, current_batch_size);

            // Loss and accuracy kernels
            int block_size = 256;
            int loss_blocks = current_batch_size;
            size_t shared_mem = (2 * OUTPUT_SIZE + block_size) * sizeof(float);

            crossEntropyLoss<<<loss_blocks, block_size, shared_mem>>>(
                cuda_net->d_batch_output, cuda_net->d_batch_target,
                d_loss, current_batch_size, OUTPUT_SIZE);

            int accuracy_blocks = (current_batch_size + block_size - 1) / block_size;
            accuracyKernel<<<accuracy_blocks, block_size>>>(
                cuda_net->d_batch_output, cuda_net->d_batch_target,
                d_correct, current_batch_size, OUTPUT_SIZE);

            // Copy back results
            CUDA_CHECK(cudaMemcpy(h_loss, d_loss,
                                  current_batch_size * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_correct, d_correct,
                                  current_batch_size * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            for (int i = 0; i < current_batch_size; i++) {
                epoch_loss += h_loss[i];
                epoch_correct += h_correct[i];
            }
        }

        CUDA_CHECK(cudaEventRecord(epoch_end));
        CUDA_CHECK(cudaEventSynchronize(epoch_end));

        float epoch_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&epoch_time_ms, epoch_start, epoch_end));

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages,
               (epoch_correct / (float)numImages) * 100.0f,
               epoch_time_ms / 1000.0f);
    }

    CUDA_CHECK(cudaEventRecord(total_end));
    CUDA_CHECK(cudaEventSynchronize(total_end));

    float total_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_time_ms, total_start, total_end));
    printf("Total training time: %.3fs\n", total_time_ms / 1000.0f);

    // Free
    free(h_batch_input);
    free(h_batch_target);
    free(h_loss);
    free(h_correct);
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_correct));
    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_end));
    CUDA_CHECK(cudaEventDestroy(epoch_start));
    CUDA_CHECK(cudaEventDestroy(epoch_end));
}

// Update host network from device network
void updateHostNetwork(double** W1, double** W2, double* b1, double* b2, CudaNetwork* cuda_net) {
    // Allocate temporary host memory
    float* h_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    float* h_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* h_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Copy from device to host
    CUDA_CHECK(cudaMemcpy(h_W1, cuda_net->d_W1, 
                         HIDDEN_SIZE * INPUT_SIZE * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_W2, cuda_net->d_W2, 
                         OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, cuda_net->d_b1, 
                         HIDDEN_SIZE * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, cuda_net->d_b2, 
                         OUTPUT_SIZE * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    // Convert from float to double and update host network
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        b1[i] = (double)h_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1[i][j] = (double)h_W1[i * INPUT_SIZE + j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2[i] = (double)h_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W2[i][j] = (double)h_W2[i * HIDDEN_SIZE + j];
        }
    }
    
    // Free temporary host memory
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);
}

// Evaluate using CUDA
void evaluateWithCuda(CudaNetwork* cuda_net, double** images, double** labels, int numImages, float* accuracy) {
    float* h_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* h_batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    int* h_correct = (int*)malloc(BATCH_SIZE * sizeof(int));
    int* d_correct;
    
    CUDA_CHECK(cudaMalloc(&d_correct, BATCH_SIZE * sizeof(int)));
    
    int total_correct = 0;
    
    // Process in mini-batches
    int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
    
    for (int batch = 0; batch < num_batches; batch++) {
        int batch_start = batch * BATCH_SIZE;
        int current_batch_size = (batch_start + BATCH_SIZE <= numImages) ? 
                                BATCH_SIZE : (numImages - batch_start);
        
        // Prepare batch data
        for (int i = 0; i < current_batch_size; i++) {
            int img_idx = batch_start + i;
            
            // Copy image data
            for (int j = 0; j < INPUT_SIZE; j++) {
                h_batch_input[i * INPUT_SIZE + j] = (float)images[img_idx][j];
            }
            
            // Copy label data
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                h_batch_target[i * OUTPUT_SIZE + j] = (float)labels[img_idx][j];
            }
        }
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(cuda_net->d_batch_input, h_batch_input, 
                             current_batch_size * INPUT_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cuda_net->d_batch_target, h_batch_target, 
                             current_batch_size * OUTPUT_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        // Forward pass only
        cudaForward(cuda_net, current_batch_size);
        
        // Calculate accuracy
        int threads_per_block = 256;
        int accuracy_blocks = (current_batch_size + threads_per_block - 1) / threads_per_block;
        accuracyKernel<<<accuracy_blocks, threads_per_block>>>(
            cuda_net->d_batch_output, cuda_net->d_batch_target, d_correct, current_batch_size, OUTPUT_SIZE);
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_correct, d_correct, 
                             current_batch_size * sizeof(int), 
                             cudaMemcpyDeviceToHost));
        
        // Count correct predictions
        for (int i = 0; i < current_batch_size; i++) {
            total_correct += h_correct[i];
        }
    }
    
    *accuracy = (float)total_correct / numImages * 100.0f;
    
    // Cleanup
    free(h_batch_input);
    free(h_batch_target);
    free(h_correct);
    CUDA_CHECK(cudaFree(d_correct));
}

// Free CUDA memory
void freeCudaNetwork(CudaNetwork* cuda_net) {
    CUDA_CHECK(cudaFree(cuda_net->d_W1));
    CUDA_CHECK(cudaFree(cuda_net->d_W2));
    CUDA_CHECK(cudaFree(cuda_net->d_b1));
    CUDA_CHECK(cudaFree(cuda_net->d_b2));
    CUDA_CHECK(cudaFree(cuda_net->d_batch_input));
    CUDA_CHECK(cudaFree(cuda_net->d_batch_hidden));
    CUDA_CHECK(cudaFree(cuda_net->d_batch_output));
    CUDA_CHECK(cudaFree(cuda_net->d_batch_target));
    CUDA_CHECK(cudaFree(cuda_net->d_batch_d_hidden));
    CUDA_CHECK(cudaFree(cuda_net->d_batch_d_output));
    free(cuda_net);
}