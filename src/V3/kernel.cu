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
#define EPOCHS 3
#define BATCH_SIZE 64
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

__global__ void matrixMultiply(float* A,float* B,float* C,float* bias,int Arows,int Acols,int Bcols,int batch_size){
    //declare the shared memory
    __shared__ float tileA[32][32]; //keeping the size to ensure better memory access.
    __shared__ float tileB[32][32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z;

    if (row >= Arows || col >= Bcols || batch >= batch_size) return;

    float* batchB = B + batch * Acols * Bcols;
    float* batchC = C + batch * Arows * Bcols;

    float sum = bias[row];

    for (int i=0;i<Acols;i+=32){
        //load the tiles into shared memory.
        tileA[threadIdx.y][threadIdx.x] = (row < Arows && i + threadIdx.x < Acols) ?
                                            A[row * Acols + i + threadIdx.x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (i + threadIdx.y < Acols && col < Bcols) ?
                                            batchB[(i + threadIdx.y) * Bcols + col] : 0.0f;

        __syncthreads();

        //partial dot product.
        for (int j=0;j< 32; j++){
            sum+= tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (col < Bcols) {
        batchC[row* Bcols + col] = sum; //store all the computed values.
    }

}


//transposed matrix multiplication.
__global__ void matrixMultiplyTranspose(float* A, float* B, float* C,int Arows, int Acols,int batch_size) {
        __shared__ float tileA[32][32]; // Tile for A^T (rows of A)
        __shared__ float tileB[32][32]; 

        int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index in C (A_cols dimension)
        int col = blockIdx.x * blockDim.x + threadIdx.x; 
        int batch = blockIdx.z;

        if (row >= Acols || col >= 1 || batch >= batch_size) return;

        float* batchB = B + batch * Arows; // B is A_rows x 1 per batch
        float* batchC = C + batch * Acols; // C is A_cols x 1 per batch

        float sum = 0.0f;

        for (int i = 0; i < Arows; i += 32) {
                // Load tiles into shared memory
                //  (row of A, transposed access)
                tileA[threadIdx.y][threadIdx.x] = (row < Acols && i + threadIdx.x < Arows) ?
                                                    A[(i + threadIdx.x) * Acols + row] : 0.0f;
                // (col of B, but col=0)
                tileB[threadIdx.y][threadIdx.x] = (i + threadIdx.y < Arows && col < 1) ?
                                                    batchB[i + threadIdx.y] : 0.0f;

                __syncthreads();

            
                for (int j = 0; j < 32; j++) {
                sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
                }

                __syncthreads();
        }

        //store all values into the output C value
        if (col < 1) {
        batchC[row] = sum;
        }
}

//vectorized kernels for the relu activation methods in the nn.c file.

__global__ void relu(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecIdx = idx*4; //each thread is to process 4 elements. based on vectorization.

    if(vecIdx < size){
        //use float 4 to load 4 elements.
        float4 val = *(float4*)&x[vecIdx];

        //apply the relu to each vlaue.
        val.x = fmax(0.0f,val.x);
        val.y = vecIdx + 1 < size ? fmaxf(0.0f, val.y) : 0.0f;
        val.z = vecIdx + 2 < size ? fmaxf(0.0f, val.z) : 0.0f;
        val.w = vecIdx + 3 < size ? fmaxf(0.0f, val.w) : 0.0f;

        *(float4*)&x[vecIdx]= val;
    }
}

//second method for relu derivation
__global__ void reluDerivativeKernel(float* x, float* dx, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vecIdx = idx * 4; // Each thread processes 4 elements

    if (vecIdx < size) {
        // Load 4 elements from x and dx
        float4 xVal = *(float4*)&x[vecIdx];
        float4 dxVal = *(float4*)&dx[vecIdx];

        // Apply derivative: dx *= (x > 0) ? 1 : 0
        dxVal.x *= (xVal.x > 0.0f) ? 1.0f : 0.0f;
        dxVal.y *= (vecIdx + 1 < size && xVal.y > 0.0f) ? 1.0f : 0.0f;
        dxVal.z *= (vecIdx + 2 < size && xVal.z > 0.0f) ? 1.0f : 0.0f;
        dxVal.w *= (vecIdx + 3 < size && xVal.w > 0.0f) ? 1.0f : 0.0f;

        // Store result
        *(float4*)&dx[vecIdx] = dxVal;
    }
}


//the softmax method
__global__ void softmax(float* x,int batch_size,int size) {
    int batchIndex = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if(batchIndex >= batch_size) {
        return;
    }


    //offset for the batch
    int offset = batchIndex*size;

    //shared memory for reducing global memory acceeses.
    extern __shared__ float shared[];
    float*  sData = shared; //space for values;
    float* sMax = &shared[threads];
    float* sSum = &shared[threads+1];

    //finding max val using reduction.
    float maxVal = -INFINITY;
    for (int i= tid; i< size;i+=threads) {
        maxVal = fmaxf(maxVal,x[offset+i]);
    }
    sData[tid] = maxVal;
    __syncthreads();

    //find the global max.
    for (int stride = threads/2;stride>0;stride >>= 1){
        if(tid < stride) {
            sData[tid] = fmaxf(sData[tid],sData[tid+stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        sMax[0] = sData[0];
    }
    __syncthreads();
    maxVal = sMax[0];
    
    //compute the partial sum.
    float partialSum = 0.0f;
    for (int i=tid; i < size; i+=threads) {
        float val = expf(x[offset+i] - maxVal);
        x[offset+i] = val;
        partialSum += val;
    }
    sData[tid] = partialSum;
    __syncthreads();

    //compute total sum.
    for(int stride = threads/22; stride > 0;stride>>=1) {
        if (tid < stride) {
            sData[tid] += sData[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        sSum[0] = sData[0];

    }
    __syncthreads();
    float sum = sSum[0];
    //normalize the values.
    for (int i = tid; i < size; i += threads) {
        x[offset + i] /= sum;
    }
}


//output gradient calculation.
__global__ void OutputGradient(float* output, float* target, float* gradient,int batch_size, int size) {
        int batchIndex = blockIdx.x;
        int tid = threadIdx.x;
        int threads = blockDim.x;

        if (batchIndex >= batch_size) {
            return;
        };

        // Shared memory for caching output and target
        extern __shared__ float shared[];
        float* outS = shared;
        float* targetS = &shared[size];

        int offset = batchIndex * size;

        // Load output and target into shared memory (coalesced)
        for (int i = tid; i < size; i += threads) {
                outS[i] = output[offset + i];
                targetS[i] = target[offset + i];
        }
        __syncthreads();

        // Compute gradient using strided loop
        for (int i = tid; i < size; i += threads) {
                gradient[offset + i] = outS[i] - targetS[i];
        }
}


//weight calculuations
__global__ void weightUpdate(float* weights, float* inputs, float* gradients,
                             int batch_size,int output_dim, int input_dim, float lr) {
            
        //get block and thread indexes separately.
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;               
        
        
        int outIndex = by*TILE_SIZE+ty;
        int inIndex = bx *TILE_SIZE+tx;


        extern __shared__ float shared[];
        float* sInputs = shared;
        float* sGradients = &shared[TILE_SIZE * input_dim];

        float weightUpdate = 0.0f;

        //process the batch
        for (int b = 0; b < batch_size; b += TILE_SIZE) {
            // Load inputs into shared memory (coalesced)
            if (ty < input_dim && tx < TILE_SIZE) {
                int batch_idx = b + tx;
                if (batch_idx < batch_size) {
                    sInputs[tx * input_dim + ty] = inputs[batch_idx * input_dim + ty];
                } else {
                    sInputs[tx * input_dim + ty] = 0.0f;
                }
            }
    
            // Load gradients into shared memory (coalesced)
            if (ty < TILE_SIZE && tx < output_dim) {
                int batch_idx = b + ty;
                if (batch_idx < batch_size && tx < output_dim) {
                    sGradients[ty * output_dim + tx] = gradients[batch_idx * output_dim + tx];
                } else {
                    sGradients[ty * output_dim + tx] = 0.0f;
                }
            }
            __syncthreads();
    
            // Compute partial update for this tile
            if (outIndex < output_dim && inIndex < input_dim) {
                for (int k = 0; k < min(TILE_SIZE,batch_size - b); k++) {
                    weightUpdate += sGradients[k * output_dim + outIndex] * sInputs[k * input_dim + inIndex];
                }
            }
            __syncthreads();
        }
    
        // Update weights (coalesced)
        if (outIndex < output_dim && inIndex < input_dim) {
            weights[outIndex * input_dim + inIndex] -= lr * (weightUpdate / BATCH_SIZE);
        }
}

//bias values updations
__global__ void biasUpdate(float* biases, float* gradients,
    int batch_size, int dim, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (idx >= dim) return;

    // Initialize bias update 
    if (tid == 0) {
        biases[idx] = biases[idx]; // No-op to ensure initialization
    }
    __syncthreads();

    // Accumulate partial sums using atomics
    float partial_sum = 0.0f;
    for (int b = tid; b < batch_size; b += threads) {
        partial_sum += gradients[b * dim + idx];
    }

    // Atomic update to biases
    atomicAdd(&biases[idx], -lr * (partial_sum / batch_size));
}


//entropy
__global__ void crossEntropyLoss(float* output, float* target, float* loss, int batch_size,int size) {
    extern __shared__ float s_data[];
    float* s_output = s_data;                     // Shared memory for output
    float* s_target = &s_data[size];       // Shared memory for target
    float* s_partial = &s_data[2 * size];  // Shared memory for partial sums

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int offset = batch_idx * size;

    // Coalesced load into shared memory
    if (tid < size) {
    s_output[tid] = output[offset + tid];
    s_target[tid] = target[offset + tid];
    }
    __syncthreads();

    // Compute partial loss in parallel
    float partial_loss = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        partial_loss -= s_target[i] * __logf(fmaxf(s_output[i], 1e-10f));
    }
    s_partial[tid] = partial_loss;
    __syncthreads();

    // Parallel reduction to sum partial losses
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_partial[tid] += s_partial[tid + stride];
    }
    __syncthreads();
    }

    // Write final loss for the batch
    if (tid == 0 && batch_idx < batch_size) {
        loss[batch_idx] = s_partial[0];
    }
}


//kernel for checking the accuracy of the obtained results.
// Accuracy kernel
__global__ void accuracyKernel(float* output, float* target, int* correct,
    int batch_size, int output_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size) {
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

        correct[batch_idx] = (pred_class == true_class) ? 1 : 0;
    }
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
void cudaForward(CudaNetwork* net, int batch_size) {
    // Define thread block dimensions
    dim3 blockDim(32, 16);
    
    // First layer: input -> hidden
    dim3 gridDim(
        (1 + blockDim.x - 1) / blockDim.x, // Usually 1 for single output column
        (HIDDEN_SIZE + blockDim.y - 1) / blockDim.y,
        batch_size
    );
    matrixMultiply<<<gridDim, blockDim>>>(
        net->d_W1, net->d_batch_input, net->d_batch_hidden, net->d_b1,
        HIDDEN_SIZE, INPUT_SIZE, 1, batch_size);
    
    // Apply ReLU activation
    int block_size = 512;
    int grid_size = (batch_size * HIDDEN_SIZE + block_size - 1) / block_size;
    relu<<<grid_size, block_size>>>(net->d_batch_hidden, batch_size * HIDDEN_SIZE);
    
    // Second layer: hidden -> output
    dim3 gridDim(
        (1 + blockDim.x - 1) / blockDim.x,
        (OUTPUT_SIZE + blockDim.y - 1) / blockDim.y,
        batch_size
    );
    matrixMultiply<<<gridDim, blockDim>>>(
        net->d_W2, net->d_batch_hidden, net->d_batch_output, net->d_b2,
        OUTPUT_SIZE, HIDDEN_SIZE, 1, batch_size);
    
    // Apply softmax
    block_size = 256;
    grid_size = (batch_size + block_size - 1) / block_size;
    size_t shared_mem = (block_size + 2) * sizeof(float); // For sData, sMax, sSum
    softmax<<<grid_size, block_size,shared_mem>>>(net->d_batch_output, batch_size, OUTPUT_SIZE);
}

// Backward pass
void cudaBackward(CudaNetwork* net, int batch_size) {
    // Calculate output gradients
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    size_t shared_mem = 2 * OUTPUT_SIZE * sizeof(float); // For outS, targetS
    OutputGradient<<<grid_size,block_size,shared_mem>>>(
        net->d_batch_output, net->d_batch_target, net->d_batch_d_output, batch_size, OUTPUT_SIZE);
    
    // Calculate hidden layer gradients
    dim3 blockDim(16, 16);
    
    // Using transpose operation for backpropagation: d_hidden = W2^T * d_output
    dim3 gridDim1((1 + 15) / 16, (HIDDEN_SIZE + 15) / 16, batch_size);
    matrixMultiplyTranspose<<<gridDim1, blockDim>>>(
        net->d_W2, net->d_batch_d_output, net->d_batch_d_hidden,
        OUTPUT_SIZE, HIDDEN_SIZE, batch_size);
    
    // Apply ReLU derivative
    int block_size = 256;
    int grid_size = (batch_size * HIDDEN_SIZE + block_size - 1) / block_size;
    reluDerivativeKernel<<<grid_size, block_size>>>(
        net->d_batch_hidden, net->d_batch_d_hidden, batch_size * HIDDEN_SIZE);
    
    // Update weights and biases
    // For W2 update
    dim3 blockW2(32, 16);
    dim3 gridW2(
        (HIDDEN_SIZE + blockW2.x - 1) / blockW2.x,
        (OUTPUT_SIZE + blockW2.y - 1) / blockW2.y
    );
    size_t shared_mem = (TILE_SIZE * INPUT_SIZE + TILE_SIZE * OUTPUT_SIZE) * sizeof(float);
    weightUpdate<<<gridW2, blockW2,shared_mem>>>(
                                        net->d_W2, net->d_batch_hidden, net->d_batch_d_output,
                                        batch_size, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    
    // For W1 update
    dim3 blockW1(32, 16);
    dim3 gridW1(
        (HIDDEN_SIZE + blockW2.x - 1) / blockW2.x,
        (OUTPUT_SIZE + blockW2.y - 1) / blockW2.y
    );
    weightUpdate<<<gridW1, blockW1,shared_mem>>>(
        net->d_W1, net->d_batch_input, net->d_batch_d_hidden,
        batch_size, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    
    // Update biases
    int block_size = 512;
    int b2_blocks = (OUTPUT_SIZE + block_size - 1) / block_size;
    biasUpdate<<<b2_blocks, block_size>>>(
        net->d_b2, net->d_batch_d_output, batch_size, OUTPUT_SIZE, LEARNING_RATE);
    
    int b1_blocks = (HIDDEN_SIZE + block_size - 1) / block_size;
    biasUpdate<<<b1_blocks, block_size>>>(
        net->d_b1, net->d_batch_d_hidden, batch_size, HIDDEN_SIZE, LEARNING_RATE);
}

// Train network using CUDA
void trainWithCuda(CudaNetwork* cuda_net, double** images, double** labels, int numImages) {
    // Host memory for batches
    float* h_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* h_batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Host and device memory for metrics
    float* h_loss = (float*)malloc(BATCH_SIZE * sizeof(float));
    int* h_correct = (int*)malloc(BATCH_SIZE * sizeof(int));
    
    float* d_loss;
    int* d_correct;
    CUDA_CHECK(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct, BATCH_SIZE * sizeof(int)));
    
    clock_t total_start = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        
        // Process in mini-batches
        int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * BATCH_SIZE;
            int current_batch_size = (batch_start + BATCH_SIZE <= numImages) ? 
                                     BATCH_SIZE : (numImages - batch_start);
            
            // Prepare batch data (convert double to float)
            for (int i = 0; i < current_batch_size; i++) {
                int img_idx = batch_start + i;
                
                // Copy and convert image data
                for (int j = 0; j < INPUT_SIZE; j++) {
                    h_batch_input[i * INPUT_SIZE + j] = (float)images[img_idx][j];
                }
                
                // Copy and convert label data
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    h_batch_target[i * OUTPUT_SIZE + j] = (float)labels[img_idx][j];
                }
            }
            
            // Copy batch to device
            CUDA_CHECK(cudaMemcpy(cuda_net->d_batch_input, h_batch_input, 
                                 current_batch_size * INPUT_SIZE * sizeof(float), 
                                 cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(cuda_net->d_batch_target, h_batch_target, 
                                 current_batch_size * OUTPUT_SIZE * sizeof(float), 
                                 cudaMemcpyHostToDevice));
            
            // Forward and backward pass
            cudaForward(cuda_net, current_batch_size);
            cudaBackward(cuda_net, current_batch_size);
            
            // Calculate loss
            int block_size = 512;
            int loss_blocks = (current_batch_size + block_size - 1) / block_size;
            size_t shared_mem = (2 * OUTPUT_SIZE + block_size) * sizeof(float); // s_output, s_target, s_partial
            crossEntropyLoss<<<loss_blocks, block_size,shared_mem>>>(
                cuda_net->d_batch_output, cuda_net->d_batch_target, d_loss, current_batch_size, OUTPUT_SIZE);
            
            // Calculate accuracy
            int accuracy_blocks = (current_batch_size + block_size - 1) / block_size;
            accuracyKernel<<<accuracy_blocks, block_size>>>(
                cuda_net->d_batch_output, cuda_net->d_batch_target, d_correct, current_batch_size, OUTPUT_SIZE);
            
            // Copy results back to host
            CUDA_CHECK(cudaMemcpy(h_loss, d_loss, 
                                 current_batch_size * sizeof(float), 
                                 cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_correct, d_correct, 
                                 current_batch_size * sizeof(int), 
                                 cudaMemcpyDeviceToHost));
            
            // Accumulate metrics
            for (int i = 0; i < current_batch_size; i++) {
                epoch_loss += h_loss[i];
                epoch_correct += h_correct[i];
            }
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages, 
               (epoch_correct / (float)numImages) * 100, 
               (float)(clock() - epoch_start) / CLOCKS_PER_SEC);
    }
    
    printf("Total training time: %.3fs\n", (float)(clock() - total_start) / CLOCKS_PER_SEC);
    
    // Clean up temporary memory
    free(h_batch_input);
    free(h_batch_target);
    free(h_loss);
    free(h_correct);
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_correct));
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