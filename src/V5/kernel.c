// kernel_openacc.c
// OpenACC implementation of the neural network training for v3 optimizations

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <openacc.h>
#include "accNet.h"

// Constants (same as CUDA)
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 8
#define BATCH_SIZE 32
#define TILE_SIZE 32

// Error checking macro
#define ACC_CHECK(call) \
do { \
    int err = call; \
    if (err != 0) { \
        fprintf(stderr, "OpenACC error in %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Device memory structure
struct AccNetwork {
    float *d_W1, *d_W2;
    float *d_b1, *d_b2;
    float *d_batch_input, *d_batch_hidden, *d_batch_output, *d_batch_target;
    float *d_batch_d_hidden, *d_batch_d_output;
};

// Matrix multiplication with OpenACC
void matrixMultiply(float* A, float* B, float* C, float* bias, int Arows, int Acols, int Bcols, int batch_size) {
    #pragma acc parallel loop gang collapse(2) present(A, B, C, bias)
    for (int batch = 0; batch < batch_size; batch++) {
        for (int row = 0; row < Arows; row++) {
            float sum = bias[row];
            #pragma acc loop vector reduction(+:sum)
            for (int col = 0; col < Acols; col++) {
                sum += A[row * Acols + col] * B[batch * Acols * Bcols + col * Bcols];
            }
            C[batch * Arows * Bcols + row * Bcols] = sum;
        }
    }
}

// Transposed matrix multiplication
void matrixMultiplyTranspose(float* A, float* B, float* C, int Arows, int Acols, int batch_size) {
    #pragma acc parallel loop gang collapse(2) present(A, B, C)
    for (int batch = 0; batch < batch_size; batch++) {
        for (int row = 0; row < Acols; row++) {
            float sum = 0.0f;
            #pragma acc loop vector reduction(+:sum)
            for (int i = 0; i < Arows; i++) {
                sum += A[i * Acols + row] * B[batch * Arows + i];
            }
            C[batch * Acols + row] = sum;
        }
    }
}

// ReLU activation
void relu(float* x, int size) {
    #pragma acc parallel loop present(x)
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

// ReLU derivative
void reluDerivative(float* x, float* dx, int size) {
    #pragma acc parallel loop present(x, dx)
    for (int i = 0; i < size; i++) {
        dx[i] *= (x[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Softmax with numerical stability
void softmax(float* x, int batch_size, int size) {
    #pragma acc parallel loop gang present(x)
    for (int batch = 0; batch < batch_size; batch++) {
        float* batchX = x + batch * size;
        float maxVal = -INFINITY;
        
        // Find max for numerical stability
        #pragma acc loop reduction(max:maxVal)
        for (int i = 0; i < size; i++) {
            maxVal = fmaxf(maxVal, batchX[i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        #pragma acc loop reduction(+:sum)
        for (int i = 0; i < size; i++) {
            batchX[i] = expf(batchX[i] - maxVal);
            sum += batchX[i];
        }
        
        // Normalize
        #pragma acc loop
        for (int i = 0; i < size; i++) {
            batchX[i] = fmaxf(batchX[i] / sum, 1e-15f);
        }
    }
}

// Output gradient
void OutputGradient(float* output, float* target, float* gradient, int batch_size, int size) {
    #pragma acc parallel loop gang collapse(2) present(output, target, gradient)
    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < size; i++) {
            int idx = batch * size + i;
            gradient[idx] = output[idx] - target[idx];
        }
    }
}

// Weight update
void weightUpdate(float* weights, float* inputs, float* gradients, int batch_size, int output_dim, int input_dim, float lr) {
    #pragma acc parallel loop gang collapse(2) present(weights, inputs, gradients)
    for (int outRow = 0; outRow < output_dim; outRow++) {
        for (int inCol = 0; inCol < input_dim; inCol++) {
            int weightIdx = outRow * input_dim + inCol;
            float weightDelta = 0.0f;
            #pragma acc loop reduction(+:weightDelta)
            for (int b = 0; b < batch_size; b++) {
                float input_val = inputs[b * input_dim + inCol];
                float grad_val = gradients[b * output_dim + outRow];
                weightDelta += grad_val * input_val;
            }
            weights[weightIdx] -= lr * (weightDelta / batch_size);
        }
    }
}

// Bias update
void biasUpdate(float* biases, float* gradients, int batch_size, int dim, float lr) {
    #pragma acc parallel loop present(biases, gradients)
    for (int idx = 0; idx < dim; idx++) {
        float bias_delta = 0.0f;
        #pragma acc loop reduction(+:bias_delta)
        for (int b = 0; b < batch_size; b++) {
            bias_delta += gradients[b * dim + idx];
        }
        biases[idx] -= lr * (bias_delta / batch_size);
    }
}

// Cross-entropy loss
void crossEntropyLoss(float* output, float* target, float* loss, int batch_size, int size) {
    #pragma acc parallel loop gang present(output, target, loss)
    for (int batch = 0; batch < batch_size; batch++) {
        float partial_loss = 0.0f;
        int offset = batch * size;
        #pragma acc loop reduction(+:partial_loss)
        for (int i = 0; i < size; i++) {
            float out_val = fmaxf(output[offset + i], 1e-15f);
            partial_loss -= target[offset + i] * logf(out_val);
        }
        loss[batch] = partial_loss;
    }
}

// Accuracy calculation
void accuracyKernel(float* output, float* target, int* correct, int batch_size, int output_size) {
    #pragma acc parallel loop present(output, target, correct)
    for (int batch = 0; batch < batch_size; batch++) {
        int offset = batch * output_size;
        int pred_class = 0;
        int true_class = 0;
        
        // Find predicted class
        for (int i = 1; i < output_size; i++) {
            if (output[offset + i] > output[offset + pred_class]) {
                pred_class = i;
            }
        }
        
        // Find true class
        for (int i = 0; i < output_size; i++) {
            if (target[offset + i] > 0.5f) {
                true_class = i;
                break;
            }
        }
        
        correct[batch] = (pred_class == true_class) ? 1 : 0;
    }
}

// Create OpenACC network
struct AccNetwork* createAccNetwork(double** W1, double** W2, double* b1, double* b2) {
    struct AccNetwork* acc_net = (struct AccNetwork*)malloc(sizeof(struct AccNetwork));
    
    // Allocate host memory (OpenACC manages device allocation)
    acc_net->d_W1 = (float*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    acc_net->d_W2 = (float*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    acc_net->d_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    acc_net->d_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    acc_net->d_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    acc_net->d_batch_hidden = (float*)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    acc_net->d_batch_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    acc_net->d_batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    acc_net->d_batch_d_hidden = (float*)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    acc_net->d_batch_d_output = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Convert host double to float
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        acc_net->d_b1[i] = (float)b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            acc_net->d_W1[i * INPUT_SIZE + j] = (float)W1[i][j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        acc_net->d_b2[i] = (float)b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            acc_net->d_W2[i * HIDDEN_SIZE + j] = (float)W2[i][j];
        }
    }
    
    // Create data on device
    #pragma acc enter data create(acc_net->d_W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                                  acc_net->d_W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                                  acc_net->d_b1[0:HIDDEN_SIZE], \
                                  acc_net->d_b2[0:OUTPUT_SIZE], \
                                  acc_net->d_batch_input[0:BATCH_SIZE*INPUT_SIZE], \
                                  acc_net->d_batch_hidden[0:BATCH_SIZE*HIDDEN_SIZE], \
                                  acc_net->d_batch_output[0:BATCH_SIZE*OUTPUT_SIZE], \
                                  acc_net->d_batch_target[0:BATCH_SIZE*OUTPUT_SIZE], \
                                  acc_net->d_batch_d_hidden[0:BATCH_SIZE*HIDDEN_SIZE], \
                                  acc_net->d_batch_d_output[0:BATCH_SIZE*OUTPUT_SIZE])
    
    // Copy initial data to device
    #pragma acc update device(acc_net->d_W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                              acc_net->d_W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                              acc_net->d_b1[0:HIDDEN_SIZE], \
                              acc_net->d_b2[0:OUTPUT_SIZE])
    
    return acc_net;
}

// Forward pass
void accForward(struct AccNetwork* net, int batch_size) {
    // First layer: input -> hidden
    matrixMultiply(net->d_W1, net->d_batch_input, net->d_batch_hidden, net->d_b1,
                   HIDDEN_SIZE, INPUT_SIZE, 1, batch_size);
    
    // ReLU activation
    relu(net->d_batch_hidden, batch_size * HIDDEN_SIZE);
    
    // Second layer: hidden -> output
    matrixMultiply(net->d_W2, net->d_batch_hidden, net->d_batch_output, net->d_b2,
                   OUTPUT_SIZE, HIDDEN_SIZE, 1, batch_size);
    
    // Softmax
    softmax(net->d_batch_output, batch_size, OUTPUT_SIZE);
}

// Backward pass
void accBackward(struct AccNetwork* net, int batch_size) {
    // Output gradients
    OutputGradient(net->d_batch_output, net->d_batch_target, net->d_batch_d_output, 
                   batch_size, OUTPUT_SIZE);
    
    // Hidden layer gradients
    matrixMultiplyTranspose(net->d_W2, net->d_batch_d_output, net->d_batch_d_hidden,
                           OUTPUT_SIZE, HIDDEN_SIZE, batch_size);
    
    // ReLU derivative
    reluDerivative(net->d_batch_hidden, net->d_batch_d_hidden, batch_size * HIDDEN_SIZE);
    
    // Update weights
    weightUpdate(net->d_W2, net->d_batch_hidden, net->d_batch_d_output,
                 batch_size, OUTPUT_SIZE, HIDDEN_SIZE, LEARNING_RATE);
    weightUpdate(net->d_W1, net->d_batch_input, net->d_batch_d_hidden,
                 batch_size, HIDDEN_SIZE, INPUT_SIZE, LEARNING_RATE);
    
    // Update biases
    biasUpdate(net->d_b2, net->d_batch_d_output, batch_size, OUTPUT_SIZE, LEARNING_RATE);
    biasUpdate(net->d_b1, net->d_batch_d_hidden, batch_size, HIDDEN_SIZE, LEARNING_RATE);
}

// Training function
void trainWithAcc(struct AccNetwork* acc_net, double** images, double** labels, int numImages) {
    float* h_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* h_batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    float* h_loss = (float*)malloc(BATCH_SIZE * sizeof(float));
    int* h_correct = (int*)malloc(BATCH_SIZE * sizeof(int));
    float* d_loss = (float*)malloc(BATCH_SIZE * sizeof(float));
    int* d_correct = (int*)malloc(BATCH_SIZE * sizeof(int));
    
    #pragma acc enter data create(d_loss[0:BATCH_SIZE], d_correct[0:BATCH_SIZE])
    
    clock_t total_start = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        
        int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * BATCH_SIZE;
            int current_batch_size = (batch_start + BATCH_SIZE <= numImages) ? 
                                     BATCH_SIZE : (numImages - batch_start);
            
            // Prepare batch data
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
            #pragma acc update device(acc_net->d_batch_input[0:current_batch_size*INPUT_SIZE], \
                                      acc_net->d_batch_target[0:current_batch_size*OUTPUT_SIZE])
            memcpy(acc_net->d_batch_input, h_batch_input, current_batch_size * INPUT_SIZE * sizeof(float));
            memcpy(acc_net->d_batch_target, h_batch_target, current_batch_size * OUTPUT_SIZE * sizeof(float));
            
            // Forward and backward passes
            accForward(acc_net, current_batch_size);
            accBackward(acc_net, current_batch_size);
            
            // Calculate loss
            crossEntropyLoss(acc_net->d_batch_output, acc_net->d_batch_target, 
                             d_loss, current_batch_size, OUTPUT_SIZE);
            
            // Calculate accuracy
            accuracyKernel(acc_net->d_batch_output, acc_net->d_batch_target, 
                           d_correct, current_batch_size, OUTPUT_SIZE);
            
            // Copy results back
            #pragma acc update host(d_loss[0:current_batch_size], d_correct[0:current_batch_size])
            memcpy(h_loss, d_loss, current_batch_size * sizeof(float));
            memcpy(h_correct, d_correct, current_batch_size * sizeof(int));
            
            // Accumulate metrics
            for (int i = 0; i < current_batch_size; i++) {
                epoch_loss += h_loss[i];
                epoch_correct += h_correct[i];
            }
        }
        
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages, 
               (epoch_correct / (float)numImages) * 100.0f, 
               (float)(clock() - epoch_start) / CLOCKS_PER_SEC);
    }
    
    printf("Total training time: %.3fs\n", (float)(clock() - total_start) / CLOCKS_PER_SEC);
    
    // Cleanup
    #pragma acc exit data delete(d_loss, d_correct)
    free(h_batch_input);
    free(h_batch_target);
    free(h_loss);
    free(h_correct);
    free(d_loss);
    free(d_correct);
}

// Update host network
void updateHostNetwork(double** W1, double** W2, double* b1, double* b2, struct AccNetwork* acc_net) {
    // Copy from device to host
    #pragma acc update host(acc_net->d_W1[0:HIDDEN_SIZE*INPUT_SIZE], \
                            acc_net->d_W2[0:OUTPUT_SIZE*HIDDEN_SIZE], \
                            acc_net->d_b1[0:HIDDEN_SIZE], \
                            acc_net->d_b2[0:OUTPUT_SIZE])
    
    // Convert to double
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        b1[i] = (double)acc_net->d_b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1[i][j] = (double)acc_net->d_W1[i * INPUT_SIZE + j];
        }
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2[i] = (double)acc_net->d_b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W2[i][j] = (double)acc_net->d_W2[i * HIDDEN_SIZE + j];
        }
    }
}

// Evaluate using OpenACC
void evaluateWithAcc(struct AccNetwork* acc_net, double** images, double** labels, int numImages, float* accuracy) {
    float* h_batch_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    float* h_batch_target = (float*)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    int* h_correct = (int*)malloc(BATCH_SIZE * sizeof(int));
    int* d_correct = (float*)malloc(BATCH_SIZE * sizeof(int));
    
    #pragma acc enter data create(d_correct[0:BATCH_SIZE])
    
    int total_correct = 0;
    int num_batches = (numImages + BATCH_SIZE - 1) / BATCH_SIZE;
    
    for (int batch = 0; batch < num_batches; batch++) {
        int batch_start = batch * BATCH_SIZE;
        int current_batch_size = (batch_start + BATCH_SIZE <= numImages) ? 
                                BATCH_SIZE : (numImages - batch_start);
        
        // Prepare batch data
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
        #pragma acc update device(acc_net->d_batch_input[0:current_batch_size*INPUT_SIZE], \
                                  acc_net->d_batch_target[0:current_batch_size*OUTPUT_SIZE])
        memcpy(acc_net->d_batch_input, h_batch_input, current_batch_size * INPUT_SIZE * sizeof(float));
        memcpy(acc_net->d_batch_target, h_batch_target, current_batch_size * OUTPUT_SIZE * sizeof(float));
        
        // Forward pass
        accForward(acc_net, current_batch_size);
        
        // Calculate accuracy
        accuracyKernel(acc_net->d_batch_output, acc_net->d_batch_target, 
                       d_correct, current_batch_size, OUTPUT_SIZE);
        
        // Copy results back
        #pragma acc update host(d_correct[0:current_batch_size])
        memcpy(h_correct, d_correct, current_batch_size * sizeof(int));
        
        // Count correct predictions
        for (int i = 0; i < current_batch_size; i++) {
            total_correct += h_correct[i];
        }
    }
    
    *accuracy = (float)total_correct / numImages * 100.0f;
    
    // Cleanup
    #pragma acc exit data delete(d_correct)
    free(h_batch_input);
    free(h_batch_target);
    free(h_correct);
    free(d_correct);
}

// Free OpenACC network
void freeAccNetwork(struct AccNetwork* acc_net) {
    #pragma acc exit data delete(acc_net->d_W1, acc_net->d_W2, acc_net->d_b1, acc_net->d_b2, \
                                 acc_net->d_batch_input, acc_net->d_batch_hidden, \
                                 acc_net->d_batch_output, acc_net->d_batch_target, \
                                 acc_net->d_batch_d_hidden, acc_net->d_batch_d_output)
    free(acc_net->d_W1);
    free(acc_net->d_W2);
    free(acc_net->d_b1);
    free(acc_net->d_b2);
    free(acc_net->d_batch_input);
    free(acc_net->d_batch_hidden);
    free(acc_net->d_batch_output);
    free(acc_net->d_batch_target);
    free(acc_net->d_batch_d_hidden);
    free(acc_net->d_batch_d_output);
    free(acc_net);
}