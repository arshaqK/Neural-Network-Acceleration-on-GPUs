// acc_network.h
// Header file for OpenACC neural network implementation

#ifndef ACC_NETWORK_H
#define ACC_NETWORK_H

// Constants (same as in kernel_openacc.c)
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 8
#define BATCH_SIZE 32
#define TILE_SIZE 32

// Forward declaration of AccNetwork structure
struct AccNetwork;

// Function prototypes for OpenACC neural network operations

// Create an OpenACC network from host network parameters
struct AccNetwork* createAccNetwork(double** W1, double** W2, double* b1, double* b2);

// Perform forward pass on the network
void accForward(struct AccNetwork* net, int batch_size);

// Perform backward pass and update weights
void accBackward(struct AccNetwork* net, int batch_size);

// Train the network with OpenACC
void trainWithAcc(struct AccNetwork* acc_net, double** images, double** labels, int numImages);

// Update host network parameters from device network
void updateHostNetwork(double** W1, double** W2, double* b1, double* b2, struct AccNetwork* acc_net);

// Evaluate the network accuracy with OpenACC
void evaluateWithAcc(struct AccNetwork* acc_net, double** images, double** labels, int numImages, float* accuracy);

// Free the OpenACC network resources
void freeAccNetwork(struct AccNetwork* acc_net);

#endif // ACC_NETWORK_H