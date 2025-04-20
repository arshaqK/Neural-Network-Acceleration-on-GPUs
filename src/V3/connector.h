#ifndef CONNECTOR_H
#define CONNECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

//cuda network memory structure initializer
typedef struct CudaNetwork CudaNetwork;



// Initialize CUDA and create network
CudaNetwork* createCudaNetwork(double** W1, double** W2, double* b1, double* b2);

// Train the network using CUDA
void trainWithCuda(CudaNetwork* cuda_net, double** images, double** labels, int numImages);

// Update host network after CUDA training
void updateHostNetwork(double** W1, double** W2, double* b1, double* b2, CudaNetwork* cuda_net);

// Free CUDA memory
void freeCudaNetwork(CudaNetwork* cuda_net);

// Evaluate network using CUDA
void evaluateWithCuda(CudaNetwork* cuda_net, double** images, double** labels, int numImages, float* accuracy);
#ifdef __cplusplus
}
#endif

#endif // CONNECTOR_H