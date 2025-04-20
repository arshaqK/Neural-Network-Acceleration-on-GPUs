// main.c
#include <stdio.h>
#include <stdlib.h>
#include "accNet.h"
#include <time.h>
#include <string.h>

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    if (!mat) {
        fprintf(stderr, "Failed to allocate matrix\n");
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
        if (!mat[i]) {
            fprintf(stderr, "Failed to allocate matrix row %d\n", i);
            exit(1);
        }
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    if (!mat) return;
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening %s\n", filename);
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
                exit(1);
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
        fprintf(stderr, "Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(1);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Initialize neural network weights and biases
void initializeNetwork(double** W1, double** W2, double* b1, double* b2) {
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        b1[i] = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        b2[i] = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;
        }
    }
}

int main() {
    printf("MNIST Neural Network Performance Profiling (OpenACC)\n\n");

    // Load training data
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);
    
    
    // Initialize network
    double** W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    double** W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    double* b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    initializeNetwork(W1, W2, b1, b2);
    struct AccNetwork* acc_net = createAccNetwork(W1, W2, b1, b2);
    // Training
    trainWithAcc(acc_net, train_images, train_labels, 60000);
    // Test evaluation
    float accuracy;
    evaluateWithAcc(acc_net, test_images, test_labels, 10000, &accuracy);
    printf("Test Accuracy: %.2f%%\n", accuracy);
    
    // Memory cleanup
    freeAccNetwork(acc_net);
    freeMatrix(W1, HIDDEN_SIZE);
    freeMatrix(W2, OUTPUT_SIZE);
    free(b1);
    free(b2);
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    
    return 0;
}