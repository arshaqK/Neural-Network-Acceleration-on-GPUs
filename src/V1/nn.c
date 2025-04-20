#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

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

// Record time and return time elapsed in seconds
double time_function(int timer_id, void (*func)()) {
    clock_t start = clock();
    func();
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    timing_info[timer_id].total_time += elapsed;
    return elapsed;
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

// Activation functions
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val);  // Subtract max for numerical stability
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Function pointers for timing
typedef struct {
    NeuralNetwork* net;
    double** images;
    double** labels;
    int numImages;
    double* input;
    double* hidden;
    double* output;
    double* target;
    const char* filename;
    int num;
} FunctionParams;

FunctionParams params;

// Initialize neural network
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

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Hidden layer calculation
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    // Output layer calculation
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update weights (gradient descent)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
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
    printf("MNIST Neural Network Performance Profiling\n\n");
    
    // Initialize timers
    int timer_total = create_timer("Total Execution");
    int timer_initial_wait = create_timer("Initial Wait");
    int timer_load_train_images = create_timer("Load Training Images");
    int timer_load_train_labels = create_timer("Load Training Labels");
    int timer_load_test_images = create_timer("Load Test Images");
    int timer_load_test_labels = create_timer("Load Test Labels");
    int timer_network_creation = create_timer("Network Creation");
    int timer_training_total = create_timer("Training (Total)");
    int timer_forward_pass = create_timer("Forward Pass (Total)");
    int timer_backward_pass = create_timer("Backward Pass (Total)");
    int timer_evaluation = create_timer("Test Evaluation");
    int timer_cleanup = create_timer("Memory Cleanup");
    
    clock_t total_start = clock();
    
    // Measure initial wait
    clock_t wait_start = clock();
    for (volatile int i = 0; i < 100000000; i++);
    end_timing(wait_start, timer_initial_wait);
    
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
    
    // Create network
    start_time = clock();
    NeuralNetwork* net = createNetwork();
    end_timing(start_time, timer_network_creation);
    
    // Training
    start_time = clock();
    clock_t training_start = clock();
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double loss = 0.0;
        int correct = 0;
        clock_t epoch_start = clock();
        
        for (int i = 0; i < 60000; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            
            // Forward pass timing
            clock_t forward_start = clock();
            forward(net, train_images[i], hidden, output);
            timing_info[timer_forward_pass].total_time += (double)(clock() - forward_start) / CLOCKS_PER_SEC;
            
            // Backward pass timing
            clock_t backward_start = clock();
            backward(net, train_images[i], hidden, output, train_labels[i]);
            timing_info[timer_backward_pass].total_time += (double)(clock() - backward_start) / CLOCKS_PER_SEC;
            
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= train_labels[i][k] * log(output[k] + 1e-10);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (train_labels[i][j] > train_labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }
        
        double epoch_time = (double)(clock() - epoch_start) / CLOCKS_PER_SEC;
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / 60000, (correct / 60000.0) * 100, epoch_time);
    }
    
    end_timing(training_start, timer_training_total);
    
    // Test evaluation
    start_time = clock();
    int correct = 0;
    for (int i = 0; i < 10000; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, test_images[i], hidden, output);
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
    freeMatrix(train_images, 60000);
    freeMatrix(train_labels, 60000);
    freeMatrix(test_images, 10000);
    freeMatrix(test_labels, 10000);
    end_timing(start_time, timer_cleanup);
    
    // Calculate total time
    double total_time = (double)(clock() - total_start) / CLOCKS_PER_SEC;
    timing_info[timer_total].total_time = total_time;
    
    // Print timing results
    print_timing_results(total_time);
    
    return 0;
}