#include "neural_network.h"

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
