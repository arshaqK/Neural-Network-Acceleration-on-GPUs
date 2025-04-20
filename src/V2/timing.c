#include "neural_network.h"

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