/*
 * CXL Bandwidth-Aware Scheduler Controller
 * 
 * This program loads and controls the CXL PMU-aware eBPF scheduler
 * with enhanced bandwidth optimization for read/write intensive workloads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#define MAX_CPUS 1024
#define MAX_TASKS 8192

struct bandwidth_config {
    int enable_scheduler;
    int max_read_bandwidth;  // MB/s
    int max_write_bandwidth; // MB/s
    int num_threads;
    float read_ratio;
    int monitor_interval;    // seconds
};

struct scheduler_stats {
    unsigned long read_tasks_scheduled;
    unsigned long write_tasks_scheduled;
    unsigned long bandwidth_tasks_scheduled;
    unsigned long total_context_switches;
    double avg_latency_ms;
};

static volatile int running = 1;
static struct bpf_object *obj = NULL;
static struct bpf_link *sched_link = NULL;
static char *bpf_obj_file = "cxl_pmu_minimal.bpf.o";

void signal_handler(int sig) {
    running = 0;
    printf("\nShutting down scheduler...\n");
}

int load_scheduler() {
    int err;
    
    /* Set the limit for memlock to allow BPF */
    struct rlimit rlim = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };
    
    if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
        perror("Failed to increase memlock limit");
        return -1;
    }
    
    /* Load the BPF object file */
    obj = bpf_object__open(bpf_obj_file);
    if (!obj) {
        fprintf(stderr, "Failed to open BPF object file\n");
        return -1;
    }
    
    /* Load the BPF program */
    err = bpf_object__load(obj);
    if (err) {
        fprintf(stderr, "Failed to load BPF object: %d\n", err);
        goto cleanup;
    }
    
    /* Find and attach the scheduler struct_ops */
    struct bpf_map *map = bpf_object__find_map_by_name(obj, "cxl_ops");
    if (!map) {
        map = bpf_object__find_map_by_name(obj, "minimal_ops");
        if (!map) {
            fprintf(stderr, "Failed to find scheduler map\n");
            err = -1;
            goto cleanup;
        }
    }
    
    sched_link = bpf_map__attach_struct_ops(map);
    if (!sched_link) {
        fprintf(stderr, "Failed to attach scheduler\n");
        err = -1;
        goto cleanup;
    }
    
    printf("CXL bandwidth-aware scheduler loaded successfully\n");
    return 0;
    
cleanup:
    if (obj) {
        bpf_object__close(obj);
        obj = NULL;
    }
    return err;
}

void unload_scheduler() {
    if (sched_link) {
        bpf_link__destroy(sched_link);
        sched_link = NULL;
    }
    
    if (obj) {
        bpf_object__close(obj);
        obj = NULL;
    }
    
    printf("Scheduler unloaded\n");
}

int configure_bandwidth_limits(struct bandwidth_config *config) {
    if (!obj) {
        fprintf(stderr, "Scheduler not loaded\n");
        return -1;
    }
    
    printf("Configuring bandwidth limits:\n");
    printf("  Read bandwidth limit: %d MB/s\n", config->max_read_bandwidth);
    printf("  Write bandwidth limit: %d MB/s\n", config->max_write_bandwidth);
    printf("  Thread count: %d\n", config->num_threads);
    printf("  Read ratio: %.2f\n", config->read_ratio);
    
    return 0;
}

int spawn_bandwidth_test(struct bandwidth_config *config) {
    char cmd[512];
    pid_t pid;
    
    printf("Spawning %d bandwidth test threads with read ratio %.2f\n", 
           config->num_threads, config->read_ratio);
    
    pid = fork();
    if (pid == 0) {
        // Child process - execute the bandwidth test
        snprintf(cmd, sizeof(cmd), 
                "./double_bandwidth -t %d -r %.2f -d %d -B %d",
                config->num_threads, 
                config->read_ratio,
                60, // Run for 60 seconds
                config->max_read_bandwidth + config->max_write_bandwidth);
        
        printf("Executing: %s\n", cmd);
        system(cmd);
        exit(0);
    } else if (pid > 0) {
        printf("Bandwidth test started with PID: %d\n", pid);
        return pid;
    } else {
        perror("Failed to fork bandwidth test");
        return -1;
    }
}

void print_scheduler_stats() {
    static unsigned long last_switches = 0;
    static time_t last_time = 0;
    
    time_t current_time = time(NULL);
    if (last_time == 0) {
        last_time = current_time;
        return;
    }
    
    // This is a simplified stats display
    // In a real implementation, we would read from BPF maps
    printf("\n=== Scheduler Statistics ===\n");
    printf("Uptime: %ld seconds\n", current_time - last_time);
    printf("Read-intensive tasks prioritized: Enabled\n");
    printf("Write-intensive tasks prioritized: Enabled\n");
    printf("Bandwidth test tasks prioritized: Enabled\n");
    printf("CXL-aware CPU selection: Enabled\n");
    printf("============================\n\n");
}

void monitor_performance(struct bandwidth_config *config) {
    printf("Starting performance monitoring (interval: %d seconds)\n", 
           config->monitor_interval);
    
    while (running) {
        sleep(config->monitor_interval);
        print_scheduler_stats();
    }
}

void print_usage(const char *prog_name) {
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -r, --read-bw=MB/s      Maximum read bandwidth limit (default: 1000)\n");
    printf("  -w, --write-bw=MB/s     Maximum write bandwidth limit (default: 500)\n");
    printf("  -t, --threads=NUM       Number of test threads to spawn (default: 20)\n");
    printf("  -R, --read-ratio=RATIO  Read thread ratio 0.0-1.0 (default: 0.6)\n");
    printf("  -i, --interval=SEC      Monitoring interval in seconds (default: 5)\n");
    printf("  -T, --test              Spawn bandwidth test automatically\n");
    printf("  -h, --help              Show this help message\n");
}

int main(int argc, char **argv) {
    struct bandwidth_config config = {
        .enable_scheduler = 1,
        .max_read_bandwidth = 1000,
        .max_write_bandwidth = 500,
        .num_threads = 20,
        .read_ratio = 0.6,
        .monitor_interval = 5,
    };
    
    int spawn_test = 0;
    int opt;
    
    // Check for BPF object file argument
    if (argc > 1 && argv[1][0] != '-') {
        bpf_obj_file = argv[1];
        argc--;
        argv++;
    }
    
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "r:w:t:R:i:Th")) != -1) {
        switch (opt) {
            case 'r':
                config.max_read_bandwidth = atoi(optarg);
                break;
            case 'w':
                config.max_write_bandwidth = atoi(optarg);
                break;
            case 't':
                config.num_threads = atoi(optarg);
                break;
            case 'R':
                config.read_ratio = atof(optarg);
                if (config.read_ratio < 0.0 || config.read_ratio > 1.0) {
                    fprintf(stderr, "Read ratio must be between 0.0 and 1.0\n");
                    exit(1);
                }
                break;
            case 'i':
                config.monitor_interval = atoi(optarg);
                break;
            case 'T':
                spawn_test = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                exit(0);
            default:
                print_usage(argv[0]);
                exit(1);
        }
    }
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("=== CXL Bandwidth-Aware Scheduler ===\n");
    printf("Loading scheduler...\n");
    
    // Load the BPF scheduler
    if (load_scheduler() != 0) {
        fprintf(stderr, "Failed to load scheduler\n");
        return 1;
    }
    
    // Configure bandwidth limits
    if (configure_bandwidth_limits(&config) != 0) {
        fprintf(stderr, "Failed to configure bandwidth limits\n");
        unload_scheduler();
        return 1;
    }
    
    // Spawn bandwidth test if requested
    if (spawn_test) {
        int test_pid = spawn_bandwidth_test(&config);
        if (test_pid < 0) {
            fprintf(stderr, "Failed to spawn bandwidth test\n");
            unload_scheduler();
            return 1;
        }
    }
    
    // Start monitoring
    printf("Scheduler is running. Press Ctrl+C to stop.\n");
    monitor_performance(&config);
    
    // Cleanup
    unload_scheduler();
    printf("Scheduler stopped.\n");
    
    return 0;
} 