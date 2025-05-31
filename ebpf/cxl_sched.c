/*
 * CXL-aware CPU scheduler using sched_ext
 * Userspace control program for the CXL PMU-based BPF scheduler
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <sys/resource.h>
#include <time.h>

static struct bpf_object *obj;
static struct bpf_link *link;
static int running = 1;

static void sigint_handler(int sig)
{
    running = 0;
}

static void print_cxl_stats(int fd_pmu, int fd_weights)
{
    __u32 key = 0, next_key;
    __u64 pmu_values[4];
    __u32 weight;
    
    printf("\n=== CXL PMU Statistics ===\n");
    printf("CPU ID | Bandwidth | Hit Rate | Latency | Utilization\n");
    printf("-------|-----------|----------|---------|------------\n");
    
    while (bpf_map_get_next_key(fd_pmu, &key, &next_key) == 0) {
        if (bpf_map_lookup_elem(fd_pmu, &next_key, pmu_values) == 0) {
            printf("  %3d  |    %3llu    |    %2llu    |   %3llu   |     %2llu\n",
                   next_key, pmu_values[0], pmu_values[1], pmu_values[2], pmu_values[3]);
        }
        key = next_key;
    }
    
    printf("\n=== Task Weights ===\n");
    printf("PID    | Weight\n");
    printf("-------|--------\n");
    
    key = 0;
    while (bpf_map_get_next_key(fd_weights, &key, &next_key) == 0) {
        if (bpf_map_lookup_elem(fd_weights, &next_key, &weight) == 0) {
            printf("%6d | %6d\n", next_key, weight);
        }
        key = next_key;
    }
}

int main(int argc, char **argv)
{
    struct bpf_program *prog;
    int fd_pmu_counters, fd_task_weights;
    int err;
    
    /* Set up libbpf errors and debug info callback */
    libbpf_set_print(NULL);
    
    /* Bump RLIMIT_MEMLOCK to allow BPF sub-system to do anything */
    struct rlimit rlim_new = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };
    
    if (setrlimit(RLIMIT_MEMLOCK, &rlim_new)) {
        fprintf(stderr, "Failed to increase RLIMIT_MEMLOCK limit!\n");
        return 1;
    }
    
    /* Open BPF application */
    obj = bpf_object__open_file("cxl_pmu.bpc.o", NULL);
    if (libbpf_get_error(obj)) {
        fprintf(stderr, "Failed to open BPF object file\n");
        return 1;
    }
    
    /* Load & verify BPF programs */
    err = bpf_object__load(obj);
    if (err) {
        fprintf(stderr, "Failed to load BPF object: %d\n", err);
        goto cleanup;
    }
    
    /* Find the struct_ops program */
    prog = bpf_object__find_program_by_name(obj, "cxl_sched_ops");
    if (!prog) {
        fprintf(stderr, "Failed to find cxl_sched_ops program\n");
        err = -ENOENT;
        goto cleanup;
    }
    
    /* Attach the scheduler */
    link = bpf_program__attach(prog);
    if (libbpf_get_error(link)) {
        fprintf(stderr, "Failed to attach BPF program\n");
        link = NULL;
        err = -1;
        goto cleanup;
    }
    
    printf("CXL-aware scheduler loaded successfully!\n");
    printf("Press Ctrl+C to stop...\n");
    
    /* Get map file descriptors for monitoring */
    fd_pmu_counters = bpf_object__find_map_fd_by_name(obj, "cxl_pmu_counters");
    fd_task_weights = bpf_object__find_map_fd_by_name(obj, "task_weights");
    
    if (fd_pmu_counters < 0 || fd_task_weights < 0) {
        fprintf(stderr, "Failed to find maps\n");
        err = -1;
        goto cleanup;
    }
    
    /* Set up signal handler */
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    
    /* Main monitoring loop */
    while (running) {
        sleep(5);  /* Print stats every 5 seconds */
        if (running) {
            print_cxl_stats(fd_pmu_counters, fd_task_weights);
        }
    }
    
    printf("\nShutting down CXL scheduler...\n");
    
cleanup:
    if (link)
        bpf_link__destroy(link);
    if (obj)
        bpf_object__close(obj);
    
    return err;
}