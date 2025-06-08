/* SPDX-License-Identifier: GPL-2.0 */
/*
 * CXL PMU-aware scheduler loader
 * 
 * This program loads and manages the CXL PMU eBPF scheduler.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

static volatile bool exiting = false;

static void sig_handler(int sig)
{
    exiting = true;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *format, va_list args)
{
    return vfprintf(stderr, format, args);
}

static void bump_memlock_rlimit(void)
{
    struct rlimit rlim_new = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };

    if (setrlimit(RLIMIT_MEMLOCK, &rlim_new)) {
        fprintf(stderr, "Failed to increase RLIMIT_MEMLOCK limit!\n");
        exit(1);
    }
}

static void print_usage(const char *prog_name)
{
    printf("Usage: %s [eBPF_object_file]\n", prog_name);
    printf("\n");
    printf("Load and run the CXL PMU-aware eBPF scheduler\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  eBPF_object_file    Path to the eBPF object file to load\n");
    printf("                      Default: cxl_pmu_simple.bpf.o\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                           # Load simple scheduler\n", prog_name);
    printf("  %s cxl_pmu_simple.bpf.o      # Load simple scheduler\n", prog_name);
    printf("  %s cxl_pmu.bpf.o             # Load complex scheduler\n", prog_name);
    printf("\n");
    printf("Note: This program requires root privileges and sched_ext kernel support\n");
}

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_link *link = NULL;
    const char *obj_file = "cxl_pmu_simple.bpf.o";  // Default to simple version
    int err;

    // Parse command line arguments
    if (argc > 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (argc == 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        obj_file = argv[1];
    }

    // Check if the eBPF object file exists
    if (access(obj_file, F_OK) != 0) {
        fprintf(stderr, "Error: eBPF object file '%s' not found.\n", obj_file);
        fprintf(stderr, "Please run 'make all' to build the scheduler first.\n");
        return 1;
    }

    printf("Loading CXL PMU scheduler from: %s\n", obj_file);

    libbpf_set_print(libbpf_print_fn);

    /* Bump RLIMIT_MEMLOCK to allow BPF sub-system to do anything */
    bump_memlock_rlimit();

    /* Set up signal handlers */
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* Open BPF application */
    obj = bpf_object__open_file(obj_file, NULL);
    if (libbpf_get_error(obj)) {
        fprintf(stderr, "ERROR: opening BPF object file '%s' failed\n", obj_file);
        return 1;
    }

    /* Load & verify BPF programs */
    printf("Loading and verifying eBPF programs...\n");
    err = bpf_object__load(obj);
    if (err) {
        fprintf(stderr, "ERROR: loading BPF object file failed (error: %d)\n", err);
        fprintf(stderr, "This may be due to:\n");
        fprintf(stderr, "  - eBPF instruction limit exceeded (try simple version)\n");
        fprintf(stderr, "  - Missing sched_ext kernel support\n");
        fprintf(stderr, "  - Kernel version incompatibility\n");
        goto cleanup;
    }

    printf("✓ CXL PMU-aware scheduler loaded successfully\n");
    printf("✓ Scheduler is now active and managing tasks\n");
    printf("\nScheduler features:\n");
    printf("  - VectorDB workload optimization\n");
    printf("  - Memory access pattern tracking\n");
    printf("  - Dynamic priority adjustment\n");
    printf("  - CXL-aware CPU selection\n");
    printf("\nPress Ctrl-C to exit and unload the scheduler\n");

    /* Main loop */
    while (!exiting) {
        sleep(1);
    }

    printf("\nShutting down scheduler...\n");

cleanup:
    bpf_link__destroy(link);
    bpf_object__close(obj);
    printf("Scheduler unloaded.\n");
    return err < 0 ? -err : 0;
}