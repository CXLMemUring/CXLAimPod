#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

int main(int argc, char **argv)
{
    struct bpf_object *obj;
    struct bpf_program *prog;
    int prog_fd, err;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <bpf_object.o>\n", argv[0]);
        return 1;
    }

    printf("=== Basic eBPF Test ===\n");
    printf("Loading %s...\n", argv[1]);

    obj = bpf_object__open(argv[1]);
    if (!obj) {
        fprintf(stderr, "Failed to open BPF object\n");
        return 1;
    }

    err = bpf_object__load(obj);
    if (err) {
        fprintf(stderr, "Failed to load BPF object: %d\n", err);
        bpf_object__close(obj);
        return 1;
    }

    printf("BPF object loaded successfully!\n");

    bpf_object__for_each_program(prog, obj) {
        printf("Program: %s\n", bpf_program__name(prog));
        prog_fd = bpf_program__fd(prog);
        if (prog_fd > 0) {
            printf("  FD: %d\n", prog_fd);
        }
    }

    bpf_object__close(obj);
    printf("Test completed successfully!\n");
    return 0;
}