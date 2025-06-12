/*
 * Simple BPF object verification tool
 * Checks if the BPF objects are properly structured without loading them
 */

#include <stdio.h>
#include <stdlib.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <bpf_object.o>\n", argv[0]);
        return 1;
    }
    
    char *filename = argv[1];
    struct bpf_object *obj;
    int err;
    
    /* Open the BPF object file */
    obj = bpf_object__open(filename);
    if (!obj) {
        printf("❌ Failed to open BPF object file: %s\n", filename);
        return 1;
    }
    
    printf("✅ Successfully opened BPF object: %s\n", filename);
    
    /* Check for struct_ops maps */
    struct bpf_map *map;
    int found_sched_ext = 0;
    
    bpf_object__for_each_map(map, obj) {
        const char *name = bpf_map__name(map);
        enum bpf_map_type type = bpf_map__type(map);
        
        printf("  Map: %s (type: %d)\n", name, type);
        
        if (strstr(name, "ops") && type == BPF_MAP_TYPE_STRUCT_OPS) {
            printf("    ✅ Found sched_ext struct_ops map: %s\n", name);
            found_sched_ext = 1;
        }
    }
    
    /* Check for struct_ops programs */
    struct bpf_program *prog;
    int prog_count = 0;
    
    bpf_object__for_each_program(prog, obj) {
        const char *name = bpf_program__name(prog);
        enum bpf_prog_type type = bpf_program__type(prog);
        
        printf("  Program: %s (type: %d)\n", name, type);
        prog_count++;
    }
    
    printf("\n📊 Summary:\n");
    printf("  Total programs: %d\n", prog_count);
    printf("  Sched_ext struct_ops found: %s\n", found_sched_ext ? "Yes" : "No");
    
    if (found_sched_ext && prog_count > 0) {
        printf("✅ BPF object appears to be a valid sched_ext scheduler\n");
        printf("💡 Note: Actual loading requires root privileges and sched_ext kernel support\n");
    } else {
        printf("❌ BPF object missing required sched_ext components\n");
    }
    
    bpf_object__close(obj);
    return found_sched_ext && prog_count > 0 ? 0 : 1;
}