/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Basic eBPF test - no sched_ext functions
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char _license[] SEC("license") = "GPL";

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, u64);
} test_map SEC(".maps");

SEC("kprobe/sys_openat")
int test_probe(struct pt_regs *ctx)
{
    u32 key = 0;
    u64 val = 1;
    bpf_map_update_elem(&test_map, &key, &val, BPF_ANY);
    return 0;
}