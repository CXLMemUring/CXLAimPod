/* SPDX-License-Identifier: GPL-2.0 */
/*
 * CXL bandwidth monitoring using tracepoints
 * This is a fallback implementation that doesn't require sched_ext
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

#define MAX_TASKS 1024

struct task_stats {
    u64 total_runtime;
    u64 total_switches;
    u32 pid;
    u32 last_cpu;
    char comm[16];
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_TASKS);
    __type(key, u32);  // PID
    __type(value, struct task_stats);
} task_monitor SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 64);  // Max CPUs
    __type(key, u32);
    __type(value, u64);       // CPU usage counter
} cpu_usage SEC(".maps");

SEC("tp/sched/sched_switch")
int trace_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    u32 prev_pid = ctx->prev_pid;
    u32 next_pid = ctx->next_pid;
    u32 cpu = bpf_get_smp_processor_id();
    
    // Update CPU usage
    u64 *cpu_count = bpf_map_lookup_elem(&cpu_usage, &cpu);
    if (cpu_count) {
        __sync_fetch_and_add(cpu_count, 1);
    } else {
        u64 initial = 1;
        bpf_map_update_elem(&cpu_usage, &cpu, &initial, BPF_ANY);
    }
    
    // Update task that's being scheduled out
    if (prev_pid > 0) {
        struct task_stats *stats = bpf_map_lookup_elem(&task_monitor, &prev_pid);
        if (stats) {
            stats->total_switches++;
            stats->last_cpu = cpu;
        } else {
            struct task_stats new_stats = {0};
            new_stats.pid = prev_pid;
            new_stats.total_switches = 1;
            new_stats.last_cpu = cpu;
            bpf_get_current_comm(new_stats.comm, sizeof(new_stats.comm));
            bpf_map_update_elem(&task_monitor, &prev_pid, &new_stats, BPF_ANY);
        }
    }
    
    // Update task that's being scheduled in
    if (next_pid > 0) {
        struct task_stats *stats = bpf_map_lookup_elem(&task_monitor, &next_pid);
        if (stats) {
            stats->total_runtime++;
            stats->last_cpu = cpu;
        } else {
            struct task_stats new_stats = {0};
            new_stats.pid = next_pid;
            new_stats.total_runtime = 1;
            new_stats.last_cpu = cpu;
            bpf_get_current_comm(new_stats.comm, sizeof(new_stats.comm));
            bpf_map_update_elem(&task_monitor, &next_pid, &new_stats, BPF_ANY);
        }
    }
    
    return 0;
}

SEC("tp/sched/sched_wakeup")
int trace_sched_wakeup(void *ctx)
{
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    if (pid > 0) {
        struct task_stats *stats = bpf_map_lookup_elem(&task_monitor, &pid);
        if (stats) {
            // Task is waking up - could use this for scheduling hints
            stats->total_runtime++;
        }
    }
    
    return 0;
}