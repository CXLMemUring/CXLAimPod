#include <linux/sched.h>
#include <linux/sched_ext.h>
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <linux/types.h>

char LICENSE[] SEC("license") = "GPL";

// CXL PMU counters map
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u32);  // CPU ID
    __type(value, u64[4]);  // Array of CXL PMU counters
} cxl_pmu_counters SEC(".maps");

// Cycle counts map for PMU tracking
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u32);  // PID
    __type(value, u64);  // Cycle count
} cycle_counts SEC(".maps");

// Task weight map based on CXL metrics
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, u32);  // PID
    __type(value, u32);  // Weight
} task_weights SEC(".maps");

// sched_ext global state
struct sched_ext_global_state {
    u64 total_tasks;
    u64 last_update;
};

// Global state map
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, u32);
    __type(value, struct sched_ext_global_state);
} global_state SEC(".maps");

SEC("perf_event")
int on_cycle_event(struct bpf_perf_event_data *ctx)
{
    u64 cycles = 1;
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    // Read hardware cycle counter (simplified implementation)
    cycles = bpf_ktime_get_ns();

    // Look up the existing count
    u64 *p = bpf_map_lookup_elem(&cycle_counts, &pid);
    if (p) {
        *p += cycles;
    } else {
        // First time seeing this PID, initialize
        bpf_map_update_elem(&cycle_counts, &pid, &cycles, BPF_ANY);
    }

    return 0;
}

// Read CXL PMU metrics for a given CPU
static inline void read_cxl_pmu(u32 cpu_id)
{
    u64 pmu_values[4] = {0};
    u64 timestamp = bpf_ktime_get_ns();
    
    // CXL PMU counter reading implementation
    // In a real implementation, these would read from actual CXL PMU registers
    // For now, we simulate realistic values based on system state
    
    // Counter 0: CXL memory bandwidth usage (MB/s)
    // Higher values under memory pressure
    u64 *existing = bpf_map_lookup_elem(&cxl_pmu_counters, &cpu_id);
    if (existing) {
        // Use previous values as baseline and add variation
        pmu_values[0] = existing[0] + ((timestamp % 100) - 50);
        pmu_values[1] = existing[1] + ((timestamp % 20) - 10);
        pmu_values[2] = existing[2] + ((timestamp % 40) - 20);
        pmu_values[3] = existing[3] + ((timestamp % 30) - 15);
    } else {
        // Initial values
        pmu_values[0] = 500 + (timestamp % 500);  // Bandwidth: 500-1000 MB/s
        pmu_values[1] = 70 + (timestamp % 30);    // Hit rate: 70-100%
        pmu_values[2] = 50 + (timestamp % 150);   // Latency: 50-200 ns
        pmu_values[3] = 40 + (timestamp % 60);    // Utilization: 40-100%
    }
    
    // Clamp values to realistic ranges
    if (pmu_values[0] > 1000) pmu_values[0] = 1000;
    if (pmu_values[0] < 100) pmu_values[0] = 100;
    if (pmu_values[1] > 100) pmu_values[1] = 100;
    if (pmu_values[1] < 20) pmu_values[1] = 20;
    if (pmu_values[2] > 300) pmu_values[2] = 300;
    if (pmu_values[2] < 30) pmu_values[2] = 30;
    if (pmu_values[3] > 100) pmu_values[3] = 100;
    if (pmu_values[3] < 10) pmu_values[3] = 10;
    
    bpf_map_update_elem(&cxl_pmu_counters, &cpu_id, &pmu_values, BPF_ANY);
}

// Calculate task weight based on CXL metrics
static inline u32 calculate_task_weight(u32 pid, u32 cpu)
{
    u64 *pmu_values;
    u32 weight = 1000; // Default weight
    
    pmu_values = bpf_map_lookup_elem(&cxl_pmu_counters, &cpu);
    if (!pmu_values)
        return weight;
    
    // Algorithm to adjust weight based on CXL metrics
    // Lower weight = higher priority
    
    // If bandwidth usage is high, reduce priority
    if (pmu_values[0] > 800)
        weight += 200;
        
    // If cache hit rate is low, reduce priority
    if (pmu_values[1] < 30)
        weight += 200;
    
    // If latency is high, reduce priority
    if (pmu_values[2] > 150)
        weight += 200;
    
    // If utilization is high, reduce priority
    if (pmu_values[3] > 80)
        weight += 200;
    
    return weight;
}

// sched_ext enqueue hook
SEC("struct_ops/sched_ext_ops")
int BPF_PROG(sched_ext_enqueue, struct task_struct *task, u64 enq_flags)
{
    u32 pid = task->pid;
    u32 cpu = bpf_get_smp_processor_id();
    u32 weight;
    
    // Read latest CXL PMU data
    read_cxl_pmu(cpu);
    
    // Calculate task weight based on CXL metrics
    weight = calculate_task_weight(pid, cpu);
    
    // Store task weight
    bpf_map_update_elem(&task_weights, &pid, &weight, BPF_ANY);
    
    // Enqueue to appropriate DSQ based on weight
    scx_bpf_dispatch(task, SCX_DSQ_GLOBAL, weight, 0);
    
    return 0;
}

// sched_ext dispatch hook
SEC("struct_ops/sched_ext_ops")
int BPF_PROG(sched_ext_dispatch, s32 cpu, struct task_struct *prev)
{
    // Update CXL PMU data for current CPU
    read_cxl_pmu(cpu);
    
    // Let the scheduler pick the next task
    scx_bpf_consume(SCX_DSQ_GLOBAL);
    
    return 0;
}

// sched_ext tick hook - periodically update CXL PMU data
SEC("struct_ops/sched_ext_ops")
int BPF_PROG(sched_ext_tick, struct task_struct *task)
{
    u32 cpu = bpf_get_smp_processor_id();
    u32 pid = task->pid;
    u32 weight;
    
    // Read latest CXL PMU data
    read_cxl_pmu(cpu);
    
    // Recalculate task weight
    weight = calculate_task_weight(pid, cpu);
    
    // Update task weight
    bpf_map_update_elem(&task_weights, &pid, &weight, BPF_ANY);
    
    return 0;
}

// Task exit hook
SEC("struct_ops/sched_ext_ops")
void BPF_PROG(sched_ext_exit_task, struct task_struct *task)
{
    u32 pid = task->pid;
    
    // Remove from our weight tracking
    bpf_map_delete_elem(&task_weights, &pid);
    bpf_map_delete_elem(&cycle_counts, &pid);
}

// Initialize sched_ext
SEC("struct_ops/sched_ext_ops")
s32 BPF_PROG(sched_ext_init)
{
    u32 key = 0;
    struct sched_ext_global_state state = {0};
    
    bpf_map_update_elem(&global_state, &key, &state, BPF_ANY);
    
    return 0;
}

// Exit sched_ext
SEC("struct_ops/sched_ext_ops")
void BPF_PROG(sched_ext_exit, struct scx_exit_info *ei)
{
}

// Define the sched_ext operations structure
SEC(".struct_ops.link")
struct sched_ext_ops cxl_sched_ops = {
    .enqueue        = (void *)sched_ext_enqueue,
    .dispatch       = (void *)sched_ext_dispatch,
    .tick           = (void *)sched_ext_tick,
    .exit_task      = (void *)sched_ext_exit_task,
    .init           = (void *)sched_ext_init,
    .exit           = (void *)sched_ext_exit,
    .name           = "cxl_sched",
};
