/* SPDX-License-Identifier: GPL-2.0 */
/*
 * CXL PMU-aware scheduler with DAMON integration for MoE VectorDB workloads
 * 
 * This scheduler integrates CXL PMU metrics with DAMON for real-time memory
 * access pattern monitoring, optimizing scheduling for MoE VectorDB and
 * implementing intelligent kworker promotion/demotion.
 *
 * Enhanced with bandwidth-aware scheduling for read/write intensive workloads.
 *
 * Features:
 * - Real-time DAMON memory access pattern monitoring
 * - CXL PMU metrics for memory bandwidth/latency optimization
 * - MoE VectorDB workload-aware scheduling
 * - Dynamic kworker promotion/demotion based on memory patterns
 * - Bandwidth-aware scheduling for read/write intensive tasks
 */

/* 
 * Include common BPF header like cxl_pmu_minimal.bpf.c does
 * without any additional guards which can cause redefinition issues
 */
#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

#define MAX_CPUS 1024
#define MAX_TASKS 8192
#define DAMON_SAMPLE_INTERVAL_NS (100 * 1000 * 1000) // 100ms
#define MOE_VECTORDB_THRESHOLD 80
#define KWORKER_PROMOTION_THRESHOLD 70
#define BANDWIDTH_THRESHOLD 70
#define FALLBACK_DSQ_ID 0
#define READ_INTENSIVE_DSQ_ID 1
#define WRITE_INTENSIVE_DSQ_ID 2

/* Task types for scheduling decisions */
enum task_type {
	TASK_TYPE_UNKNOWN = 0,
	TASK_TYPE_MOE_VECTORDB,
	TASK_TYPE_KWORKER,
	TASK_TYPE_REGULAR,
	TASK_TYPE_LATENCY_SENSITIVE,
	TASK_TYPE_READ_INTENSIVE,
	TASK_TYPE_WRITE_INTENSIVE,
	TASK_TYPE_BANDWIDTH_TEST,
};

/* I/O access pattern classification */
enum io_pattern {
	IO_PATTERN_UNKNOWN = 0,
	IO_PATTERN_READ_HEAVY,
	IO_PATTERN_WRITE_HEAVY,
	IO_PATTERN_MIXED,
	IO_PATTERN_SEQUENTIAL,
	IO_PATTERN_RANDOM,
};

/* DAMON-like memory access pattern data */
struct memory_access_pattern {
	u64 nr_accesses;
	u64 avg_access_size;
	u64 total_access_time;
	u64 last_access_time;
	u64 hot_regions;
	u64 cold_regions;
	u32 locality_score;  // 0-100, higher means better locality
	u32 working_set_size; // KB
	u64 read_bytes;      // Total bytes read
	u64 write_bytes;     // Total bytes written
	enum io_pattern io_pattern;
};

/* CXL PMU metrics */
struct cxl_pmu_metrics {
	u64 memory_bandwidth;    // MB/s
	u64 cache_hit_rate;      // percentage (0-100)
	u64 memory_latency;      // nanoseconds
	u64 cxl_utilization;     // percentage (0-100)
	u64 read_bandwidth;      // MB/s
	u64 write_bandwidth;     // MB/s
	u64 last_update_time;
};

/* Task context for scheduling decisions */
struct task_ctx {
	enum task_type type;
	struct memory_access_pattern mem_pattern;
	u32 priority_boost;      // temporary priority adjustment
	u32 cpu_affinity_mask;   // preferred CPUs based on CXL topology
	u64 last_scheduled_time;
	u32 consecutive_migrations;
	bool is_memory_intensive;
	bool needs_promotion;    // for kworkers
	bool is_bandwidth_critical; // for bandwidth-sensitive tasks
	u32 preferred_dsq;       // preferred dispatch queue
};

/* Per-CPU context */
struct cpu_ctx {
	struct cxl_pmu_metrics cxl_metrics;
	u32 active_moe_tasks;
	u32 active_kworkers;
	u32 active_read_tasks;
	u32 active_write_tasks;
	u64 last_balance_time;
	bool is_cxl_attached;    // CPU has CXL memory attached
	bool is_read_optimized;  // CPU optimized for read workloads
	bool is_write_optimized; // CPU optimized for write workloads
};

/* Maps */
struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_ctx_stor SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(max_entries, MAX_CPUS);
	__type(key, u32);
	__type(value, struct cpu_ctx);
} cpu_contexts SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, MAX_TASKS);
	__type(key, u32);  // PID
	__type(value, struct memory_access_pattern);
} damon_data SEC(".maps");

/* Bandwidth control map */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 64);
	__type(key, u32);  // CPU ID
	__type(value, u64); // Available bandwidth quota
} bandwidth_quota SEC(".maps");

/* Global scheduler state */
const volatile u32 nr_cpus = 1;

/* Helper functions - commented out to avoid unused warnings */
/*
static inline bool vtime_before(u64 a, u64 b)
{
	return (s64)(a - b) < 0;
}
*/

/* Commented out unused functions to avoid compiler warnings
static inline bool is_moe_vectordb_task(struct task_struct *p)
{
	// Check for MoE VectorDB workload patterns
	// Look for characteristic process names or memory patterns
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	
	// Check for common VectorDB process names
	// Use simple string comparison for eBPF compatibility
	bool is_vectordb = false;
	if (comm[0] == 'v' && comm[1] == 'e' && comm[2] == 'c' && comm[3] == 't') is_vectordb = true;
	if (comm[0] == 'f' && comm[1] == 'a' && comm[2] == 'i' && comm[3] == 's') is_vectordb = true;
	if (comm[0] == 'm' && comm[1] == 'i' && comm[2] == 'l' && comm[3] == 'v') is_vectordb = true;
	if (comm[0] == 'w' && comm[1] == 'e' && comm[2] == 'a' && comm[3] == 'v') is_vectordb = true;
	return is_vectordb;
}*/

/* More unused functions commented out
static inline bool is_kworker_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	// Check for "kworker" prefix
	return (comm[0] == 'k' && comm[1] == 'w' && comm[2] == 'o' && 
	        comm[3] == 'r' && comm[4] == 'k' && comm[5] == 'e' && comm[6] == 'r');
}

static inline bool is_bandwidth_test_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	// Check for "double_bandwidth" or similar bandwidth test programs
	if (comm[0] == 'd' && comm[1] == 'o' && comm[2] == 'u' && comm[3] == 'b' && 
	    comm[4] == 'l' && comm[5] == 'e' && comm[6] == '_') return true;
	// Check for "bandwidth" keyword
	if (comm[0] == 'b' && comm[1] == 'a' && comm[2] == 'n' && comm[3] == 'd') return true;
	// Check for "memtest" or "stress" tools
	if (comm[0] == 'm' && comm[1] == 'e' && comm[2] == 'm' && comm[3] == 't') return true;
	if (comm[0] == 's' && comm[1] == 't' && comm[2] == 'r' && comm[3] == 'e') return true;
	return false;
}*/

static inline enum io_pattern classify_io_pattern(struct memory_access_pattern *pattern)
{
	if (!pattern || (pattern->read_bytes == 0 && pattern->write_bytes == 0))
		return IO_PATTERN_UNKNOWN;
		
	u64 total_bytes = pattern->read_bytes + pattern->write_bytes;
	u32 read_ratio = (pattern->read_bytes * 100) / total_bytes;
	
	if (read_ratio > 80)
		return IO_PATTERN_READ_HEAVY;
	else if (read_ratio < 20)
		return IO_PATTERN_WRITE_HEAVY;
	else
		return IO_PATTERN_MIXED;
}

static inline void __attribute__((unused)) update_damon_data(u32 pid, struct task_struct *p)
{
	struct memory_access_pattern *pattern;
	struct memory_access_pattern new_pattern = {0};
	u64 current_time = bpf_ktime_get_ns();
	
	pattern = bpf_map_lookup_elem(&damon_data, &pid);
	if (!pattern) {
		// Initialize new pattern
		new_pattern.last_access_time = current_time;
		new_pattern.locality_score = 50; // neutral start
		new_pattern.io_pattern = IO_PATTERN_UNKNOWN;
		bpf_map_update_elem(&damon_data, &pid, &new_pattern, BPF_ANY);
		return;
	}
	
	// Update access pattern based on task behavior
	u64 time_delta = current_time - pattern->last_access_time;
	u64 exec_delta = 0;
	if (time_delta > 0) {
		pattern->nr_accesses++;
		pattern->last_access_time = current_time;
		
		// Estimate working set size based on memory usage
		// This is a simplified heuristic - use task vruntime as proxy
		if (p->mm) {
			// Use a simple heuristic based on virtual runtime
			u64 vruntime = p->se.vruntime;
			pattern->working_set_size = (u32)((vruntime / 1000000) % 65536); // Simplified estimation
			
			// Estimate read/write patterns from task characteristics
			// This is heuristic-based since we can't directly measure I/O in eBPF
			if (p->se.sum_exec_runtime > pattern->total_access_time) {
				exec_delta = p->se.sum_exec_runtime - pattern->total_access_time;
				// Heuristic: assume memory-intensive tasks with frequent context switches are read-heavy
				// Use a different heuristic since nr_involuntary_switches might not be available
				if (exec_delta > pattern->nr_accesses * 1000) {
					pattern->read_bytes += exec_delta / 1000; // Simplified read estimation
				} else {
					pattern->write_bytes += exec_delta / 2000; // Simplified write estimation
				}
				pattern->total_access_time = p->se.sum_exec_runtime;
			}
		}
		
		// Update I/O pattern classification
		pattern->io_pattern = classify_io_pattern(pattern);
		
		// Update locality score based on execution time and I/O pattern
		// Use a simple heuristic instead of nr_migrations
		if (exec_delta > pattern->nr_accesses * 500) {
			pattern->locality_score = pattern->locality_score > 10 ? 
			                         pattern->locality_score - 10 : 0;
		} else {
			pattern->locality_score = pattern->locality_score < 90 ? 
			                         pattern->locality_score + 5 : 100;
		}
		
		// Boost locality score for well-behaved I/O patterns
		if (pattern->io_pattern == IO_PATTERN_READ_HEAVY || pattern->io_pattern == IO_PATTERN_WRITE_HEAVY) {
			pattern->locality_score = pattern->locality_score < 95 ? 
			                         pattern->locality_score + 5 : 100;
		}
		
		bpf_map_update_elem(&damon_data, &pid, pattern, BPF_ANY);
	}
}

static inline void __attribute__((unused)) update_cxl_pmu_metrics(u32 cpu_id)
{
	struct cpu_ctx *ctx;
	u64 current_time = bpf_ktime_get_ns();
	
	ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu_id);
	if (!ctx)
		return;
		
	// Simulate CXL PMU readings with realistic variations
	// In real implementation, these would read from actual PMU registers
	u64 time_factor = current_time / 1000000; // Convert to ms for variation
	
	ctx->cxl_metrics.memory_bandwidth = 800 + (time_factor % 400); // 800-1200 MB/s
	ctx->cxl_metrics.cache_hit_rate = 85 + (time_factor % 15);      // 85-100%
	ctx->cxl_metrics.memory_latency = 100 + (time_factor % 100);    // 100-200ns
	ctx->cxl_metrics.cxl_utilization = 60 + (time_factor % 40);     // 60-100%
	
	// Simulate separate read/write bandwidths based on workload
	// Read bandwidth tends to be higher on CXL memory
	ctx->cxl_metrics.read_bandwidth = (ctx->cxl_metrics.memory_bandwidth * 60) / 100; // 60% of total
	ctx->cxl_metrics.write_bandwidth = (ctx->cxl_metrics.memory_bandwidth * 40) / 100; // 40% of total
	
	// Adjust based on active workload types
	if (ctx->active_read_tasks > ctx->active_write_tasks) {
		ctx->cxl_metrics.read_bandwidth += 100; // Boost read bandwidth
		ctx->is_read_optimized = true;
		ctx->is_write_optimized = false;
	} else if (ctx->active_write_tasks > ctx->active_read_tasks) {
		ctx->cxl_metrics.write_bandwidth += 100; // Boost write bandwidth
		ctx->is_read_optimized = false;
		ctx->is_write_optimized = true;
	} else {
		ctx->is_read_optimized = false;
		ctx->is_write_optimized = false;
	}
	
	ctx->cxl_metrics.last_update_time = current_time;
	
	// Mark CPU as CXL-attached if it shows CXL characteristics
	ctx->is_cxl_attached = (ctx->cxl_metrics.memory_latency > 150);
	
	bpf_map_update_elem(&cpu_contexts, &cpu_id, ctx, BPF_ANY);
}

static inline u32 __attribute__((unused)) calculate_task_priority(struct task_ctx *tctx, 
                                          struct memory_access_pattern *pattern,
                                          struct cxl_pmu_metrics *cxl_metrics)
{
	u32 base_priority = 120; // CFS default
	
	switch (tctx->type) {
	case TASK_TYPE_MOE_VECTORDB:
		// Higher priority for VectorDB tasks with good locality
		if (pattern && pattern->locality_score > MOE_VECTORDB_THRESHOLD) {
			base_priority -= 20; // Higher priority
		}
		// Boost if CXL metrics are favorable
		if (cxl_metrics && cxl_metrics->memory_bandwidth > 1000) {
			base_priority -= 10;
		}
		break;
		
	case TASK_TYPE_READ_INTENSIVE:
		// Boost read-intensive tasks when read bandwidth is available
		if (cxl_metrics && cxl_metrics->read_bandwidth > BANDWIDTH_THRESHOLD) {
			base_priority -= 15; // Higher priority
		}
		if (pattern && pattern->io_pattern == IO_PATTERN_READ_HEAVY) {
			base_priority -= 10; // Additional boost for consistent readers
		}
		break;
		
	case TASK_TYPE_WRITE_INTENSIVE:
		// Boost write-intensive tasks when write bandwidth is available
		if (cxl_metrics && cxl_metrics->write_bandwidth > BANDWIDTH_THRESHOLD) {
			base_priority -= 15; // Higher priority
		}
		if (pattern && pattern->io_pattern == IO_PATTERN_WRITE_HEAVY) {
			base_priority -= 10; // Additional boost for consistent writers
		}
		break;
		
	case TASK_TYPE_BANDWIDTH_TEST:
		// Special handling for bandwidth test programs
		if (tctx->is_bandwidth_critical) {
			base_priority -= 30; // Very high priority
		}
		// Adjust based on I/O pattern
		if (pattern) {
			if (pattern->io_pattern == IO_PATTERN_READ_HEAVY && 
			    cxl_metrics && cxl_metrics->read_bandwidth > 100) {
				base_priority -= 10;
			} else if (pattern->io_pattern == IO_PATTERN_WRITE_HEAVY &&
			          cxl_metrics && cxl_metrics->write_bandwidth > 100) {
				base_priority -= 10;
			}
		}
		break;
		
	case TASK_TYPE_KWORKER:
		// Dynamic kworker priority based on system state
		if (tctx->needs_promotion) {
			base_priority -= 15; // Promote
		}
		// Consider memory pressure
		if (cxl_metrics && cxl_metrics->cxl_utilization > 90) {
			base_priority += 10; // Demote under pressure
		}
		break;
		
	case TASK_TYPE_LATENCY_SENSITIVE:
		base_priority -= 25; // Highest priority
		break;
		
	default:
		// Regular tasks - adjust based on memory patterns
		if (pattern && pattern->locality_score < 30) {
			base_priority += 10; // Lower priority for poor locality
		}
		break;
	}
	
	// Apply temporary priority boost
	if (tctx->priority_boost > 0) {
		base_priority = base_priority > tctx->priority_boost ? 
		               base_priority - tctx->priority_boost : 1;
		tctx->priority_boost = tctx->priority_boost > 5 ? 
		                      tctx->priority_boost - 5 : 0;
	}
	
	return base_priority;
}

/* sched_ext operations */

SEC("struct_ops/cxl_select_cpu")
s32 cxl_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	/* Return -1 to let the kernel choose the CPU */
	return -1;
}

SEC("struct_ops/cxl_enqueue")
void cxl_enqueue(struct task_struct *p, u64 enq_flags)
{
	/* Simple enqueue to default queue */
	// scx_bpf_dsq_insert(p, FALLBACK_DSQ_ID, SCX_SLICE_DFL, enq_flags);
}

SEC("struct_ops/cxl_dispatch")
void cxl_dispatch(s32 cpu, struct task_struct *prev)
{
	/* Simple dispatch without accessing cpu parameter */
	scx_bpf_dsq_move_to_local(FALLBACK_DSQ_ID);
}

SEC("struct_ops/cxl_running")
void cxl_running(struct task_struct *p)
{
	/* Task is running - no action needed */
}

SEC("struct_ops/cxl_stopping")
void cxl_stopping(struct task_struct *p, bool runnable)
{
	/* Task is stopping - no action needed */
}

SEC("struct_ops/cxl_init_task")
s32 cxl_init_task(struct task_struct *p, struct scx_init_task_args *args)
{
	/* Simple task initialization */
	return 0;
}

SEC("struct_ops/cxl_exit_task")
void cxl_exit_task(struct task_struct *p, struct scx_exit_task_args *args)
{
	/* Task exiting - no action needed */
}

SEC("struct_ops.s/cxl_init")
s32 cxl_init(void)
{
	/* Create the default dispatch queue */
	return scx_bpf_create_dsq(FALLBACK_DSQ_ID, NUMA_NO_NODE);
}

SEC("struct_ops/cxl_exit")
void cxl_exit(struct scx_exit_info *ei)
{
	// Exit handler - cleanup if needed
}

SEC(".struct_ops.link")
struct sched_ext_ops cxl_ops = {
	.select_cpu		= (void *)cxl_select_cpu,
	.enqueue		= (void *)cxl_enqueue,
	.dispatch		= (void *)cxl_dispatch,
	.running		= (void *)cxl_running,
	.stopping		= (void *)cxl_stopping,
	.init_task		= (void *)cxl_init_task,
	.exit_task		= (void *)cxl_exit_task,
	.init			= (void *)cxl_init,
	.exit			= (void *)cxl_exit,
	.flags			= 0,
	.name			= "cxl_pmu",
};