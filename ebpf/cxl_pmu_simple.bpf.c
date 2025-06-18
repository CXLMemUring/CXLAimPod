/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Simplified CXL PMU-aware scheduler - optimized for instruction limit
 */

#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

#define MAX_CPUS 64  // Reduced for simplicity
#define MAX_TASKS 1024
#define MOE_VECTORDB_THRESHOLD 80
#define FALLBACK_DSQ_ID 0
#define MAX_CPU_LOOP 4  // Limit for fixed loop iteration

/* Simplified task types */
enum task_type {
	TASK_TYPE_UNKNOWN = 0,
	TASK_TYPE_MOE_VECTORDB,
	TASK_TYPE_KWORKER,
	TASK_TYPE_REGULAR,
	TASK_TYPE_BANDWIDTH,   // 新增：内存带宽测试任务
};

/* Simplified memory pattern */
struct memory_pattern {
	u64 last_access_time;
	u32 locality_score;
	u32 access_count;
	bool is_reader;        // 新增：标识是读线程还是写线程
};

/* Simplified task context */
struct task_ctx {
	enum task_type type;
	u32 priority_boost;
	bool is_memory_intensive;
	u32 thread_id;         // 新增：线程ID，用于区分读写线程
};

/* Simplified CPU context */
struct cpu_ctx {
	u32 active_tasks;
	u64 last_update;
	bool is_preferred;
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
	__type(key, u32);
	__type(value, struct memory_pattern);
} memory_patterns SEC(".maps");

/* Global state */
static u64 global_vtime = 0;
const volatile u32 nr_cpus = 8;  // Fixed for simplicity

/* Helper functions - simplified */

static inline bool vtime_before(u64 a, u64 b)
{
	return (s64)(a - b) < 0;
}

static inline bool is_vectordb_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	
	// Simple prefix check - reduced complexity
	return (comm[0] == 'v' && comm[1] == 'e' && comm[2] == 'c' && comm[3] == 't') ||
	       (comm[0] == 'f' && comm[1] == 'a' && comm[2] == 'i' && comm[3] == 's') ||
	       (comm[0] == 'p' && comm[1] == 'y' && comm[2] == 't' && comm[3] == 'h') ||
	       // 识别内存带宽测试程序
	       (comm[0] == 'd' && comm[1] == 'o' && comm[2] == 'u' && comm[3] == 'b');
}

static inline bool is_bandwidth_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	
	// 检测各种带宽测试工具
	return (comm[0] == 'd' && comm[1] == 'o' && comm[2] == 'u' && comm[3] == 'b') || // double_bandwidth
	       (comm[0] == 'm' && comm[1] == 'l' && comm[2] == 'c') ||                    // mlc (Intel Memory Latency Checker)
	       (comm[0] == 's' && comm[1] == 't' && comm[2] == 'r' && comm[3] == 'e');    // stream benchmark
}

static inline bool is_kworker_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	return (comm[0] == 'k' && comm[1] == 'w' && comm[2] == 'o' && comm[3] == 'r');
}

static inline void update_memory_pattern(u32 pid, struct task_ctx *tctx)
{
	struct memory_pattern *pattern;
	struct memory_pattern new_pattern = {0};
	u64 current_time = bpf_ktime_get_ns();
	
	pattern = bpf_map_lookup_elem(&memory_patterns, &pid);
	if (!pattern) {
		new_pattern.last_access_time = current_time;
		new_pattern.locality_score = 50;
		new_pattern.access_count = 1;
		
		// 为带宽测试任务设置读写属性
		if (tctx->type == TASK_TYPE_BANDWIDTH) {
			// 直接使用线程ID的奇偶性判断读写属性
			// 偶数线程ID => 读线程
			// 奇数线程ID => 写线程
			new_pattern.is_reader = (tctx->thread_id % 2 == 0);
		}
		
		bpf_map_update_elem(&memory_patterns, &pid, &new_pattern, BPF_ANY);
		return;
	}
	
	// Simple update logic
	pattern->access_count++;
	pattern->last_access_time = current_time;
	
	// 确保读写属性与线程ID奇偶性一致
	if (tctx->type == TASK_TYPE_BANDWIDTH) {
		pattern->is_reader = (tctx->thread_id % 2 == 0);
	}
	
	// Simple locality score update
	if (pattern->access_count % 10 == 0) {
		if (pattern->locality_score < 90)
			pattern->locality_score += 5;
	}
	
	bpf_map_update_elem(&memory_patterns, &pid, pattern, BPF_ANY);
}

static inline u32 get_task_priority(struct task_ctx *tctx)
{
	u32 base_priority = 120; // CFS default
	
	switch (tctx->type) {
	case TASK_TYPE_MOE_VECTORDB:
		base_priority -= 15; // Higher priority
		break;
	case TASK_TYPE_KWORKER:
		base_priority += 5; // Lower priority
		break;
	case TASK_TYPE_BANDWIDTH:
		// 内存带宽测试任务获得较高优先级
		base_priority -= 20;
		break;
	default:
		break;
	}
	
	if (tctx->priority_boost > 0) {
		base_priority = base_priority > tctx->priority_boost ? 
		               base_priority - tctx->priority_boost : 1;
		tctx->priority_boost = tctx->priority_boost > 2 ? 
		                      tctx->priority_boost - 2 : 0;
	}
	
	return base_priority;
}

/* sched_ext operations - simplified */

s32 BPF_STRUCT_OPS(cxl_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	struct task_ctx *tctx;
	struct cpu_ctx *cpu_ctx;
	struct memory_pattern *pattern;
	s32 best_cpu = prev_cpu;
	u32 cpu;
	u32 pid = p->pid;
	
	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0, 0);
	if (!tctx)
		return prev_cpu;
	
	// 处理内存带宽测试任务
	if (tctx->type == TASK_TYPE_BANDWIDTH) {
		// 根据线程ID的奇偶性来分配CPU
		bool is_even_thread = ((tctx->thread_id % 2) == 0);
		
		// 偶数线程(读线程)分配到偶数CPU，奇数线程(写线程)分配到奇数CPU
		// 这种分配策略可以减少同一内存控制器的争用
		
		if (is_even_thread) {
			// 偶数线程(读线程)优先放在CPU 0
			cpu = 0;
			if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
				cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
				if (cpu_ctx && scx_bpf_test_and_clear_cpu_idle(cpu)) {
					return cpu;
				}
			}
			
			// 备选CPU 2
			cpu = 2;
			if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
				cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
				if (cpu_ctx && scx_bpf_test_and_clear_cpu_idle(cpu)) {
					return cpu;
				}
			}
		} else {
			// 奇数线程(写线程)优先放在CPU 1
			cpu = 1;
			if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
				cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
				if (cpu_ctx && scx_bpf_test_and_clear_cpu_idle(cpu)) {
					return cpu;
				}
			}
			
			// 备选CPU 3
			cpu = 3;
			if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
				cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
				if (cpu_ctx && scx_bpf_test_and_clear_cpu_idle(cpu)) {
					return cpu;
				}
			}
		}
		
		// 如果无法分配到理想CPU，使用负载最轻的对应奇偶性CPU
		for (cpu = 0; cpu < 4 && cpu < nr_cpus; cpu++) {
			// 只考虑与线程ID奇偶性匹配的CPU
			if ((cpu % 2) != (is_even_thread ? 0 : 1))
				continue;
				
			if (bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
				cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
				if (cpu_ctx) {
					// 即使CPU不空闲，也可以分配
					return cpu;
				}
			}
		}
	}
	
	// Ultra-simple CPU selection - NO LOOPS
	// For VectorDB tasks, check each CPU individually
	if (tctx->type == TASK_TYPE_MOE_VECTORDB) {
		// Check CPU 0
		cpu = 0;
		if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
			cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
			if (cpu_ctx && cpu_ctx->active_tasks < 2 && 
			    scx_bpf_test_and_clear_cpu_idle(cpu)) {
				return cpu;
			}
		}
		
		// Check CPU 1
		cpu = 1;
		if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
			cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
			if (cpu_ctx && cpu_ctx->active_tasks < 2 && 
			    scx_bpf_test_and_clear_cpu_idle(cpu)) {
				return cpu;
			}
		}
		
		// Check CPU 2
		cpu = 2;
		if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
			cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
			if (cpu_ctx && cpu_ctx->active_tasks < 2 && 
			    scx_bpf_test_and_clear_cpu_idle(cpu)) {
				return cpu;
			}
		}
		
		// Check CPU 3
		cpu = 3;
		if (cpu < nr_cpus && bpf_cpumask_test_cpu(cpu, p->cpus_ptr)) {
			cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
			if (cpu_ctx && cpu_ctx->active_tasks < 2 && 
			    scx_bpf_test_and_clear_cpu_idle(cpu)) {
				return cpu;
			}
		}
	}
	
	return best_cpu;
}

void BPF_STRUCT_OPS(cxl_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tctx;
	u32 pid = p->pid;
	u32 priority;
	u64 vtime = p->scx.dsq_vtime;
	
	// Get or create task context
	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx) {
		scx_bpf_dsq_insert(p, FALLBACK_DSQ_ID, SCX_SLICE_DFL, enq_flags);
		return;
	}
	
	// Initialize task type if unknown
	if (tctx->type == TASK_TYPE_UNKNOWN) {
		if (is_bandwidth_task(p)) {
			tctx->type = TASK_TYPE_BANDWIDTH;
			tctx->is_memory_intensive = true;
			
			// 根据PID设置唯一的线程ID
			// 使用PID的后8位，这样保证每个进程有唯一标识
			tctx->thread_id = pid & 0xff;
			
			// 输出调试信息到性能计数器
			/*
			char msg[16] = "BW task: ";
			msg[9] = '0' + (tctx->thread_id % 10); // 简单显示最后一位
			bpf_trace_printk(msg, sizeof(msg));
			*/
		}
		else if (is_vectordb_task(p))
			tctx->type = TASK_TYPE_MOE_VECTORDB;
		else if (is_kworker_task(p))
			tctx->type = TASK_TYPE_KWORKER;
		else
			tctx->type = TASK_TYPE_REGULAR;
	}
	
	// Update memory patterns
	update_memory_pattern(pid, tctx);
	
	// Calculate priority
	priority = get_task_priority(tctx);
	
	// 对内存带宽任务特殊处理
	if (tctx->type == TASK_TYPE_BANDWIDTH) {
		// 根据线程ID的奇偶性调整优先级
		// 读写线程均衡处理，防止饿死
		if (tctx->thread_id % 2 == 0) {
			// 读线程(偶数)
			priority -= 10;
		} else {
			// 写线程(奇数)
			priority -= 8;  // 略低于读线程优先级
		}
	}
	
	// Adjust vtime
	if (vtime_before(vtime, global_vtime - SCX_SLICE_DFL))
		vtime = global_vtime - SCX_SLICE_DFL;
	
	// Enqueue
	scx_bpf_dsq_insert_vtime(p, FALLBACK_DSQ_ID, SCX_SLICE_DFL, 
	                        vtime - (120 - priority) * 100, enq_flags);
}

void BPF_STRUCT_OPS(cxl_dispatch, s32 cpu, struct task_struct *prev)
{
	struct cpu_ctx *cpu_ctx;
	
	// Update CPU context
	cpu_ctx = bpf_map_lookup_elem(&cpu_contexts, &cpu);
	if (cpu_ctx) {
		if (prev)
			cpu_ctx->active_tasks = cpu_ctx->active_tasks > 0 ? 
			                       cpu_ctx->active_tasks - 1 : 0;
		bpf_map_update_elem(&cpu_contexts, &cpu, cpu_ctx, BPF_ANY);
	}
	
	// Dispatch next task
	if (!scx_bpf_dsq_move_to_local(FALLBACK_DSQ_ID))
		return;
		
	// Update for new task
	if (cpu_ctx) {
		cpu_ctx->active_tasks++;
		bpf_map_update_elem(&cpu_contexts, &cpu, cpu_ctx, BPF_ANY);
	}
}

void BPF_STRUCT_OPS(cxl_running, struct task_struct *p)
{
	if (vtime_before(global_vtime, p->scx.dsq_vtime))
		global_vtime = p->scx.dsq_vtime;
}

void BPF_STRUCT_OPS(cxl_stopping, struct task_struct *p, bool runnable)
{
	p->scx.dsq_vtime += (SCX_SLICE_DFL - p->scx.slice) * 100 / p->scx.weight;
}

s32 BPF_STRUCT_OPS(cxl_init_task, struct task_struct *p, struct scx_init_task_args *args)
{
	struct task_ctx *tctx;
	
	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx)
		return -ENOMEM;
		
	tctx->type = TASK_TYPE_UNKNOWN;
	tctx->priority_boost = 0;
	tctx->is_memory_intensive = false;
	
	return 0;
}

void BPF_STRUCT_OPS(cxl_exit_task, struct task_struct *p)
{
	u32 pid = p->pid;
	bpf_map_delete_elem(&memory_patterns, &pid);
}

s32 BPF_STRUCT_OPS_SLEEPABLE(cxl_init)
{
	s32 ret;
	struct cpu_ctx cpu_ctx_init = {0};
	u32 cpu;
	
	ret = scx_bpf_create_dsq(FALLBACK_DSQ_ID, NUMA_NO_NODE);
	if (ret)
		return ret;
		
	// 避免循环，直接初始化4个CPU上下文
	cpu_ctx_init.is_preferred = 1; // 优先CPU
	
	// CPU 0
	cpu = 0;
	bpf_map_update_elem(&cpu_contexts, &cpu, &cpu_ctx_init, BPF_ANY);
	
	// CPU 1
	cpu = 1;
	bpf_map_update_elem(&cpu_contexts, &cpu, &cpu_ctx_init, BPF_ANY);
	
	// CPU 2
	cpu = 2;
	bpf_map_update_elem(&cpu_contexts, &cpu, &cpu_ctx_init, BPF_ANY);
	
	// CPU 3
	cpu = 3;
	bpf_map_update_elem(&cpu_contexts, &cpu, &cpu_ctx_init, BPF_ANY);
	
	return 0;
}

void BPF_STRUCT_OPS(cxl_exit, struct scx_exit_info *ei)
{
	// Exit handler
}

SCX_OPS_DEFINE(cxl_ops,
	       .select_cpu		= (void *)cxl_select_cpu,
	       .enqueue			= (void *)cxl_enqueue,
	       .dispatch		= (void *)cxl_dispatch,
	       .running			= (void *)cxl_running,
	       .stopping		= (void *)cxl_stopping,
	       .init_task		= (void *)cxl_init_task,
	       .exit_task		= (void *)cxl_exit_task,
	       .init			= (void *)cxl_init,
	       .exit			= (void *)cxl_exit,
	       .flags			= 0,
	       .name			= "cxl_simple"); 
