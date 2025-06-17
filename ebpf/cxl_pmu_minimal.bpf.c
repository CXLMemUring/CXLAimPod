/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Ultra-minimal CXL scheduler - avoids all loop issues
 */

#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

#define FALLBACK_DSQ_ID 0

/* Minimal task context - just type detection */
struct task_ctx {
	bool is_vectordb;
	bool is_kworker;
};

/* Maps */
struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_ctx_stor SEC(".maps");

/* Global state */
static u64 global_vtime = 0;

/* Helper functions - ultra-simplified */

static inline bool vtime_before(u64 a, u64 b)
{
	return (s64)(a - b) < 0;
}

static inline bool is_vectordb_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	
	// Ultra-simple check - just first character
	return (comm[0] == 'v') || (comm[0] == 'f') || (comm[0] == 'p');
}

static inline bool is_kworker_task(struct task_struct *p)
{
	char comm[16];
	bpf_probe_read_kernel_str(comm, sizeof(comm), p->comm);
	return (comm[0] == 'k' && comm[1] == 'w');
}

/* sched_ext operations - minimal implementation */

s32 BPF_STRUCT_OPS(cxl_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	struct task_ctx *tctx;
	
	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0, 0);
	if (!tctx)
		return prev_cpu;
	
	// Ultra-simple CPU selection - NO LOOPS
	// For VectorDB tasks, prefer CPU 0-1, others use prev_cpu
	if (tctx->is_vectordb) {
		if (bpf_cpumask_test_cpu(0, p->cpus_ptr) && scx_bpf_test_and_clear_cpu_idle(0))
			return 0;
		if (bpf_cpumask_test_cpu(1, p->cpus_ptr) && scx_bpf_test_and_clear_cpu_idle(1))
			return 1;
	}
	
	return prev_cpu;
}

void BPF_STRUCT_OPS(cxl_enqueue, struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tctx;
	u64 vtime = p->scx.dsq_vtime;
	u64 slice = SCX_SLICE_DFL;
	
	// Get or create task context
	tctx = bpf_task_storage_get(&task_ctx_stor, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx) {
		scx_bpf_dsq_insert(p, FALLBACK_DSQ_ID, slice, enq_flags);
		return;
	}
	
	// Initialize task type if needed
	if (!tctx->is_vectordb && !tctx->is_kworker) {
		tctx->is_vectordb = is_vectordb_task(p);
		tctx->is_kworker = is_kworker_task(p);
	}
	
	// Adjust vtime - simple logic
	if (vtime_before(vtime, global_vtime - slice))
		vtime = global_vtime - slice;
	
	// VectorDB tasks get priority boost
	if (tctx->is_vectordb)
		vtime -= slice;  // Higher priority
	
	// Kworkers get lower priority
	if (tctx->is_kworker)
		vtime += slice;  // Lower priority
	
	scx_bpf_dsq_insert_vtime(p, FALLBACK_DSQ_ID, slice, vtime, enq_flags);
}

void BPF_STRUCT_OPS(cxl_dispatch, s32 cpu, struct task_struct *prev)
{
	// Ultra-simple dispatch
	scx_bpf_dsq_move_to_local(FALLBACK_DSQ_ID);
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
		
	tctx->is_vectordb = false;
	tctx->is_kworker = false;
	
	return 0;
}

void BPF_STRUCT_OPS(cxl_exit_task, struct task_struct *p)
{
	// Task cleanup - storage automatically freed
}

s32 BPF_STRUCT_OPS_SLEEPABLE(cxl_init)
{
	return scx_bpf_create_dsq(FALLBACK_DSQ_ID, NUMA_NO_NODE);
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
	       .name			= "cxl_minimal"); 
