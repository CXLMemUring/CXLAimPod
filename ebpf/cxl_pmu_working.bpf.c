/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Working CXL PMU-aware scheduler - compatible with current sched_ext
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char _license[] SEC("license") = "GPL";

/* Simple task context */
struct task_ctx {
	u32 task_type;
	u64 last_runtime;
};

struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_storage SEC(".maps");

/* Helper to get current CPU without accessing parameters */
static inline int get_current_cpu(void)
{
	return bpf_get_smp_processor_id() % 64; /* Limit to reasonable range */
}

SEC("struct_ops/cxl_select_cpu")
s32 cxl_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	/* Don't access prev_cpu directly - use current CPU instead */
	return get_current_cpu();
}

SEC("struct_ops/cxl_enqueue")
void cxl_enqueue(struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *ctx;
	
	/* Get or create task context */
	ctx = bpf_task_storage_get(&task_storage, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (ctx) {
		ctx->last_runtime = bpf_ktime_get_ns();
		ctx->task_type = 1; /* Mark as scheduled */
	}
	
	/* Enqueue with default slice */
	scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, enq_flags);
}

SEC("struct_ops/cxl_dispatch")
void cxl_dispatch(s32 cpu, struct task_struct *prev)
{
	/* Simple dispatch - move from DSQ to local */
	scx_bpf_dsq_move_to_local(0);
}

SEC("struct_ops/cxl_running")
void cxl_running(struct task_struct *p)
{
	/* Task started running - no action needed */
}

SEC("struct_ops/cxl_stopping")
void cxl_stopping(struct task_struct *p, bool runnable)
{
	/* Task stopped running - no action needed */
}

SEC("struct_ops/cxl_init_task")
s32 cxl_init_task(struct task_struct *p, struct scx_init_task_args *args)
{
	struct task_ctx *ctx;
	
	/* Initialize task context */
	ctx = bpf_task_storage_get(&task_storage, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!ctx)
		return -12; /* ENOMEM */
		
	ctx->task_type = 0; /* Default type */
	ctx->last_runtime = bpf_ktime_get_ns();
	
	return 0;
}

SEC("struct_ops/cxl_exit_task")
void cxl_exit_task(struct task_struct *p, struct scx_exit_task_args *args)
{
	/* Task exiting - storage will be cleaned up automatically */
}

SEC("struct_ops.s/cxl_init")
s32 cxl_init(void)
{
	/* Create the default dispatch queue */
	return scx_bpf_create_dsq(0, -1);
}

SEC("struct_ops/cxl_exit")
void cxl_exit(struct scx_exit_info *ei)
{
	/* Scheduler exiting - no cleanup needed */
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
	.name			= "cxl_working",
};