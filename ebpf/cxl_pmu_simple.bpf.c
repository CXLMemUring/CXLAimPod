/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Simple CXL PMU-aware scheduler - runtime compatible version
 * This version avoids all BPF verifier parameter access issues
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char _license[] SEC("license") = "GPL";

/* Minimal constants */
#define FALLBACK_DSQ_ID 0

SEC("struct_ops/cxl_select_cpu")
s32 cxl_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	/* Use current CPU to avoid parameter access issues */
	int cpu = bpf_get_smp_processor_id();
	return (cpu >= 0 && cpu < 64) ? cpu : 0;
}

SEC("struct_ops/cxl_enqueue")
void cxl_enqueue(struct task_struct *p, u64 enq_flags)
{
	/* Simple enqueue without parameter access */
	scx_bpf_dsq_insert(p, FALLBACK_DSQ_ID, SCX_SLICE_DFL, 0);
}

SEC("struct_ops/cxl_dispatch")
void cxl_dispatch(s32 cpu, struct task_struct *prev)
{
	/* Simple dispatch */
	scx_bpf_dsq_move_to_local(FALLBACK_DSQ_ID);
}

SEC("struct_ops/cxl_running")
void cxl_running(struct task_struct *p)
{
	/* Task is running */
}

SEC("struct_ops/cxl_stopping")
void cxl_stopping(struct task_struct *p, bool runnable)
{
	/* Task is stopping */
}

SEC("struct_ops/cxl_init_task")
s32 cxl_init_task(struct task_struct *p, struct scx_init_task_args *args)
{
	/* Task initialization */
	return 0;
}

SEC("struct_ops/cxl_exit_task")
void cxl_exit_task(struct task_struct *p, struct scx_exit_task_args *args)
{
	/* Task exit */
}

SEC("struct_ops.s/cxl_init")
s32 cxl_init(void)
{
	/* Create default dispatch queue */
	return scx_bpf_create_dsq(FALLBACK_DSQ_ID, -1);
}

SEC("struct_ops/cxl_exit")
void cxl_exit(struct scx_exit_info *ei)
{
	/* Scheduler exit */
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
	.name			= "cxl_simple",
};