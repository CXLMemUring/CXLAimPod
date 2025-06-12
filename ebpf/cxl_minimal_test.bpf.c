/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Minimal CXL scheduler for testing
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char _license[] SEC("license") = "GPL";

SEC("struct_ops/cxl_select_cpu")
s32 cxl_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	return -1; /* Let kernel choose */
}

SEC("struct_ops/cxl_enqueue")
void cxl_enqueue(struct task_struct *p, u64 enq_flags)
{
	scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, 0);
}

SEC("struct_ops/cxl_dispatch")
void cxl_dispatch(s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(0);
}

SEC("struct_ops.s/cxl_init")
s32 cxl_init(void)
{
	return scx_bpf_create_dsq(0, -1);
}

SEC("struct_ops/cxl_exit")
void cxl_exit(struct scx_exit_info *ei)
{
}

SEC(".struct_ops.link")
struct sched_ext_ops cxl_ops = {
	.select_cpu	= (void *)cxl_select_cpu,
	.enqueue	= (void *)cxl_enqueue,
	.dispatch	= (void *)cxl_dispatch,
	.init		= (void *)cxl_init,
	.exit		= (void *)cxl_exit,
	.flags		= 0,
	.name		= "cxl_minimal",
};