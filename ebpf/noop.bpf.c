/* SPDX-License-Identifier: GPL-2.0 */
/*
 * No-operation sched_ext scheduler - doesn't access any parameters
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char _license[] SEC("license") = "GPL";

SEC("struct_ops/noop_select_cpu")
s32 noop_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	return 0; /* Always return CPU 0 */
}

SEC("struct_ops/noop_enqueue")
void noop_enqueue(struct task_struct *p, u64 enq_flags)
{
	scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, 0);
}

SEC("struct_ops/noop_dispatch")
void noop_dispatch(s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(0);
}

SEC("struct_ops.s/noop_init")
s32 noop_init(void)
{
	return scx_bpf_create_dsq(0, -1);
}

SEC("struct_ops/noop_exit")
void noop_exit(struct scx_exit_info *ei)
{
}

SEC(".struct_ops.link")
struct sched_ext_ops noop_ops = {
	.select_cpu	= (void *)noop_select_cpu,
	.enqueue	= (void *)noop_enqueue,
	.dispatch	= (void *)noop_dispatch,
	.init		= (void *)noop_init,
	.exit		= (void *)noop_exit,
	.name		= "noop",
};