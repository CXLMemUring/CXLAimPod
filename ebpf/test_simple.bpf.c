/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Ultra-minimal sched_ext test - just return values without accessing anything
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char _license[] SEC("license") = "GPL";

SEC("struct_ops/test_select_cpu")
s32 test_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	return prev_cpu;
}

SEC("struct_ops/test_enqueue")
void test_enqueue(struct task_struct *p, u64 enq_flags)
{
	scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, enq_flags);
}

SEC("struct_ops/test_dispatch")
void test_dispatch(s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(0);
}

SEC("struct_ops.s/test_init")
s32 test_init(void)
{
	return scx_bpf_create_dsq(0, -1);
}

SEC("struct_ops/test_exit")
void test_exit(struct scx_exit_info *ei)
{
}

SEC(".struct_ops.link")
struct sched_ext_ops test_ops = {
	.select_cpu	= (void *)test_select_cpu,
	.enqueue	= (void *)test_enqueue,
	.dispatch	= (void *)test_dispatch,
	.init		= (void *)test_init,
	.exit		= (void *)test_exit,
	.name		= "test",
};