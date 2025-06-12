/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Minimal CXL PMU-aware scheduler - basic version without loops
 * This version avoids eBPF verifier complexity and instruction limits
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

/* sched_ext specific definitions - use what's in vmlinux.h */
#define BPF_STRUCT_OPS(name, args...) name(args)
#define BPF_STRUCT_OPS_SLEEPABLE(name, args...) name(args)
#define NUMA_NO_NODE		(-1)

#define SCX_OPS_DEFINE(name, ...) \
	SEC(".struct_ops") \
	struct sched_ext_ops name = { __VA_ARGS__ }

char _license[] SEC("license") = "GPL";

/* Minimal scheduler operations */

SEC("struct_ops/minimal_select_cpu")
s32 minimal_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	/* Simple CPU selection - just return previous CPU */
	return prev_cpu;
}

SEC("struct_ops/minimal_enqueue")
void minimal_enqueue(struct task_struct *p, u64 enq_flags)
{
	scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, enq_flags);
}

SEC("struct_ops/minimal_dispatch")
void minimal_dispatch(s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(0);
}

SEC("struct_ops/minimal_running")
void minimal_running(struct task_struct *p)
{
	/* No operation needed */
}

SEC("struct_ops/minimal_stopping")
void minimal_stopping(struct task_struct *p, bool runnable)
{
	/* No operation needed */
}

SEC("struct_ops.s/minimal_init")
s32 minimal_init(void)
{
	return scx_bpf_create_dsq(0, NUMA_NO_NODE);
}

SEC("struct_ops/minimal_exit")
void minimal_exit(struct scx_exit_info *ei)
{
	/* No cleanup needed */
}

SEC(".struct_ops.link")
struct sched_ext_ops minimal_ops = {
	.select_cpu		= (void *)minimal_select_cpu,
	.enqueue		= (void *)minimal_enqueue,
	.dispatch		= (void *)minimal_dispatch,
	.running		= (void *)minimal_running,
	.stopping		= (void *)minimal_stopping,
	.init			= (void *)minimal_init,
	.exit			= (void *)minimal_exit,
	.flags			= 0,
	.name			= "minimal",
};