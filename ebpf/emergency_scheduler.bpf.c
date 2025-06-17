#include <scx/common.bpf.h>
char _license[] SEC("license") = "GPL";
s32 BPF_STRUCT_OPS(emergency_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags) { return prev_cpu; }
void BPF_STRUCT_OPS(emergency_enqueue, struct task_struct *p, u64 enq_flags) { scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, enq_flags); }
void BPF_STRUCT_OPS(emergency_dispatch, s32 cpu, struct task_struct *prev) { scx_bpf_dsq_move_to_local(0); }
s32 BPF_STRUCT_OPS_SLEEPABLE(emergency_init) { return scx_bpf_create_dsq(0, NUMA_NO_NODE); }
void BPF_STRUCT_OPS(emergency_exit, struct scx_exit_info *ei) {}
SCX_OPS_DEFINE(emergency_ops, .select_cpu = (void *)emergency_select_cpu, .enqueue = (void *)emergency_enqueue, .dispatch = (void *)emergency_dispatch, .init = (void *)emergency_init, .exit = (void *)emergency_exit, .name = "emergency");
