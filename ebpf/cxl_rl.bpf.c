/* SPDX-License-Identifier: GPL-2.0 */
/*
 * CXL RL (Reinforcement Learning) Scheduler
 * 使用强化学习策略进行CXL内存带宽优化调度
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char _license[] SEC("license") = "GPL";

#define MAX_CPUS 64
#define MAX_TASKS 1024
#define MAX_ACTIONS 8
#define MAX_STATES 8
#define HISTORY_SIZE 16
#define FALLBACK_DSQ_ID 0
#define LEARNING_RATE 10    // 实际值为0.1 (10/100)
#define DISCOUNT_FACTOR 95  // 实际值为0.95 (95/100)
#define EPSILON 10          // 实际值为0.1 (10/100) for epsilon-greedy

/* Simplified sched_ext definitions */
#ifndef SCX_SLICE_DFL
#define SCX_SLICE_DFL 20000000ULL
#endif
#define NUMA_NO_NODE (-1)

/* BPF helpers */
void scx_bpf_dsq_insert(struct task_struct *p, u64 dsq_id, u64 slice, u64 flags) __ksym;
void scx_bpf_dsq_insert_vtime(struct task_struct *p, u64 dsq_id, u64 slice, u64 vtime, u64 flags) __ksym;
bool scx_bpf_dsq_move_to_local(u64 dsq_id) __ksym;
s32 scx_bpf_create_dsq(u64 dsq_id, s32 node) __ksym;
bool scx_bpf_test_and_clear_cpu_idle(s32 cpu) __ksym;

/* RL状态定义 */
enum rl_state {
	STATE_LOW_BW = 0,
	STATE_HIGH_BW,
	STATE_READ_HEAVY,
	STATE_WRITE_HEAVY,
	STATE_BALANCED,
	STATE_CONTENDED,
	STATE_IDLE,
	STATE_MAX
};

/* RL动作定义 */
enum rl_action {
	ACTION_LOCAL = 0,
	ACTION_REMOTE,
	ACTION_ISOLATE_R,
	ACTION_ISOLATE_W,
	ACTION_COLOCATE,
	ACTION_SPREAD,
	ACTION_BOOST,
	ACTION_THROTTLE
};

/* 带宽信息 */
struct bw_info {
	u64 read_bytes;
	u64 write_bytes;
	u64 timestamp;
};

/* Q值条目 */
struct q_entry {
	s32 values[MAX_ACTIONS];
};

/* 任务上下文 */
struct task_ctx {
	u8 state;
	u8 action;
	struct bw_info bw;
	s32 reward;
	bool is_reader;
	bool is_writer;
};

/* CPU上下文 */
struct cpu_ctx {
	u32 nr_tasks;
	u32 nr_readers;
	u32 nr_writers;
	bool is_cxl;
};

/* Maps */
struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} task_stor SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAX_CPUS);
	__type(key, u32);
	__type(value, struct cpu_ctx);
} cpu_ctxs SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__uint(max_entries, MAX_STATES);
	__type(key, u32);
	__type(value, struct q_entry);
} q_table SEC(".maps");

/* 获取任务名称的前4个字符 */
static inline void get_task_comm_prefix(struct task_struct *p, char *prefix)
{
	bpf_probe_read_kernel_str(prefix, 5, p->comm);
}

/* 判断任务类型 */
static inline void classify_task(struct task_ctx *tctx, struct task_struct *p)
{
	char prefix[5] = {0};
	get_task_comm_prefix(p, prefix);
	
	// 简单的启发式判断
	if (prefix[0] == 'r' && prefix[1] == 'e' && prefix[2] == 'a' && prefix[3] == 'd')
		tctx->is_reader = true;
	else if (prefix[0] == 'w' && prefix[1] == 'r' && prefix[2] == 'i' && prefix[3] == 't')
		tctx->is_writer = true;
	else if (prefix[0] == 'd' && prefix[1] == 'o' && prefix[2] == 'u' && prefix[3] == 'b') {
		// double_bandwidth测试程序 - 根据PID奇偶性判断
		u32 pid = p->pid;
		tctx->is_reader = (pid & 1) == 0;
		tctx->is_writer = !tctx->is_reader;
	}
}

/* 计算带宽 (简化版) */
static inline u32 calc_bandwidth_mb(struct bw_info *bw, u64 now)
{
	u64 delta = now - bw->timestamp;
	if (delta < 1000000) // < 1ms
		return 0;
	
	u64 bytes = bw->read_bytes + bw->write_bytes;
	// 避免除法，使用移位近似: MB/s ≈ bytes >> 20 (for 1ms interval)
	return (u32)(bytes >> 10); // 简化计算
}

/* 确定状态 */
static inline u8 determine_state(struct task_ctx *tctx, u32 bw_mb)
{
	if (bw_mb < 100)
		return STATE_LOW_BW;
	else if (bw_mb > 1000) {
		if (tctx->is_reader && !tctx->is_writer)
			return STATE_READ_HEAVY;
		else if (tctx->is_writer && !tctx->is_reader)
			return STATE_WRITE_HEAVY;
		else
			return STATE_HIGH_BW;
	}
	else
		return STATE_BALANCED;
}

/* 计算奖励 */
static inline s32 calc_reward(struct task_ctx *tctx, struct cpu_ctx *cpu, u32 bw_mb)
{
	s32 reward = 0;
	
	// 带宽奖励
	if (bw_mb > 500)
		reward += 50;
	else if (bw_mb > 100)
		reward += 20;
	else
		reward -= 10;
	
	// CPU负载奖励
	if (cpu && cpu->nr_tasks < 4)
		reward += 10;
	else if (cpu && cpu->nr_tasks > 8)
		reward -= 20;
	
	// 读写分离奖励
	if (tctx->is_reader && cpu && cpu->nr_writers == 0)
		reward += 15;
	else if (tctx->is_writer && cpu && cpu->nr_readers == 0)
		reward += 15;
	
	return reward;
}

/* 选择动作 (简化版) */
static inline u8 select_action(u8 state)
{
	struct q_entry *q;
	u32 state_idx = state;
	u32 rand = bpf_get_prandom_u32();
	
	// Epsilon-greedy
	if ((rand % 100) < EPSILON)
		return rand % MAX_ACTIONS;
	
	q = bpf_map_lookup_elem(&q_table, &state_idx);
	if (!q)
		return ACTION_LOCAL;
	
	// 找最大Q值
	s32 max_q = q->values[0];
	u8 best = 0;
	
	// 手动展开循环避免验证器问题
	if (q->values[1] > max_q) { max_q = q->values[1]; best = 1; }
	if (q->values[2] > max_q) { max_q = q->values[2]; best = 2; }
	if (q->values[3] > max_q) { max_q = q->values[3]; best = 3; }
	if (q->values[4] > max_q) { max_q = q->values[4]; best = 4; }
	if (q->values[5] > max_q) { max_q = q->values[5]; best = 5; }
	if (q->values[6] > max_q) { max_q = q->values[6]; best = 6; }
	if (q->values[7] > max_q) { max_q = q->values[7]; best = 7; }
	
	return best;
}

/* 更新Q表 (简化版，避免除法) */
static inline void update_q_value(u8 state, u8 action, s32 reward, u8 next_state)
{
	struct q_entry *q, *next_q;
	u32 s_idx = state, ns_idx = next_state;
	
	q = bpf_map_lookup_elem(&q_table, &s_idx);
	next_q = bpf_map_lookup_elem(&q_table, &ns_idx);
	if (!q || !next_q)
		return;
	
	// 找下一状态最大Q值
	s32 max_next = next_q->values[0];
	if (next_q->values[1] > max_next) max_next = next_q->values[1];
	if (next_q->values[2] > max_next) max_next = next_q->values[2];
	if (next_q->values[3] > max_next) max_next = next_q->values[3];
	if (next_q->values[4] > max_next) max_next = next_q->values[4];
	if (next_q->values[5] > max_next) max_next = next_q->values[5];
	if (next_q->values[6] > max_next) max_next = next_q->values[6];
	if (next_q->values[7] > max_next) max_next = next_q->values[7];
	
	// 简化Q更新，避免除法
	// Q = Q + α(r + γ*max_next - Q)
	// 使用移位代替除法: α=0.1≈1/10, γ=0.95≈15/16
	s32 old_q = q->values[action];
	s32 target = reward + ((max_next * 15) >> 4); // γ ≈ 15/16
	s32 delta = target - old_q;
	q->values[action] = old_q + (delta >> 3); // α ≈ 1/8
	
	bpf_map_update_elem(&q_table, &s_idx, q, BPF_ANY);
}

/* 根据动作选择CPU */
static s32 apply_action(struct task_ctx *tctx, u8 action, s32 prev_cpu)
{
	struct cpu_ctx *cpu;
	u32 cpu_id;
	
	switch (action) {
	case ACTION_LOCAL:
		return prev_cpu;
		
	case ACTION_REMOTE:
		// 选择CXL CPU (假设CPU 2,3)
		cpu_id = 2;
		cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
		if (cpu && cpu->nr_tasks < 4)
			return 2;
		cpu_id = 3;
		cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
		if (cpu && cpu->nr_tasks < 4)
			return 3;
		break;
		
	case ACTION_ISOLATE_R:
		// 读任务到偶数CPU
		if (tctx->is_reader) {
			cpu_id = 0;
			cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
			if (cpu && cpu->nr_writers == 0 && cpu->nr_tasks < 4)
				return 0;
			cpu_id = 2;
			cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
			if (cpu && cpu->nr_writers == 0 && cpu->nr_tasks < 4)
				return 2;
		}
		break;
		
	case ACTION_ISOLATE_W:
		// 写任务到奇数CPU
		if (tctx->is_writer) {
			cpu_id = 1;
			cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
			if (cpu && cpu->nr_readers == 0 && cpu->nr_tasks < 4)
				return 1;
			cpu_id = 3;
			cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
			if (cpu && cpu->nr_readers == 0 && cpu->nr_tasks < 4)
				return 3;
		}
		break;
		
	case ACTION_SPREAD: {
		// 找负载最轻的CPU
		u32 min_tasks = 0xFFFF;
		s32 best_cpu = prev_cpu;
		
		for (cpu_id = 0; cpu_id < 4; cpu_id++) {
			cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
			if (cpu && cpu->nr_tasks < min_tasks) {
				min_tasks = cpu->nr_tasks;
				best_cpu = cpu_id;
			}
		}
		return best_cpu;
	}
	}
	
	return prev_cpu;
}

/* sched_ext operations */

SEC("struct_ops/cxl_rl_select_cpu")
s32 cxl_rl_select_cpu(struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	struct task_ctx *tctx;
	struct cpu_ctx *cpu;
	u64 now = bpf_ktime_get_ns();
	u32 bw_mb;
	u8 action;
	s32 cpu_id;
	
	tctx = bpf_task_storage_get(&task_stor, p, 0, 0);
	if (!tctx)
		return prev_cpu;
	
	// 计算带宽
	bw_mb = calc_bandwidth_mb(&tctx->bw, now);
	
	// 确定状态
	tctx->state = determine_state(tctx, bw_mb);
	
	// 选择动作
	action = select_action(tctx->state);
	tctx->action = action;
	
	// 应用动作
	cpu_id = apply_action(tctx, action, prev_cpu);
	
	// 验证CPU可用
	if (cpu_id >= 0 && cpu_id < 4) {
		if (bpf_cpumask_test_cpu(cpu_id, p->cpus_ptr)) {
			if (scx_bpf_test_and_clear_cpu_idle(cpu_id))
				return cpu_id;
		}
	}
	
	return prev_cpu;
}

SEC("struct_ops/cxl_rl_enqueue")
void cxl_rl_enqueue(struct task_struct *p, u64 enq_flags)
{
	struct task_ctx *tctx;
	struct cpu_ctx *cpu;
	u64 now = bpf_ktime_get_ns();
	u32 bw_mb;
	s32 reward;
	u8 new_state;
	u32 cpu_id = bpf_get_smp_processor_id();
	
	// 获取或创建任务上下文
	tctx = bpf_task_storage_get(&task_stor, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx) {
		scx_bpf_dsq_insert(p, FALLBACK_DSQ_ID, SCX_SLICE_DFL, enq_flags);
		return;
	}
	
	// 首次初始化
	if (tctx->bw.timestamp == 0) {
		tctx->bw.timestamp = now;
		classify_task(tctx, p);
	}
	
	// 模拟带宽更新 (实际应从PMU读取)
	u64 delta = now - tctx->bw.timestamp;
	if (delta > 1000000) { // 每1ms
		if (tctx->is_reader)
			tctx->bw.read_bytes += 1048576; // 1MB
		if (tctx->is_writer)
			tctx->bw.write_bytes += 1048576;
		tctx->bw.timestamp = now;
	}
	
	// 计算带宽和奖励
	bw_mb = calc_bandwidth_mb(&tctx->bw, now);
	cpu = bpf_map_lookup_elem(&cpu_ctxs, &cpu_id);
	reward = calc_reward(tctx, cpu, bw_mb);
	tctx->reward += reward;
	
	// 更新Q表
	new_state = determine_state(tctx, bw_mb);
	update_q_value(tctx->state, tctx->action, reward, new_state);
	tctx->state = new_state;
	
	// 根据累积奖励调整优先级
	u64 vtime = p->scx.dsq_vtime;
	if (tctx->reward > 0) {
		// 简单的优先级提升
		vtime -= (tctx->reward << 6); // reward * 64
	}
	
	// 入队
	scx_bpf_dsq_insert_vtime(p, FALLBACK_DSQ_ID, SCX_SLICE_DFL, vtime, enq_flags);
}

SEC("struct_ops/cxl_rl_dispatch")
void cxl_rl_dispatch(s32 cpu, struct task_struct *prev)
{
	struct cpu_ctx *ctx;
	struct task_ctx *tctx;
	
	// 更新CPU统计
	ctx = bpf_map_lookup_elem(&cpu_ctxs, &cpu);
	if (ctx && prev) {
		tctx = bpf_task_storage_get(&task_stor, prev, 0, 0);
		if (tctx) {
			if (tctx->is_reader && ctx->nr_readers > 0)
				ctx->nr_readers--;
			if (tctx->is_writer && ctx->nr_writers > 0)
				ctx->nr_writers--;
		}
		if (ctx->nr_tasks > 0)
			ctx->nr_tasks--;
	}
	
	// 分发下一个任务
	scx_bpf_dsq_move_to_local(FALLBACK_DSQ_ID);
}

SEC("struct_ops/cxl_rl_running")
void cxl_rl_running(struct task_struct *p)
{
	struct cpu_ctx *ctx;
	struct task_ctx *tctx;
	s32 cpu = bpf_get_smp_processor_id();
	
	// 更新CPU统计
	ctx = bpf_map_lookup_elem(&cpu_ctxs, &cpu);
	tctx = bpf_task_storage_get(&task_stor, p, 0, 0);
	
	if (ctx && tctx) {
		ctx->nr_tasks++;
		if (tctx->is_reader)
			ctx->nr_readers++;
		if (tctx->is_writer)
			ctx->nr_writers++;
	}
}

SEC("struct_ops/cxl_rl_stopping")
void cxl_rl_stopping(struct task_struct *p, bool runnable)
{
	// 可扩展：收集更多统计
}

SEC("struct_ops/cxl_rl_init_task")
s32 cxl_rl_init_task(struct task_struct *p, struct scx_init_task_args *args)
{
	struct task_ctx *tctx;
	
	tctx = bpf_task_storage_get(&task_stor, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (!tctx)
		return -12; // ENOMEM
		
	// 初始化
	tctx->state = STATE_LOW_BW;
	tctx->action = ACTION_LOCAL;
	tctx->reward = 0;
	
	return 0;
}

SEC("struct_ops/cxl_rl_exit_task")
void cxl_rl_exit_task(struct task_struct *p)
{
	// 清理任务相关资源
}

SEC("struct_ops.s/cxl_rl_init")
s32 cxl_rl_init(void)
{
	struct cpu_ctx cpu_init = {0};
	struct q_entry q_init = {0};
	u32 i;
	
	// 创建DSQ
	if (scx_bpf_create_dsq(FALLBACK_DSQ_ID, NUMA_NO_NODE))
		return -1;
	
	// 初始化CPU上下文
	for (i = 0; i < 4; i++) {
		cpu_init.is_cxl = (i >= 2); // CPU 2,3连接CXL
		bpf_map_update_elem(&cpu_ctxs, &i, &cpu_init, BPF_ANY);
	}
	
	// 初始化Q表
	for (i = 0; i < MAX_STATES; i++) {
		bpf_map_update_elem(&q_table, &i, &q_init, BPF_ANY);
	}
	
	return 0;
}

SEC("struct_ops/cxl_rl_exit")
void cxl_rl_exit(struct scx_exit_info *ei)
{
	// 清理
}

SEC(".struct_ops.link")
struct sched_ext_ops cxl_rl_ops = {
	.select_cpu	= (void *)cxl_rl_select_cpu,
	.enqueue	= (void *)cxl_rl_enqueue,
	.dispatch	= (void *)cxl_rl_dispatch,
	.running	= (void *)cxl_rl_running,
	.stopping	= (void *)cxl_rl_stopping,
	.init_task	= (void *)cxl_rl_init_task,
	.exit_task	= (void *)cxl_rl_exit_task,
	.init		= (void *)cxl_rl_init,
	.exit		= (void *)cxl_rl_exit,
	.flags		= 0,
	.name		= "cxl_rl",
};