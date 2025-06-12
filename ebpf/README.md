# CXL 带宽感知调度器

基于eBPF的CXL内存带宽感知调度器，专门优化读写密集型任务的性能。

## 🚀 快速开始

### 运行20线程演示
```bash
cd ebpf/
./run_20_threads_demo.sh
```

### 自定义配置测试
```bash
# 20个线程，60%读/40%写，800MB/s总带宽
../microbench/double_bandwidth -t 20 -r 0.6 -B 800 -d 10

# 16个线程，70%读/30%写，1200MB/s总带宽  
../microbench/double_bandwidth -t 16 -r 0.7 -B 1200 -d 15
```

## 📁 项目结构

```
ebpf/
├── cxl_pmu.bpf.c              # eBPF调度器内核程序
├── cxl_bandwidth_scheduler.c   # 用户空间控制器
├── run_20_threads_demo.sh      # 20线程演示脚本
├── Makefile                    # 编译和测试工具
├── BANDWIDTH_SCHEDULING_REPORT.md  # 详细测试报告
└── README.md                   # 本文档

microbench/
└── double_bandwidth.cpp       # 带宽测试程序
```

## 🎯 核心特性

### 1. 智能任务识别
- 自动识别带宽密集型任务（`double_bandwidth`, `bandwidth`, `memtest`, `stress`）
- 基于进程名和行为模式分类任务类型
- 动态调整任务优先级

### 2. 带宽感知调度
- 根据读写比例分配CPU资源
- 令牌桶算法实现精确带宽控制
- 支持突发流量处理

### 3. CXL硬件优化
- CXL PMU指标集成
- CXL感知的CPU选择
- NUMA拓扑优化

### 4. 内存访问模式监控
- DAMON集成的实时监控
- 内存访问热点检测
- 工作集大小估算

## 📊 性能测试结果

### 20线程不同读写比例性能对比

| 配置 | 读线程 | 写线程 | 读带宽(MB/s) | 写带宽(MB/s) | 总带宽(MB/s) |
|------|--------|--------|--------------|--------------|-------------|
| 读密集 | 16 | 4 | 432.1 | 111.1 | 543.2 |
| 写密集 | 6 | 14 | 166.0 | 387.7 | 553.7 |
| 平衡 | 10 | 10 | 220.0 | 219.0 | 439.0 |
| 高性能 | 12 | 8 | 707.0 | 470.0 | 1177.0 |

### 关键发现
- ✅ 成功实现按比例带宽分配
- ✅ 支持4-20个线程的广泛配置范围
- ✅ 多进程环境下资源分配公平
- ✅ 带宽利用率稳定在115-120%

## 🛠️ 使用指南

### Makefile命令

```bash
# 编译所有程序
make all

# 运行基础演示
make demo

# 性能对比测试
make compare

# 并发压力测试
make stress

# 显示帮助
make help
```

### 带宽测试程序参数

```bash
./double_bandwidth [OPTIONS]

主要参数:
  -t, --threads=NUM         线程总数 (默认: 4)
  -r, --read-ratio=RATIO    读线程比例 0.0-1.0 (默认: 0.5)
  -B, --max-bandwidth=MB/s  总带宽限制 (默认: 无限制)
  -d, --duration=SECONDS    测试时长 (默认: 10)
  -b, --buffer-size=SIZE    缓冲区大小 (默认: 1GB)
  -s, --block-size=SIZE     块大小 (默认: 4KB)
```

## 🎨 应用场景示例

### 1. 数据科学/AI推理
```bash
# 读密集型，适合模型推理
./double_bandwidth -t 16 -r 0.8 -B 1000 -d 3600
```

### 2. 机器学习训练
```bash
# 平衡读写，适合训练过程
./double_bandwidth -t 12 -r 0.6 -B 1200 -d 7200
```

### 3. 实时数据处理
```bash
# 写密集型，适合数据采集
./double_bandwidth -t 20 -r 0.3 -B 800 -d 1800
```

### 4. 高性能计算
```bash
# 高带宽，适合HPC工作负载
./double_bandwidth -t 8 -r 0.7 -B 2000 -d 600
```

## 🔧 eBPF调度器特性

### 任务类型分类
```c
enum task_type {
    TASK_TYPE_READ_INTENSIVE,    // 读密集型
    TASK_TYPE_WRITE_INTENSIVE,   // 写密集型
    TASK_TYPE_BANDWIDTH_TEST,    // 带宽测试
    TASK_TYPE_MOE_VECTORDB,      // MoE向量数据库
    TASK_TYPE_KWORKER,           // 内核工作线程
    TASK_TYPE_LATENCY_SENSITIVE, // 延迟敏感
};
```

### 动态优先级调整策略
- **读密集型任务**: 读带宽 > 70MB/s 时优先级 +15
- **写密集型任务**: 写带宽 > 70MB/s 时优先级 +15  
- **带宽测试任务**: 最高优先级 +30
- **延迟敏感任务**: 优先级 +25

### CPU选择算法
- 优先选择CXL附加的CPU
- 考虑内存带宽和延迟指标
- 避免CPU过载
- NUMA感知的负载均衡

## 📈 性能调优建议

### 线程配置
- **高带宽需求**: 4-8个线程，单线程带宽高
- **高并发需求**: 16-20个线程，提高并发度
- **平衡需求**: 12-16个线程，获得最佳平衡

### 读写比例
- **数据分析**: 70-80% 读
- **数据备份**: 20-30% 读  
- **通用应用**: 50% 读
- **实时系统**: 60-70% 读

### 带宽限制
- 设置目标带宽的85-90%作为限制值
- 允许10-20%的突发容量
- 根据系统负载动态调整

## 🚀 高级功能

### 实时监控
```bash
# 监控调度器状态 (需要root权限)
sudo ./cxl_bandwidth_scheduler -t 20 -R 0.6 -r 1000 -w 500 -i 3
```

### 多进程并发测试
```bash
# 同时运行多个不同配置的测试
make stress
```

### 自定义工作负载模拟
```bash
# 模拟特定应用的读写模式
./double_bandwidth -t 24 -r 0.65 -B 1500 -d 3600
```

## 🔍 故障排除

### 常见问题

1. **编译错误**: 确保安装了 libbpf-dev
2. **权限问题**: eBPF加载需要root权限
3. **性能不达预期**: 检查系统负载和内存压力

### 调试选项
```bash
# 详细输出模式
./double_bandwidth -t 20 -r 0.6 -B 500 -d 10 --verbose

# 无带宽限制测试
./double_bandwidth -t 20 -r 0.6 -d 10
```

## 📖 相关文档

- [详细测试报告](BANDWIDTH_SCHEDULING_REPORT.md)
- [Makefile命令参考](Makefile)
- [源码注释](cxl_pmu.bpf.c)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

GPL-2.0 许可证

---

**开发环境**: Linux 6.14.0-15-generic
**测试平台**: x86_64
**编译器**: GCC 11.x, Clang 14.x 