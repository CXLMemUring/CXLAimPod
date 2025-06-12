# Makefile for CXL PMU eBPF scheduler

CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL ?= bpftool
CC ?= gcc

# Directories
LIBBPF_DIR = /usr/lib/x86_64-linux-gnu
VMLINUX_H = vmlinux.h

# Compiler flags
BPF_CFLAGS := -O2 -g -Wall -Werror
BPF_CFLAGS += -target bpf
BPF_CFLAGS += -D__TARGET_ARCH_x86
BPF_CFLAGS += -I$(LIBBPF_DIR)
BPF_CFLAGS += -I.

USER_CFLAGS := -O2 -g -Wall
USER_CFLAGS += -I$(LIBBPF_DIR)

USER_LDFLAGS := -lbpf -lelf -lz

# Source files
BPF_SRC = cxl_pmu.bpf.c
BPF_SIMPLE_SRC = cxl_pmu_simple.bpf.c
BPF_MINIMAL_SRC = cxl_pmu_minimal.bpf.c
BPF_WORKING_SRC = cxl_pmu_working.bpf.c
BPF_OBJ = $(BPF_SRC:.c=.o)
BPF_SIMPLE_OBJ = $(BPF_SIMPLE_SRC:.c=.o)
BPF_MINIMAL_OBJ = $(BPF_MINIMAL_SRC:.c=.o)
BPF_WORKING_OBJ = $(BPF_WORKING_SRC:.c=.o)

USER_SRC = cxl_bandwidth_scheduler.c
USER_BIN = cxl_bandwidth_scheduler

# Targets
BANDWIDTH_TEST = ../microbench/double_bandwidth
SCHEDULER_CTRL = cxl_bandwidth_scheduler

# Default target - build minimal version to avoid all issues
all: $(BANDWIDTH_TEST) $(SCHEDULER_CTRL)

# Build complex version (may hit instruction limit)
complex: $(BPF_OBJ) $(USER_BIN)

# Build simple version (may still have loop issues)
simple: $(BPF_SIMPLE_OBJ) $(USER_BIN)

# Build minimal version (no loops, guaranteed to work)
minimal: $(BPF_MINIMAL_OBJ) $(USER_BIN)

# Build working version (parameter-access safe)
working: $(BPF_WORKING_OBJ) $(USER_BIN)

# Generate vmlinux.h if it doesn't exist
$(VMLINUX_H):
	@echo "Generating vmlinux.h..."
	@if command -v bpftool >/dev/null 2>&1; then \
		bpftool btf dump file /sys/kernel/btf/vmlinux format c > $(VMLINUX_H); \
	else \
		echo "Warning: bpftool not found, using system headers"; \
		touch $(VMLINUX_H); \
	fi

# Compile eBPF object
%.bpf.o: %.bpf.c $(VMLINUX_H)
	@echo "Compiling eBPF program $<..."
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@
	$(LLVM_STRIP) -g $@

# Compile userspace program
$(USER_BIN): $(USER_SRC)
	@echo "Compiling userspace program $<..."
	$(CC) $(USER_CFLAGS) $< -o $@ $(USER_LDFLAGS)

# Clean
clean:
	rm -f *.o $(USER_BIN) $(VMLINUX_H)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y \
		clang \
		llvm \
		libbpf-dev \
		libelf-dev \
		zlib1g-dev \
		linux-tools-common \
		linux-tools-generic \
		bpftool

# Load the minimal scheduler (recommended)
load: minimal
	@echo "Loading minimal CXL scheduler (no loops, guaranteed to work)..."
	@echo "Note: This requires root privileges and sched_ext support"
	sudo ./$(USER_BIN) cxl_pmu_minimal.bpf.o

# Load the simple scheduler
load-simple: simple
	@echo "Loading simple CXL scheduler..."
	@echo "Warning: This may still have loop issues"
	sudo ./$(USER_BIN) cxl_pmu_simple.bpf.o

# Load the complex scheduler (may fail due to instruction limit)
load-complex: complex
	@echo "Loading complex CXL scheduler..."
	@echo "Warning: This may fail due to eBPF instruction limit"
	sudo ./$(USER_BIN) cxl_pmu.bpf.o

# Test the scheduler
test: minimal
	@echo "Testing minimal scheduler..."
	sudo ./test_scheduler.sh

# Show eBPF program info
info:
	@if [ -f "cxl_pmu_minimal.bpf.o" ]; then \
		echo "=== Minimal Scheduler Info ==="; \
		file cxl_pmu_minimal.bpf.o; \
		readelf -S cxl_pmu_minimal.bpf.o | grep -E "(Name|\.maps|struct_ops)"; \
	fi
	@if [ -f "cxl_pmu_simple.bpf.o" ]; then \
		echo "=== Simple Scheduler Info ==="; \
		file cxl_pmu_simple.bpf.o; \
		readelf -S cxl_pmu_simple.bpf.o | grep -E "(Name|\.maps|struct_ops)"; \
	fi
	@if [ -f "cxl_pmu.bpf.o" ]; then \
		echo "=== Complex Scheduler Info ==="; \
		file cxl_pmu.bpf.o; \
		readelf -S cxl_pmu.bpf.o | grep -E "(Name|\.maps|struct_ops)"; \
	fi

# Emergency bypass - create simplest possible scheduler
emergency: 
	@echo "Creating emergency simple scheduler..."
	@echo '#include <scx/common.bpf.h>' > emergency_scheduler.bpf.c
	@echo 'char _license[] SEC("license") = "GPL";' >> emergency_scheduler.bpf.c
	@echo 's32 BPF_STRUCT_OPS(emergency_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags) { return prev_cpu; }' >> emergency_scheduler.bpf.c
	@echo 'void BPF_STRUCT_OPS(emergency_enqueue, struct task_struct *p, u64 enq_flags) { scx_bpf_dsq_insert(p, 0, SCX_SLICE_DFL, enq_flags); }' >> emergency_scheduler.bpf.c
	@echo 'void BPF_STRUCT_OPS(emergency_dispatch, s32 cpu, struct task_struct *prev) { scx_bpf_dsq_move_to_local(0); }' >> emergency_scheduler.bpf.c
	@echo 's32 BPF_STRUCT_OPS_SLEEPABLE(emergency_init) { return scx_bpf_create_dsq(0, NUMA_NO_NODE); }' >> emergency_scheduler.bpf.c
	@echo 'void BPF_STRUCT_OPS(emergency_exit, struct scx_exit_info *ei) {}' >> emergency_scheduler.bpf.c
	@echo 'SCX_OPS_DEFINE(emergency_ops, .select_cpu = (void *)emergency_select_cpu, .enqueue = (void *)emergency_enqueue, .dispatch = (void *)emergency_dispatch, .init = (void *)emergency_init, .exit = (void *)emergency_exit, .name = "emergency");' >> emergency_scheduler.bpf.c
	$(CLANG) $(BPF_CFLAGS) -c emergency_scheduler.bpf.c -o emergency_scheduler.bpf.o
	$(LLVM_STRIP) -g emergency_scheduler.bpf.o
	@echo "Emergency scheduler created: emergency_scheduler.bpf.o"
	@echo "Load with: sudo ./cxl_sched emergency_scheduler.bpf.o"

# Bandwidth test program (already compiled)
$(BANDWIDTH_TEST):
	@echo "Bandwidth test already compiled in microbench/"

# Scheduler controller
$(SCHEDULER_CTRL): $(USER_SRC)
	@echo "Compiling scheduler controller..."
	$(CC) $(USER_CFLAGS) $< -o $@ $(USER_LDFLAGS)

# Run scheduler with automatic bandwidth test
run-scheduler: $(SCHEDULER_CTRL) $(BANDWIDTH_TEST)
	@echo "=== Starting CXL Bandwidth-Aware Scheduler ==="
	@echo "Note: This requires root privileges for eBPF loading"
	@sudo ./$(SCHEDULER_CTRL) -t 20 -R 0.6 -r 1000 -w 500 -T -i 3

# Demo without eBPF (just bandwidth tests)
demo:
	@echo "=== Bandwidth Scheduling Demo (without eBPF) ==="
	@echo "This demo shows how bandwidth limits work with different thread ratios"
	@echo ""
	@echo "Running 3 different scenarios with 20 threads each:"
	@echo ""
	@echo "Scenario 1: Read-heavy workload (70% read, 30% write)"
	@cd ../microbench && ./double_bandwidth -t 20 -r 0.7 -d 8 -B 400
	@echo ""
	@echo "Scenario 2: Write-heavy workload (30% read, 70% write)"
	@cd ../microbench && ./double_bandwidth -t 20 -r 0.3 -d 8 -B 400
	@echo ""
	@echo "Scenario 3: Balanced workload (50% read, 50% write)"
	@cd ../microbench && ./double_bandwidth -t 20 -r 0.5 -d 8 -B 400

# Performance comparison
compare:
	@echo "=== Performance Comparison ==="
	@echo "Comparing different thread configurations:"
	@echo ""
	@echo "1. Few threads, high bandwidth limit:"
	@cd ../microbench && ./double_bandwidth -t 4 -r 0.6 -d 5 -B 1000
	@echo ""
	@echo "2. Many threads, medium bandwidth limit:"
	@cd ../microbench && ./double_bandwidth -t 20 -r 0.6 -d 5 -B 500
	@echo ""
	@echo "3. Many threads, low bandwidth limit:"
	@cd ../microbench && ./double_bandwidth -t 20 -r 0.6 -d 5 -B 200

# Stress test with multiple processes
stress:
	@echo "=== Stress Test: Multiple Bandwidth Processes ==="
	@echo "Starting 3 concurrent bandwidth test processes..."
	@cd ../microbench && \
		(./double_bandwidth -t 8 -r 0.7 -d 15 -B 300 &) && \
		(./double_bandwidth -t 6 -r 0.3 -d 15 -B 200 &) && \
		(./double_bandwidth -t 6 -r 0.5 -d 15 -B 250 &) && \
		wait

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Compile all programs"
	@echo "  test         - Run bandwidth tests with different configurations"
	@echo "  demo         - Demo bandwidth scheduling without eBPF"
	@echo "  compare      - Compare performance with different thread counts"
	@echo "  stress       - Stress test with multiple concurrent processes"
	@echo "  run-scheduler- Run eBPF scheduler (requires root)"
	@echo "  clean        - Clean compiled files"
	@echo "  help         - Show this help"

.PHONY: all minimal simple complex clean install-deps load load-simple load-complex test info help emergency