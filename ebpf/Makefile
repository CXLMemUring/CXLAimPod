# Makefile for CXL PMU eBPF scheduler

CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL ?= bpftool
CC ?= gcc

# Directories
LIBBPF_DIR = /usr/include
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
BPF_OBJ = $(BPF_SRC:.c=.o)
BPF_SIMPLE_OBJ = $(BPF_SIMPLE_SRC:.c=.o)
BPF_MINIMAL_OBJ = $(BPF_MINIMAL_SRC:.c=.o)

USER_SRC = cxl_sched.c
USER_BIN = cxl_sched

# Default target - build minimal version to avoid all issues
all: $(BPF_MINIMAL_OBJ) $(USER_BIN)

# Build complex version (may hit instruction limit)
complex: $(BPF_OBJ) $(USER_BIN)

# Build simple version (may still have loop issues)
simple: $(BPF_SIMPLE_OBJ) $(USER_BIN)

# Build minimal version (no loops, guaranteed to work)
minimal: $(BPF_MINIMAL_OBJ) $(USER_BIN)

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

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build minimal scheduler (default, no loops)"
	@echo "  minimal      - Build ultra-minimal scheduler (guaranteed to work)"
	@echo "  simple       - Build simple scheduler (may have loop issues)"
	@echo "  complex      - Build complex scheduler (instruction limit issues)"
	@echo "  load         - Build and load minimal scheduler (requires root)"
	@echo "  load-simple  - Build and load simple scheduler (requires root)"
	@echo "  load-complex - Build and load complex scheduler (requires root)"
	@echo "  test         - Run comprehensive tests (requires root)"
	@echo "  info         - Show eBPF program information"
	@echo "  emergency    - Create ultra-simple emergency scheduler"
	@echo "  clean        - Remove generated files"
	@echo "  install-deps - Install required dependencies"
	@echo "  help         - Show this help message"

.PHONY: all minimal simple complex clean install-deps load load-simple load-complex test info help emergency