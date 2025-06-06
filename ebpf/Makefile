# Makefile for CXL-aware scheduler

CLANG ?= clang
LLC ?= llc
BPFTOOL ?= bpftool

LIBBPF_DIR = /usr/include
VMLINUX_H = vmlinux.h

# BPF program
BPF_PROG = cxl_pmu.bpf.c
BPF_OBJ = cxl_pmu.bpf.o

# Userspace program
USER_PROG = cxl_sched.c
USER_BIN = cxl_sched

CFLAGS = -O2 -g -Wall -Wextra
BPF_CFLAGS = -O2 -g -Wall -target bpf -D__TARGET_ARCH_x86

LIBS = -lbpf

.PHONY: all clean vmlinux

all: $(BPF_OBJ) $(USER_BIN)

$(VMLINUX_H):
	@if [ ! -f $(VMLINUX_H) ]; then \
		echo "Generating $(VMLINUX_H)..."; \
		$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $(VMLINUX_H); \
	fi

$(BPF_OBJ): $(BPF_PROG) $(VMLINUX_H)
	$(CLANG) $(BPF_CFLAGS) -I$(LIBBPF_DIR) -I. -c $(BPF_PROG) -o $(BPF_OBJ)

$(USER_BIN): $(USER_PROG)
	$(CC) $(CFLAGS) -I$(LIBBPF_DIR) $(USER_PROG) -o $(USER_BIN) $(LIBS)

vmlinux: $(VMLINUX_H)

install: all
	@echo "To load the scheduler, run: sudo ./$(USER_BIN)"

clean:
	rm -f $(BPF_OBJ) $(USER_BIN) $(VMLINUX_H)

help:
	@echo "Available targets:"
	@echo "  all      - Build BPF program and userspace loader"
	@echo "  vmlinux  - Generate vmlinux.h header"
	@echo "  install  - Build and show installation instructions"
	@echo "  clean    - Remove built files"
	@echo "  help     - Show this help message"