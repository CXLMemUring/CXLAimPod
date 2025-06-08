#!/bin/bash

# Quick loading test for different scheduler versions

set +e  # Don't exit on errors

echo "=== eBPF Scheduler Loading Test ==="
echo "Testing which versions can bypass loader issues..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_scheduler() {
    local scheduler_file="$1"
    local scheduler_name="$2"
    
    echo -e "\n${YELLOW}Testing: ${scheduler_name}${NC}"
    echo "File: $scheduler_file"
    
    if [[ ! -f "$scheduler_file" ]]; then
        echo -e "${RED}✗ File not found${NC}"
        return 1
    fi
    
    echo "Size: $(stat -c%s $scheduler_file) bytes"
    
    # Test loading with timeout (3 seconds)
    timeout 3s ./cxl_sched "$scheduler_file" >/dev/null 2>test_error.log &
    local pid=$!
    
    sleep 1
    
    if kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}✓ Loaded successfully - no errors detected!${NC}"
        kill $pid 2>/dev/null || true
        wait $pid 2>/dev/null || true
        rm -f test_error.log
        return 0
    else
        echo -e "${RED}✗ Failed to load${NC}"
        if [[ -f test_error.log ]]; then
            echo "Error details:"
            cat test_error.log | head -5
        fi
        rm -f test_error.log
        return 1
    fi
}

echo "Note: Testing requires root privileges for actual loading"
echo "Non-root users will see permission errors but can still detect compilation issues"

# Test 1: Emergency scheduler (ultra-simple)
test_scheduler "emergency_scheduler.bpf.o" "Emergency Scheduler (Ultra-Simple)"
emergency_result=$?

# Test 2: Minimal scheduler  
test_scheduler "cxl_pmu_minimal.bpf.o" "Minimal Scheduler (No Loops)"
minimal_result=$?

# Test 3: Simple scheduler (if exists)
if [[ -f "cxl_pmu_simple.bpf.o" ]]; then
    test_scheduler "cxl_pmu_simple.bpf.o" "Simple Scheduler (May Have Issues)"
    simple_result=$?
else
    echo -e "\n${YELLOW}Simple scheduler not built - skipping${NC}"
    simple_result=2
fi

# Test 4: Complex scheduler (if exists)
if [[ -f "cxl_pmu.bpf.o" ]]; then
    test_scheduler "cxl_pmu.bpf.o" "Complex Scheduler (Instruction Limit)"
    complex_result=$?
else
    echo -e "\n${YELLOW}Complex scheduler not built - skipping${NC}"
    complex_result=2
fi

echo -e "\n=== Results Summary ==="

if [[ $emergency_result -eq 0 ]]; then
    echo -e "${GREEN}✓ Emergency Scheduler: WORKS${NC} - Use this if others fail"
else
    echo -e "${RED}✗ Emergency Scheduler: FAILED${NC}"
fi

if [[ $minimal_result -eq 0 ]]; then
    echo -e "${GREEN}✓ Minimal Scheduler: WORKS${NC} - Recommended for production"
else
    echo -e "${RED}✗ Minimal Scheduler: FAILED${NC}"
fi

if [[ $simple_result -eq 0 ]]; then
    echo -e "${GREEN}✓ Simple Scheduler: WORKS${NC} - Good balance of features"
elif [[ $simple_result -eq 2 ]]; then
    echo -e "${YELLOW}- Simple Scheduler: NOT BUILT${NC}"
else
    echo -e "${RED}✗ Simple Scheduler: FAILED${NC} - Likely has loop issues"
fi

if [[ $complex_result -eq 0 ]]; then
    echo -e "${GREEN}✓ Complex Scheduler: WORKS${NC} - Full features available"
elif [[ $complex_result -eq 2 ]]; then
    echo -e "${YELLOW}- Complex Scheduler: NOT BUILT${NC}"
else
    echo -e "${RED}✗ Complex Scheduler: FAILED${NC} - Instruction limit exceeded"
fi

echo -e "\n=== Recommendations ==="

if [[ $minimal_result -eq 0 ]]; then
    echo -e "${GREEN}🎯 RECOMMENDED: Use minimal scheduler${NC}"
    echo "   Command: sudo ./cxl_sched cxl_pmu_minimal.bpf.o"
    echo "   Features: VectorDB detection, basic priority adjustment"
elif [[ $emergency_result -eq 0 ]]; then
    echo -e "${YELLOW}🚨 FALLBACK: Use emergency scheduler${NC}"
    echo "   Command: sudo ./cxl_sched emergency_scheduler.bpf.o"
    echo "   Features: Basic scheduling only, no VectorDB optimization"
else
    echo -e "${RED}❌ CRITICAL: No schedulers work on this system${NC}"
    echo "   This may indicate:"
    echo "   - Missing sched_ext kernel support"
    echo "   - eBPF subsystem issues"
    echo "   - Kernel version incompatibility"
fi

echo -e "\n=== For VectorDB Benchmarks ==="
if [[ $minimal_result -eq 0 ]]; then
    echo "# Start optimized scheduler for VectorDB workloads"
    echo "sudo ./cxl_sched cxl_pmu_minimal.bpf.o &"
    echo ""
    echo "# Run VectorDB benchmark (should work with VTune now)"
    echo "python3 -m vectordb_bench.cli.vectordbbench vsag ..."
    echo ""
    echo "# VTune profiling should now work without interference"
    echo "sudo vtune -collect memory-access -- your_vectordb_workload"
elif [[ $emergency_result -eq 0 ]]; then
    echo "# Start basic scheduler (limited VectorDB optimization)"
    echo "sudo ./cxl_sched emergency_scheduler.bpf.o &"
    echo ""
    echo "# Run VectorDB benchmark"
    echo "python3 -m vectordb_bench.cli.vectordbbench vsag ..."
fi

echo -e "\n${GREEN}✓ Testing complete!${NC}" 