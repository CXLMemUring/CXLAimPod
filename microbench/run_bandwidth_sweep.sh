#!/bin/bash

# Configuration
EXECUTABLE="./double_bandwidth"
DURATION=60  # Duration for each test in seconds
STEP=0.05     # Step size for read ratio increment
OUTPUT_FILE="bandwidth_results.txt"

# Default parameters (you can modify these)
BUFFER_SIZE="10737418240"  # 10GB
BLOCK_SIZE="4096"         # 4KB
THREADS="20"
DEVICE_PATH=""            # Leave empty for memory test, or specify device path

# Clear previous results
echo "=== CXL Bandwidth Sweep Test ===" > $OUTPUT_FILE
echo "Date: $(date)" >> $OUTPUT_FILE
echo "Duration per test: ${DURATION} seconds" >> $OUTPUT_FILE
echo "Read ratio step: ${STEP}" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Function to run benchmark with specific read ratio
run_benchmark() {
    local read_ratio=$1
    echo "Running benchmark with read ratio: $read_ratio"
    echo "----------------------------------------" >> $OUTPUT_FILE
    echo "Read Ratio: $read_ratio" >> $OUTPUT_FILE
    echo "Timestamp: $(date)" >> $OUTPUT_FILE
    
    # Build command line arguments
    CMD="$EXECUTABLE -b $BUFFER_SIZE -s $BLOCK_SIZE -t $THREADS -d $DURATION -r $read_ratio"
    
    # Add device path if specified
    if [ ! -z "$DEVICE_PATH" ]; then
        CMD="$CMD -D $DEVICE_PATH"
    fi
    
    # Run the benchmark and capture output
    echo "Command: $CMD" >> $OUTPUT_FILE
    $CMD >> $OUTPUT_FILE 2>&1
    echo "" >> $OUTPUT_FILE
    
    echo "Completed read ratio: $read_ratio"
}

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable $EXECUTABLE not found!"
    echo "Please compile the double_bandwidth.cpp first:"
    echo "  g++ -std=c++11 -pthread -O2 -o double_bandwidth microbench/double_bandwidth.cpp"
    exit 1
fi

echo "Starting bandwidth sweep test..."
echo "Results will be saved to: $OUTPUT_FILE"
echo "Press Ctrl+C to stop the test"

# Main loop - sweep read ratio from 0 to 1
read_ratio=0.0
while (( $(echo "$read_ratio <= 1.0" | bc -l) )); do
    run_benchmark $read_ratio
    
    # Increment read ratio
    read_ratio=$(echo "$read_ratio + $STEP" | bc -l)
    
    # Sleep between tests (optional, since each test already runs for DURATION seconds)
    # sleep 1
done

echo "Sweep test completed!"
echo "Results saved to: $OUTPUT_FILE"

# Generate summary
echo ""
echo "=== SUMMARY ===" >> $OUTPUT_FILE
echo "Total tests run: $(echo "scale=0; 1.0 / $STEP + 1" | bc -l)" >> $OUTPUT_FILE
echo "Total time: approximately $(echo "scale=0; (1.0 / $STEP + 1) * $DURATION" | bc -l) seconds" >> $OUTPUT_FILE

echo "Summary appended to results file." 