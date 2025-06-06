#!/usr/bin/env python3
"""
PCM Memory Bandwidth Parser
Parses Intel PCM memory.out files and extracts read/write bandwidth with timestamps.
"""

import re
import sys
import csv
from datetime import datetime, timedelta

def parse_pcm_memory_output(filename, output_csv=None):
    """
    Parse PCM memory output file and extract bandwidth data.
    
    Args:
        filename: Path to the PCM memory.out file
        output_csv: Optional CSV filename for output
    
    Returns:
        List of tuples: (timestamp, read_bandwidth_mb_s, write_bandwidth_mb_s)
    """
    
    # Patterns to match the system throughput lines
    read_pattern = r"System Read Throughput\(MB/s\):\s+(\d+\.\d+)"
    write_pattern = r"System Write Throughput\(MB/s\):\s+(\d+\.\d+)"
    
    results = []
    sample_interval = 1  # PCM updates every 1 second as shown in the output
    
    try:
        with open(filename, 'r') as file:
            content = file.read()
            
            # Find all read and write throughput values
            read_matches = re.findall(read_pattern, content)
            write_matches = re.findall(write_pattern, content)
            
            # Ensure we have matching read/write pairs
            if len(read_matches) != len(write_matches):
                print(f"Warning: Mismatched read ({len(read_matches)}) and write ({len(write_matches)}) samples")
                min_len = min(len(read_matches), len(write_matches))
                read_matches = read_matches[:min_len]
                write_matches = write_matches[:min_len]
            
            # Generate timestamps (assuming measurement starts at time 0)
            start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            for i, (read_bw, write_bw) in enumerate(zip(read_matches, write_matches)):
                timestamp = start_time + timedelta(seconds=i * sample_interval)
                results.append((timestamp, float(read_bw), float(write_bw)))
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error parsing file: {e}")
        return []
    
    # Output to CSV if specified
    if output_csv:
        try:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Read_Bandwidth_MB_s', 'Write_Bandwidth_MB_s', 'Total_Bandwidth_MB_s'])
                
                for timestamp, read_bw, write_bw in results:
                    total_bw = read_bw + write_bw
                    writer.writerow([timestamp.strftime('%H:%M:%S'), read_bw, write_bw, total_bw])
            
            print(f"Results written to {output_csv}")
        except Exception as e:
            print(f"Error writing CSV: {e}")
    
    return results

def print_summary(results):
    """Print summary statistics of the bandwidth data."""
    if not results:
        print("No data to summarize")
        return
    
    read_values = [r[1] for r in results]
    write_values = [r[2] for r in results]
    total_values = [r[1] + r[2] for r in results]
    
    print(f"\nSummary for {len(results)} samples:")
    print(f"Read Bandwidth (MB/s)  - Min: {min(read_values):.2f}, Max: {max(read_values):.2f}, Avg: {sum(read_values)/len(read_values):.2f}")
    print(f"Write Bandwidth (MB/s) - Min: {min(write_values):.2f}, Max: {max(write_values):.2f}, Avg: {sum(write_values)/len(write_values):.2f}")
    print(f"Total Bandwidth (MB/s) - Min: {min(total_values):.2f}, Max: {max(total_values):.2f}, Avg: {sum(total_values)/len(total_values):.2f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python pcm_memory_parser.py <memory.out> [output.csv]")
        print("Example: python pcm_memory_parser.py memory.out bandwidth_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Parse the file
    results = parse_pcm_memory_output(input_file, output_csv)
    
    if results:
        print(f"Parsed {len(results)} bandwidth measurements")
        print_summary(results)
        
        # Display first few and last few samples
        print(f"\nFirst 5 samples:")
        print("Time     | Read MB/s | Write MB/s | Total MB/s")
        print("-" * 50)
        for i, (timestamp, read_bw, write_bw) in enumerate(results[:5]):
            total_bw = read_bw + write_bw
            print(f"{timestamp.strftime('%H:%M:%S')} | {read_bw:9.2f} | {write_bw:10.2f} | {total_bw:10.2f}")
        
        if len(results) > 10:
            print("...")
            print("Last 5 samples:")
            print("Time     | Read MB/s | Write MB/s | Total MB/s")
            print("-" * 50)
            for timestamp, read_bw, write_bw in results[-5:]:
                total_bw = read_bw + write_bw
                print(f"{timestamp.strftime('%H:%M:%S')} | {read_bw:9.2f} | {write_bw:10.2f} | {total_bw:10.2f}")
    else:
        print("No data could be parsed from the file")

if __name__ == "__main__":
    main()