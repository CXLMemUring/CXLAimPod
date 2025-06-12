/**
 * cxl_memory_test.cpp - CXL Memory Access Test Program
 *
 * This program tests CXL memory access using different methods:
 * 1. System memory allocation (CXL integrated as system RAM)
 * 2. Direct physical memory access via /dev/mem
 * 3. Multi-threaded bandwidth testing
 */

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <getopt.h>
#include <iostream>
#include <mutex>
#include <numa.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Default parameters
constexpr size_t DEFAULT_BUFFER_SIZE = 1 * 1024 * 1024 * 1024UL; // 1GB
constexpr size_t DEFAULT_BLOCK_SIZE = 4096;                      // 4KB
constexpr int DEFAULT_DURATION = 60;                             // seconds
constexpr int DEFAULT_NUM_THREADS = 10;   // total threads
constexpr float DEFAULT_READ_RATIO = 0.5; // 50% readers, 50% writers

struct ThreadStats {
  size_t bytes_processed = 0;
  size_t operations = 0;
  std::string operation_type;
  int thread_id = -1;
};

enum class MemoryMode {
  SYSTEM_RAM,      // Use system memory allocation
  PHYSICAL_ACCESS, // Use /dev/mem for physical address access
  NUMA_AWARE,      // NUMA-aware system memory
  CXL_INTERLEAVE,  // CXL memory interleave mode (physical access)
  CXL_NUMA,        // CXL memory via NUMA allocation
  CXL_MULTI        // Multiple CXL buffers on NUMA node
};

struct TestConfig {
  size_t buffer_size = DEFAULT_BUFFER_SIZE;
  size_t block_size = DEFAULT_BLOCK_SIZE;
  int duration = DEFAULT_DURATION;
  int num_threads = DEFAULT_NUM_THREADS;
  float read_ratio = DEFAULT_READ_RATIO;
  MemoryMode mode = MemoryMode::SYSTEM_RAM;
  uint64_t physical_addr = 0x4080000000ULL; // Default CXL region0 address
  bool use_numa = false;
  int numa_node = -1;
  bool enable_interleave = false;
  std::vector<int> cxl_nodes = {0, 1}; // Default CXL NUMA nodes
  int num_cxl_buffers = 2; // Number of CXL buffers for multi mode
  std::vector<uint64_t> cxl_physical_addrs = {0x2080000000ULL, 0x2a5c0000000ULL}; // CXL Window 0, Window 1 physical addresses
};

void print_usage(const char *prog_name) {
  std::cerr
      << "Usage: " << prog_name << " [OPTIONS]\n"
      << "CXL Memory Testing Tool\n\n"
      << "Options:\n"
      << "  -b, --buffer-size=SIZE    Buffer size in bytes (default: 1GB)\n"
      << "  -s, --block-size=SIZE     Block size for operations (default: "
         "4KB)\n"
      << "  -t, --threads=NUM         Number of threads (default: 10)\n"
      << "  -d, --duration=SECONDS    Test duration in seconds (default: 60)\n"
      << "  -r, --read-ratio=RATIO    Read ratio (0.0-1.0, default: 0.5)\n"
      << "  -m, --mode=MODE           Memory access mode:\n"
      << "                              system: System RAM allocation "
         "(default)\n"
      << "                              physical: Direct physical memory via "
         "/dev/mem\n"
      << "                              numa: NUMA-aware system memory\n"
      << "                              interleave: CXL memory interleave "
         "mode (physical access)\n"
      << "                              cxl: CXL memory via NUMA node\n"
      << "                              multi: Multiple CXL buffers on NUMA node\n"
      << "  -a, --address=ADDR        Physical address for physical mode "
         "(hex)\n"
      << "  -n, --numa-node=NODE      NUMA node for numa mode\n"
      << "  -i, --interleave          Enable interleave across CXL nodes\n"
      << "  -c, --cxl-nodes=NODES     CXL NUMA nodes (comma-separated, e.g., "
         "0,1)\n"
      << "  -p, --cxl-addrs=ADDRS     CXL physical addresses (comma-separated hex, e.g., "
         "0x2080000000,0x2a5c0000000)\n"
      << "  -h, --help                Show this help message\n\n"
      << "Examples:\n"
      << "  # System RAM test (CXL memory included in system RAM)\n"
      << "  " << prog_name << " -m system -t 16 -r 0.6 -d 30\n\n"
      << "  # Direct physical memory access to CXL region\n"
      << "  " << prog_name << " -m physical -a 0x4080000000 -t 8 -d 30\n\n"
      << "  # NUMA-aware test\n"
      << "  " << prog_name << " -m numa -n 1 -t 12 -r 0.7 -d 45\n\n"
      << "  # CXL interleave test across Window 0 and Window 1\n"
      << "  " << prog_name << " -m interleave -p 0x2080000000,0x2a5c0000000 -t 16 -r 0.6 -d 60\n\n"
      << "  # CXL memory test via NUMA node 2\n"
      << "  " << prog_name << " -m cxl -n 2 -t 16 -r 0.6 -d 60\n\n"
      << "  # Multiple CXL buffers on NUMA node 2 (simulates 2 devices)\n"
      << "  " << prog_name << " -m multi -n 2 -c 2 -t 16 -r 0.6 -d 60\n";
}

TestConfig parse_args(int argc, char *argv[]) {
  TestConfig config;

  static struct option long_options[] = {
      {"buffer-size", required_argument, 0, 'b'},
      {"block-size", required_argument, 0, 's'},
      {"threads", required_argument, 0, 't'},
      {"duration", required_argument, 0, 'd'},
      {"read-ratio", required_argument, 0, 'r'},
      {"mode", required_argument, 0, 'm'},
      {"address", required_argument, 0, 'a'},
      {"numa-node", required_argument, 0, 'n'},
      {"interleave", no_argument, 0, 'i'},
      {"cxl-nodes", required_argument, 0, 'c'},
      {"cxl-addrs", required_argument, 0, 'p'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "b:s:t:d:r:m:a:n:ic:p:h", long_options,
                            &option_index)) != -1) {
    switch (opt) {
    case 'b':
      config.buffer_size = std::stoull(optarg);
      break;
    case 's':
      config.block_size = std::stoull(optarg);
      break;
    case 't':
      config.num_threads = std::stoi(optarg);
      break;
    case 'd':
      config.duration = std::stoi(optarg);
      break;
    case 'r':
      config.read_ratio = std::stof(optarg);
      if (config.read_ratio < 0.0 || config.read_ratio > 1.0) {
        std::cerr << "Read ratio must be between 0.0 and 1.0\n";
        exit(1);
      }
      break;
    case 'm':
      if (std::string(optarg) == "system") {
        config.mode = MemoryMode::SYSTEM_RAM;
      } else if (std::string(optarg) == "physical") {
        config.mode = MemoryMode::PHYSICAL_ACCESS;
      } else if (std::string(optarg) == "numa") {
        config.mode = MemoryMode::NUMA_AWARE;
        config.use_numa = true;
      } else if (std::string(optarg) == "interleave") {
        config.mode = MemoryMode::CXL_INTERLEAVE;
        config.enable_interleave = true;
      } else if (std::string(optarg) == "cxl") {
        config.mode = MemoryMode::CXL_NUMA;
        config.use_numa = true;
        config.numa_node = 2; // Default to CXL NUMA node
      } else if (std::string(optarg) == "multi") {
        config.mode = MemoryMode::CXL_MULTI;
        config.use_numa = true;
        config.numa_node = 2; // Default to CXL NUMA node
        config.enable_interleave = true;
      } else {
        std::cerr
            << "Invalid mode. Use: system, physical, numa, interleave, cxl, or multi\n";
        exit(1);
      }
      break;
    case 'a':
      config.physical_addr = std::stoull(optarg, nullptr, 16);
      break;
    case 'n':
      config.numa_node = std::stoi(optarg);
      config.use_numa = true;
      break;
    case 'i':
      config.enable_interleave = true;
      break;
    case 'c': {
      // Parse comma-separated CXL nodes
      config.cxl_nodes.clear();
      std::string nodes_str(optarg);
      size_t pos = 0;
      while (pos < nodes_str.length()) {
        size_t comma_pos = nodes_str.find(',', pos);
        if (comma_pos == std::string::npos)
          comma_pos = nodes_str.length();
        int node = std::stoi(nodes_str.substr(pos, comma_pos - pos));
        config.cxl_nodes.push_back(node);
        pos = comma_pos + 1;
      }
      // Also update num_cxl_buffers for multi mode
      if (config.mode == MemoryMode::CXL_MULTI) {
        config.num_cxl_buffers = config.cxl_nodes.size();
      }
      break;
    }
    case 'p': {
      // Parse comma-separated CXL physical addresses
      config.cxl_physical_addrs.clear();
      std::string addrs_str(optarg);
      size_t pos = 0;
      while (pos < addrs_str.length()) {
        size_t comma_pos = addrs_str.find(',', pos);
        if (comma_pos == std::string::npos)
          comma_pos = addrs_str.length();
        uint64_t addr = std::stoull(addrs_str.substr(pos, comma_pos - pos), nullptr, 16);
        config.cxl_physical_addrs.push_back(addr);
        pos = comma_pos + 1;
      }
      break;
    }
    case 'h':
      print_usage(argv[0]);
      exit(0);
    default:
      print_usage(argv[0]);
      exit(1);
    }
  }

  return config;
}

void system_reader_thread(void *buffer, size_t buffer_size, size_t block_size,
                          std::atomic<bool> &stop_flag, ThreadStats &stats,
                          int thread_id) {
  std::vector<char> local_buffer(block_size);
  size_t offset = 0;

  stats.thread_id = thread_id;
  stats.operation_type = "read";

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Read block from the buffer
    std::memcpy(local_buffer.data(), static_cast<char *>(buffer) + offset,
                block_size);

    // Move to next block with wrap-around
    offset = (offset + block_size) % (buffer_size - block_size);

    // Update statistics
    stats.bytes_processed += block_size;
    stats.operations++;
  }
}

void system_writer_thread(void *buffer, size_t buffer_size, size_t block_size,
                          std::atomic<bool> &stop_flag, ThreadStats &stats,
                          int thread_id) {
  std::vector<char> local_buffer(block_size, 'W');
  size_t offset = 0;

  stats.thread_id = thread_id;
  stats.operation_type = "write";

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Write block to the buffer
    std::memcpy(static_cast<char *>(buffer) + offset, local_buffer.data(),
                block_size);

    // Move to next block with wrap-around
    offset = (offset + block_size) % (buffer_size - block_size);

    // Update statistics
    stats.bytes_processed += block_size;
    stats.operations++;
  }
}

// Interleave memory reader thread
void interleave_reader_thread(std::vector<void *> &buffers, size_t buffer_size,
                              size_t block_size, std::atomic<bool> &stop_flag,
                              ThreadStats &stats, int thread_id) {
  std::vector<char> local_buffer(block_size);
  size_t offset = 0;
  size_t buffer_idx = 0;

  stats.thread_id = thread_id;
  stats.operation_type = "read";

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Interleave between different CXL memory regions
    void *current_buffer = buffers[buffer_idx % buffers.size()];

    // Read block from the current buffer
    std::memcpy(local_buffer.data(),
                static_cast<char *>(current_buffer) + offset, block_size);

    // Move to next block and buffer
    offset = (offset + block_size) % (buffer_size - block_size);
    buffer_idx++;

    // Update statistics
    stats.bytes_processed += block_size;
    stats.operations++;
  }
}

// Interleave memory writer thread
void interleave_writer_thread(std::vector<void *> &buffers, size_t buffer_size,
                              size_t block_size, std::atomic<bool> &stop_flag,
                              ThreadStats &stats, int thread_id) {
  std::vector<char> local_buffer(block_size, 'W');
  size_t offset = 0;
  size_t buffer_idx = 0;

  stats.thread_id = thread_id;
  stats.operation_type = "write";

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Interleave between different CXL memory regions
    void *current_buffer = buffers[buffer_idx % buffers.size()];

    // Write block to the current buffer
    std::memcpy(static_cast<char *>(current_buffer) + offset,
                local_buffer.data(), block_size);

    // Move to next block and buffer
    offset = (offset + block_size) % (buffer_size - block_size);
    buffer_idx++;

    // Update statistics
    stats.bytes_processed += block_size;
    stats.operations++;
  }
}

void show_system_info() {
  std::cout << "\n=== System Information ===" << std::endl;

  // Show total system memory
  struct sysinfo si;
  if (sysinfo(&si) == 0) {
    double total_ram_gb =
        (double)si.totalram * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Total system RAM: " << total_ram_gb << " GB" << std::endl;
  }

  // Show CXL information if available
  std::cout << "\nCXL Information:" << std::endl;
  system("cat /sys/devices/platform/ACPI0017:00/root0/decoder0.0/region0/size "
         "2>/dev/null | "
         "awk '{printf \"CXL Region Size: %.2f GB\\n\", "
         "strtonum($0)/(1024^3)}' || echo 'CXL region info not available'");
  
  // Show CXL memory regions from /proc/iomem
  std::cout << "\nCXL Memory Regions from /proc/iomem:" << std::endl;
  system("grep -i cxl /proc/iomem 2>/dev/null || echo 'No CXL regions found in /proc/iomem'");
  
  // Show NUMA memory info
  std::cout << "\nNUMA Memory Information:" << std::endl;
  system("numactl --hardware 2>/dev/null | grep -E 'node.*size|node.*free' || echo 'numactl not available'");

  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  TestConfig config = parse_args(argc, argv);

  std::cout << "=== CXL Memory Test Program ===" << std::endl;
  show_system_info();

  // Calculate reader and writer thread counts
  int num_readers = static_cast<int>(config.num_threads * config.read_ratio);
  int num_writers = config.num_threads - num_readers;

  std::cout << "Test Configuration:" << std::endl;
  std::cout << "  Buffer size: " << config.buffer_size << " bytes" << std::endl;
  std::cout << "  Block size: " << config.block_size << " bytes" << std::endl;
  std::cout << "  Duration: " << config.duration << " seconds" << std::endl;
  std::cout << "  Total threads: " << config.num_threads << std::endl;
  std::cout << "  Read ratio: " << config.read_ratio << " (" << num_readers
            << " readers, " << num_writers << " writers)" << std::endl;

  std::string mode_str;
  switch (config.mode) {
  case MemoryMode::SYSTEM_RAM:
    mode_str = "System RAM allocation";
    break;
  case MemoryMode::PHYSICAL_ACCESS:
    mode_str = "Physical memory access via /dev/mem";
    break;
  case MemoryMode::NUMA_AWARE:
    mode_str = "NUMA-aware system memory";
    break;
  case MemoryMode::CXL_INTERLEAVE:
    mode_str = "CXL memory interleave mode (physical access)";
    break;
  case MemoryMode::CXL_NUMA:
    mode_str = "CXL memory via NUMA allocation";
    break;
  case MemoryMode::CXL_MULTI:
    mode_str = "Multiple CXL buffers on NUMA node";
    break;
  }
  std::cout << "  Memory mode: " << mode_str << std::endl;

  if (config.mode == MemoryMode::PHYSICAL_ACCESS) {
    std::cout << "  Physical address: 0x" << std::hex << config.physical_addr
              << std::dec << std::endl;
  }
  if (config.use_numa) {
    std::cout << "  NUMA node: " << config.numa_node << std::endl;
  }
  if (config.enable_interleave) {
    if (config.mode == MemoryMode::CXL_INTERLEAVE) {
      std::cout << "  CXL physical addresses for interleave: ";
      for (size_t i = 0; i < config.cxl_physical_addrs.size(); i++) {
        std::cout << "0x" << std::hex << config.cxl_physical_addrs[i] << std::dec;
        if (i < config.cxl_physical_addrs.size() - 1)
          std::cout << ",";
      }
      std::cout << std::endl;
    } else if (config.mode == MemoryMode::CXL_MULTI) {
      std::cout << "  Number of CXL buffers: " << config.num_cxl_buffers << std::endl;
    }
  }

  std::cout << "\nInitializing memory..." << std::endl;

  void *buffer = nullptr;
  std::vector<void *> interleave_buffers;
  int fd = -1;

  try {
    if (config.mode == MemoryMode::PHYSICAL_ACCESS) {
      // Use /dev/mem for direct physical access
      fd = open("/dev/mem", O_RDWR);
      if (fd < 0) {
        std::cerr << "Failed to open /dev/mem: " << strerror(errno)
                  << std::endl;
        std::cerr << "Note: This requires root privileges" << std::endl;
        return 1;
      }

      buffer = mmap(NULL, config.buffer_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, config.physical_addr);
      if (buffer == MAP_FAILED) {
        std::cerr << "Failed to mmap physical memory: " << strerror(errno)
                  << std::endl;
        close(fd);
        return 1;
      }
      std::cout << "  Mapped physical memory at 0x" << std::hex
                << config.physical_addr << std::dec << std::endl;
    } else if (config.mode == MemoryMode::CXL_INTERLEAVE) {
      // Use /dev/mem for direct physical access to multiple CXL devices
      fd = open("/dev/mem", O_RDWR);
      if (fd < 0) {
        std::cerr << "Failed to open /dev/mem: " << strerror(errno)
                  << std::endl;
        std::cerr << "Note: This requires root privileges" << std::endl;
        return 1;
      }

      // Map each CXL physical address to separate buffers
      for (size_t i = 0; i < config.cxl_physical_addrs.size(); i++) {
        uint64_t phys_addr = config.cxl_physical_addrs[i];
        
        std::cout << "  Attempting to map CXL device " << i << " at 0x" 
                  << std::hex << phys_addr << std::dec << "..." << std::endl;
        
        void *cxl_buffer = mmap(NULL, config.buffer_size, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, phys_addr);
        if (cxl_buffer == MAP_FAILED) {
          std::cerr << "Failed to mmap CXL memory at 0x" << std::hex << phys_addr 
                    << std::dec << ": " << strerror(errno) << std::endl;
          
          if (errno == EPERM) {
            std::cerr << "Permission denied. Possible solutions:" << std::endl;
            std::cerr << "1. Run as root: sudo " << argv[0] << " ..." << std::endl;
            std::cerr << "2. Check if physical address is correct" << std::endl;
            std::cerr << "3. Verify CXL memory is properly configured" << std::endl;
            std::cerr << "4. Try using system memory mode instead: -m system" << std::endl;
          } else if (errno == EINVAL) {
            std::cerr << "Invalid address. The physical address 0x" << std::hex 
                      << phys_addr << std::dec << " may not be valid." << std::endl;
            std::cerr << "Check /proc/iomem for correct CXL memory ranges" << std::endl;
          }
          
          // Clean up previously mapped buffers
          for (void *buf : interleave_buffers) {
            munmap(buf, config.buffer_size);
          }
          close(fd);
          return 1;
        }

        // Initialize buffer with device-specific pattern
        std::memset(cxl_buffer, 'A' + i, config.buffer_size);
        interleave_buffers.push_back(cxl_buffer);

        std::cout << "  Mapped CXL device " << i << " at physical address 0x" 
                  << std::hex << phys_addr << std::dec 
                  << " (" << config.buffer_size << " bytes)" << std::endl;
      }
      std::cout << "  Total CXL devices mapped: " << interleave_buffers.size()
                << std::endl;
    } else if (config.mode == MemoryMode::CXL_NUMA) {
      // Use NUMA allocation on CXL node
      buffer = numa_alloc_onnode(config.buffer_size, config.numa_node);
      if (!buffer) {
        std::cerr << "Failed to allocate memory on NUMA node " << config.numa_node
                  << ": " << strerror(errno) << std::endl;
        std::cerr << "Note: Make sure NUMA node " << config.numa_node 
                  << " (CXL memory) is available" << std::endl;
        return 1;
      }
      // Initialize buffer with some data
      std::memset(buffer, 'C', config.buffer_size);
      std::cout << "  Allocated " << config.buffer_size
                << " bytes on CXL NUMA node " << config.numa_node << std::endl;
    } else if (config.mode == MemoryMode::CXL_MULTI) {
      // Allocate multiple separate buffers on CXL NUMA node
      for (int i = 0; i < config.num_cxl_buffers; i++) {
        void *cxl_buffer = numa_alloc_onnode(config.buffer_size, config.numa_node);
        if (!cxl_buffer) {
          std::cerr << "Failed to allocate CXL buffer " << i << " on NUMA node " 
                    << config.numa_node << ": " << strerror(errno) << std::endl;
          // Clean up previously allocated buffers
          for (void *buf : interleave_buffers) {
            numa_free(buf, config.buffer_size);
          }
          return 1;
        }
        
        // Initialize buffer with device-specific pattern
        std::memset(cxl_buffer, 'M' + i, config.buffer_size);
        interleave_buffers.push_back(cxl_buffer);
        
        std::cout << "  Allocated CXL buffer " << i << " (" << config.buffer_size
                  << " bytes) on NUMA node " << config.numa_node << std::endl;
      }
      std::cout << "  Total CXL buffers allocated: " << interleave_buffers.size()
                << std::endl;
    } else {
      // Use system memory allocation
      buffer = aligned_alloc(4096, config.buffer_size);

      if (!buffer) {
        std::cerr << "Failed to allocate memory: " << strerror(errno)
                  << std::endl;
        return 1;
      }

      // Initialize buffer with some data
      std::memset(buffer, 'A', config.buffer_size);
      std::cout << "  Allocated " << config.buffer_size
                << " bytes of system memory" << std::endl;
    }

    std::cout << "\nStarting benchmark..." << std::endl;

    // Prepare threads and resources
    std::vector<std::thread> threads;
    std::vector<ThreadStats> thread_stats(config.num_threads);
    std::atomic<bool> stop_flag(false);

    // Create reader threads
    for (int i = 0; i < num_readers; i++) {
      if (config.mode == MemoryMode::CXL_INTERLEAVE || config.mode == MemoryMode::CXL_MULTI) {
        threads.emplace_back(interleave_reader_thread,
                             std::ref(interleave_buffers), config.buffer_size,
                             config.block_size, std::ref(stop_flag),
                             std::ref(thread_stats[i]), i);
      } else {
        threads.emplace_back(system_reader_thread, buffer, config.buffer_size,
                             config.block_size, std::ref(stop_flag),
                             std::ref(thread_stats[i]), i);
      }
    }

    // Create writer threads
    for (int i = 0; i < num_writers; i++) {
      if (config.mode == MemoryMode::CXL_INTERLEAVE || config.mode == MemoryMode::CXL_MULTI) {
        threads.emplace_back(
            interleave_writer_thread, std::ref(interleave_buffers),
            config.buffer_size, config.block_size, std::ref(stop_flag),
            std::ref(thread_stats[num_readers + i]), num_readers + i);
      } else {
        threads.emplace_back(system_writer_thread, buffer, config.buffer_size,
                             config.block_size, std::ref(stop_flag),
                             std::ref(thread_stats[num_readers + i]),
                             num_readers + i);
      }
    }

    // Run the benchmark for the specified duration
    auto start_time = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(config.duration));
    stop_flag.store(true, std::memory_order_relaxed);
    auto end_time = std::chrono::steady_clock::now();

    // Wait for all threads to finish
    for (auto &t : threads) {
      if (t.joinable()) {
        t.join();
      }
    }

    // Calculate results
    double elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    size_t total_read_bytes = 0;
    size_t total_read_ops = 0;
    size_t total_write_bytes = 0;
    size_t total_write_ops = 0;

    for (const auto &stats : thread_stats) {
      if (stats.operation_type == "read") {
        total_read_bytes += stats.bytes_processed;
        total_read_ops += stats.operations;
      } else if (stats.operation_type == "write") {
        total_write_bytes += stats.bytes_processed;
        total_write_ops += stats.operations;
      }
    }

    // Print results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Test duration: " << elapsed_seconds << " seconds"
              << std::endl;

    if (num_readers > 0) {
      double read_bandwidth_mbps =
          (total_read_bytes / (1024.0 * 1024.0)) / elapsed_seconds;
      double read_iops = total_read_ops / elapsed_seconds;
      std::cout << "Read bandwidth: " << read_bandwidth_mbps << " MB/s"
                << std::endl;
      std::cout << "Read IOPS: " << read_iops << " ops/s" << std::endl;
    }

    if (num_writers > 0) {
      double write_bandwidth_mbps =
          (total_write_bytes / (1024.0 * 1024.0)) / elapsed_seconds;
      double write_iops = total_write_ops / elapsed_seconds;
      std::cout << "Write bandwidth: " << write_bandwidth_mbps << " MB/s"
                << std::endl;
      std::cout << "Write IOPS: " << write_iops << " ops/s" << std::endl;
    }

    double total_bandwidth_mbps =
        ((total_read_bytes + total_write_bytes) / (1024.0 * 1024.0)) /
        elapsed_seconds;
    double total_iops = (total_read_ops + total_write_ops) / elapsed_seconds;
    std::cout << "Total bandwidth: " << total_bandwidth_mbps << " MB/s"
              << std::endl;
    std::cout << "Total IOPS: " << total_iops << " ops/s" << std::endl;

    // Performance analysis
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    std::cout << "Average per-thread bandwidth: "
              << total_bandwidth_mbps / config.num_threads << " MB/s"
              << std::endl;
    std::cout << "Memory efficiency: "
              << (total_bandwidth_mbps * 100.0) / (40000.0)
              << "% (assuming 40GB/s peak)" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  // Clean up resources
  if (config.mode == MemoryMode::CXL_INTERLEAVE) {
    // Clean up interleave buffers (physical memory mappings)
    for (void *buf : interleave_buffers) {
      if (buf && buf != MAP_FAILED) {
        munmap(buf, config.buffer_size);
      }
    }
  } else if (config.mode == MemoryMode::CXL_MULTI) {
    // Clean up multiple CXL buffers (NUMA allocations)
    for (void *buf : interleave_buffers) {
      if (buf) {
        numa_free(buf, config.buffer_size);
      }
    }
  } else if (buffer) {
    if (config.mode == MemoryMode::PHYSICAL_ACCESS) {
      if (buffer != MAP_FAILED) {
        munmap(buffer, config.buffer_size);
      }
    } else if (config.mode == MemoryMode::CXL_NUMA) {
      numa_free(buffer, config.buffer_size);
    } else {
      free(buffer);
    }
  }

  if (fd >= 0) {
    close(fd);
  }

  return 0;
}