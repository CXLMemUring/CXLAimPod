/**
 * double_bandwidth.cpp - CXL double bandwidth microbenchmark
 *
 * This microbenchmark measures the bandwidth of CXL by adjusting
 * the ratio of readers to writers, simulating bidirectional traffic.
 */

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

// 添加gettid函数，用于获取Linux线程ID
pid_t gettid() { return syscall(SYS_gettid); }

// CPU亲和性设置函数
bool set_cpu_affinity(int thread_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  // 根据线程ID的奇偶性分配CPU核心
  // 偶数ID（读线程）-> 偶数核心，奇数ID（写线程）-> 奇数核心
  int num_cpus = std::thread::hardware_concurrency();

  if (thread_id % 2 == 0) {
    // 偶数ID：分配到偶数核心
    for (int i = 0; i < num_cpus; i += 2) {
      CPU_SET(i, &cpuset);
    }
  } else {
    // 奇数ID：分配到奇数核心
    for (int i = 1; i < num_cpus; i += 2) {
      CPU_SET(i, &cpuset);
    }
  }

  int result = sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    std::cerr << "Warning: Failed to set CPU affinity for thread " << thread_id
              << " (TID: " << gettid() << "): " << strerror(errno) << std::endl;
    return false;
  }

  std::cout << "Thread " << thread_id << " (TID: " << gettid() << ") bound to "
            << (thread_id % 2 == 0 ? "even" : "odd") << " CPU cores"
            << std::endl;
  return true;
}

// Default parameters
constexpr size_t DEFAULT_BUFFER_SIZE = 1 * 1024 * 1024 * 1024UL; // 1GB
constexpr size_t DEFAULT_BLOCK_SIZE = 4096;                      // 4KB
constexpr int DEFAULT_DURATION = 10;                             // seconds
constexpr int DEFAULT_NUM_THREADS = 16;      // total threads
constexpr float DEFAULT_READ_RATIO = 0.5;   // 50% readers, 50% writers
constexpr size_t DEFAULT_MAX_BANDWIDTH = 0; // 0 means unlimited (MB/s)

struct ThreadStats {
  size_t bytes_processed = 0;
  size_t operations = 0;
  int thread_id = 0; // 新增：线程ID
};

// Rate limiter using token bucket algorithm
class RateLimiter {
private:
  std::mutex mutex_;
  size_t tokens_;
  size_t max_tokens_;
  size_t tokens_per_second_;
  std::chrono::steady_clock::time_point last_refill_;

public:
  RateLimiter(size_t max_bandwidth_mbps)
      : tokens_(0), max_tokens_(0), tokens_per_second_(0) {
    if (max_bandwidth_mbps > 0) {
      max_tokens_ = max_bandwidth_mbps * 1024 * 1024; // Convert MB/s to bytes/s
      tokens_per_second_ = max_tokens_;
      tokens_ = max_tokens_;
    }
    last_refill_ = std::chrono::steady_clock::now();
  }

  bool acquire(size_t bytes) {
    if (tokens_per_second_ == 0) {
      return true; // Unlimited
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Refill tokens based on elapsed time
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        now - last_refill_);

    if (elapsed.count() > 0) {
      size_t new_tokens = (tokens_per_second_ * elapsed.count()) / 1000000;
      tokens_ = std::min(tokens_ + new_tokens, max_tokens_);
      last_refill_ = now;
    }

    // Check if we have enough tokens
    if (tokens_ >= bytes) {
      tokens_ -= bytes;
      return true;
    }

    return false;
  }

  void wait_for_tokens(size_t bytes) {
    if (tokens_per_second_ == 0) {
      return; // Unlimited
    }

    while (!acquire(bytes)) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
};

struct BenchmarkConfig {
  size_t buffer_size = DEFAULT_BUFFER_SIZE;
  size_t block_size = DEFAULT_BLOCK_SIZE;
  int duration = DEFAULT_DURATION;
  int num_threads = DEFAULT_NUM_THREADS;
  float read_ratio = DEFAULT_READ_RATIO;
  size_t max_bandwidth_mbps = DEFAULT_MAX_BANDWIDTH;
  std::string device_path;
  bool use_mmap = false;
  bool is_cxl_mem = false;
};

void print_usage(const char *prog_name) {
  std::cerr
      << "Usage: " << prog_name << " [OPTIONS]\n"
      << "Options:\n"
      << "  -b, --buffer-size=SIZE    Total buffer size in bytes (default: "
         "1GB)\n"
      << "  -s, --block-size=SIZE     Block size for read/write operations "
         "(default: 4KB)\n"
      << "  -t, --threads=NUM         Total number of threads (default: 4)\n"
      << "  -d, --duration=SECONDS    Test duration in seconds (default: 10)\n"
      << "  -r, --read-ratio=RATIO    Ratio of readers (0.0-1.0, default: "
         "0.5)\n"
      << "  -B, --max-bandwidth=MB/s  Maximum total bandwidth in MB/s "
         "(0=unlimited, default: 0)\n"
      << "  -D, --device=PATH         CXL device path (if not specified, "
         "memory is used)\n"
      << "  -m, --mmap                Use mmap instead of read/write syscalls\n"
      << "  -c, --cxl-mem             Indicate the device is CXL memory\n"
      << "  -h, --help                Show this help message\n";
}

BenchmarkConfig parse_args(int argc, char *argv[]) {
  BenchmarkConfig config;

  static struct option long_options[] = {
      {"buffer-size", required_argument, 0, 'b'},
      {"block-size", required_argument, 0, 's'},
      {"threads", required_argument, 0, 't'},
      {"duration", required_argument, 0, 'd'},
      {"read-ratio", required_argument, 0, 'r'},
      {"max-bandwidth", required_argument, 0, 'B'},
      {"device", required_argument, 0, 'D'},
      {"mmap", no_argument, 0, 'm'},
      {"cxl-mem", no_argument, 0, 'c'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, "b:s:t:d:r:B:D:mch", long_options,
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
    case 'B':
      config.max_bandwidth_mbps = std::stoull(optarg);
      break;
    case 'D':
      config.device_path = optarg;
      break;
    case 'm':
      config.use_mmap = true;
      break;
    case 'c':
      config.is_cxl_mem = true;
      break;
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

void reader_thread(void *buffer, size_t buffer_size, size_t block_size,
                   std::atomic<bool> &stop_flag, ThreadStats &stats,
                   RateLimiter *rate_limiter, int thread_id) {
  std::vector<char> local_buffer(block_size);
  size_t offset = 0;

  // 保存线程ID用于调度和统计
  stats.thread_id = thread_id;

  // 设置CPU亲和性：偶数ID绑定到偶数核心
  set_cpu_affinity(thread_id);

  // 打印线程ID以便调试
  std::cout << "Reader thread started with ID: " << thread_id
            << " (TID: " << gettid() << ")" << std::endl;

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Wait for rate limiter tokens
    if (rate_limiter) {
      rate_limiter->wait_for_tokens(block_size);
    }

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

void writer_thread(void *buffer, size_t buffer_size, size_t block_size,
                   std::atomic<bool> &stop_flag, ThreadStats &stats,
                   RateLimiter *rate_limiter, int thread_id) {
  std::vector<char> local_buffer(block_size, 'W'); // Fill with 'W' for writers
  size_t offset = 0;

  // 保存线程ID用于调度和统计
  stats.thread_id = thread_id;

  // 设置CPU亲和性：奇数ID绑定到奇数核心
  set_cpu_affinity(thread_id);

  // 打印线程ID以便调试
  std::cout << "Writer thread started with ID: " << thread_id
            << " (TID: " << gettid() << ")" << std::endl;

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Wait for rate limiter tokens
    if (rate_limiter) {
      rate_limiter->wait_for_tokens(block_size);
    }

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

void device_reader_thread(int fd, size_t file_size, size_t block_size,
                          std::atomic<bool> &stop_flag, ThreadStats &stats,
                          RateLimiter *rate_limiter, int thread_id) {
  std::vector<char> local_buffer(block_size);
  size_t offset = 0;

  // 保存线程ID用于调度和统计
  stats.thread_id = thread_id;

  // 设置CPU亲和性：偶数ID绑定到偶数核心
  set_cpu_affinity(thread_id);

  // 打印线程ID以便调试
  std::cout << "Device reader thread started with ID: " << thread_id
            << " (TID: " << gettid() << ")" << std::endl;

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Wait for rate limiter tokens
    if (rate_limiter) {
      rate_limiter->wait_for_tokens(block_size);
    }

    // Seek to the position
    lseek(fd, offset, SEEK_SET);

    // Read block from the device
    ssize_t bytes_read = read(fd, local_buffer.data(), block_size);
    if (bytes_read <= 0) {
      std::cerr << "Error reading from device: " << strerror(errno)
                << std::endl;
      break;
    }

    // Move to next block with wrap-around
    offset = (offset + block_size) % (file_size - block_size);

    // Update statistics
    stats.bytes_processed += bytes_read;
    stats.operations++;
  }
}

void device_writer_thread(int fd, size_t file_size, size_t block_size,
                          std::atomic<bool> &stop_flag, ThreadStats &stats,
                          RateLimiter *rate_limiter, int thread_id) {
  std::vector<char> local_buffer(block_size, 'W');
  size_t offset = 0;

  // 保存线程ID用于调度和统计
  stats.thread_id = thread_id;

  // 设置CPU亲和性：奇数ID绑定到奇数核心
  set_cpu_affinity(thread_id);

  // 打印线程ID以便调试
  std::cout << "Device writer thread started with ID: " << thread_id
            << " (TID: " << gettid() << ")" << std::endl;

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Wait for rate limiter tokens
    if (rate_limiter) {
      rate_limiter->wait_for_tokens(block_size);
    }

    // Seek to the position
    lseek(fd, offset, SEEK_SET);

    // Write block to the device
    ssize_t bytes_written = write(fd, local_buffer.data(), block_size);
    if (bytes_written <= 0) {
      std::cerr << "Error writing to device: " << strerror(errno) << std::endl;
      break;
    }

    // Move to next block with wrap-around
    offset = (offset + block_size) % (file_size - block_size);

    // Update statistics
    stats.bytes_processed += bytes_written;
    stats.operations++;
  }
}

void mmap_reader_thread(void *mapped_area, size_t file_size, size_t block_size,
                        std::atomic<bool> &stop_flag, ThreadStats &stats,
                        RateLimiter *rate_limiter, int thread_id) {
  std::vector<char> local_buffer(block_size);
  size_t offset = 0;

  // 保存线程ID用于调度和统计
  stats.thread_id = thread_id;

  // 设置CPU亲和性：偶数ID绑定到偶数核心
  set_cpu_affinity(thread_id);

  // 打印线程ID以便调试
  std::cout << "MMAP reader thread started with ID: " << thread_id
            << " (TID: " << gettid() << ")" << std::endl;

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Wait for rate limiter tokens
    if (rate_limiter) {
      rate_limiter->wait_for_tokens(block_size);
    }

    // Read block from the mapped area
    std::memcpy(local_buffer.data(), static_cast<char *>(mapped_area) + offset,
                block_size);

    // Move to next block with wrap-around
    offset = (offset + block_size) % (file_size - block_size);

    // Update statistics
    stats.bytes_processed += block_size;
    stats.operations++;
  }
}

void mmap_writer_thread(void *mapped_area, size_t file_size, size_t block_size,
                        std::atomic<bool> &stop_flag, ThreadStats &stats,
                        RateLimiter *rate_limiter, int thread_id) {
  std::vector<char> local_buffer(block_size, 'W');
  size_t offset = 0;

  // 保存线程ID用于调度和统计
  stats.thread_id = thread_id;

  // 设置CPU亲和性：奇数ID绑定到奇数核心
  set_cpu_affinity(thread_id);

  // 打印线程ID以便调试
  std::cout << "MMAP writer thread started with ID: " << thread_id
            << " (TID: " << gettid() << ")" << std::endl;

  while (!stop_flag.load(std::memory_order_relaxed)) {
    // Wait for rate limiter tokens
    if (rate_limiter) {
      rate_limiter->wait_for_tokens(block_size);
    }

    // Write block to the mapped area
    std::memcpy(static_cast<char *>(mapped_area) + offset, local_buffer.data(),
                block_size);

    // Move to next block with wrap-around
    offset = (offset + block_size) % (file_size - block_size);

    // Update statistics
    stats.bytes_processed += block_size;
    stats.operations++;
  }
}

int main(int argc, char *argv[]) {
  BenchmarkConfig config = parse_args(argc, argv);

  // Calculate reader and writer thread counts
  int num_readers = static_cast<int>(config.num_threads * config.read_ratio);
  int num_writers = config.num_threads - num_readers;

  // Create rate limiters for read and write based on ratio
  std::unique_ptr<RateLimiter> read_limiter = nullptr;
  std::unique_ptr<RateLimiter> write_limiter = nullptr;

  if (config.max_bandwidth_mbps > 0) {
    size_t read_bandwidth =
        static_cast<size_t>(config.max_bandwidth_mbps * config.read_ratio);
    size_t write_bandwidth = config.max_bandwidth_mbps - read_bandwidth;

    if (num_readers > 0) {
      read_limiter = std::make_unique<RateLimiter>(read_bandwidth);
    }
    if (num_writers > 0) {
      write_limiter = std::make_unique<RateLimiter>(write_bandwidth);
    }
  }

  std::cout << "=== CXL Double Bandwidth Microbenchmark ===" << std::endl;
  std::cout << "Buffer size: " << config.buffer_size << " bytes" << std::endl;
  std::cout << "Block size: " << config.block_size << " bytes" << std::endl;
  std::cout << "Duration: " << config.duration << " seconds" << std::endl;
  std::cout << "Total threads: " << config.num_threads << std::endl;
  std::cout << "Read ratio: " << config.read_ratio << " (" << num_readers
            << " readers, " << num_writers << " writers)" << std::endl;
  std::cout << "Thread assignment: Even IDs for readers (CPU even cores), Odd "
               "IDs for writers (CPU odd cores)"
            << std::endl;

  if (config.max_bandwidth_mbps > 0) {
    size_t read_bandwidth =
        static_cast<size_t>(config.max_bandwidth_mbps * config.read_ratio);
    size_t write_bandwidth = config.max_bandwidth_mbps - read_bandwidth;
    std::cout << "Bandwidth limit: " << config.max_bandwidth_mbps
              << " MB/s total "
              << "(" << read_bandwidth << " MB/s read, " << write_bandwidth
              << " MB/s write)" << std::endl;
  } else {
    std::cout << "Bandwidth limit: Unlimited" << std::endl;
  }

  if (!config.device_path.empty()) {
    std::cout << "Device: " << config.device_path << std::endl;
    std::cout << "Access method: " << (config.use_mmap ? "mmap" : "read/write")
              << std::endl;
    std::cout << "Memory type: "
              << (config.is_cxl_mem ? "CXL memory" : "Regular device")
              << std::endl;
  } else {
    std::cout << "Using system memory for testing" << std::endl;
  }

  std::cout << "\nStarting benchmark..." << std::endl;

  // Prepare threads and resources
  std::vector<std::thread> threads;
  std::vector<ThreadStats> thread_stats(config.num_threads);
  std::atomic<bool> stop_flag(false);

  void *buffer = nullptr;
  int fd = -1;
  void *mapped_area = nullptr;

  try {
    if (config.device_path.empty()) {
      // Allocate memory buffer
      buffer = aligned_alloc(4096, config.buffer_size);
      if (!buffer) {
        std::cerr << "Failed to allocate memory: " << strerror(errno)
                  << std::endl;
        return 1;
      }

      // Initialize buffer with some data
      std::memset(buffer, 'A', config.buffer_size);

      // Create reader and writer threads for memory
      for (int i = 0; i < num_readers; i++) {
        int reader_id = i * 2; // 生成偶数ID: 0, 2, 4, ...
        threads.emplace_back(reader_thread, buffer, config.buffer_size,
                             config.block_size, std::ref(stop_flag),
                             std::ref(thread_stats[i]), read_limiter.get(),
                             reader_id);
      }

      // 为写线程分配奇数ID (1, 3, 5...)
      for (int i = 0; i < num_writers; i++) {
        int writer_id = i * 2 + 1; // 生成奇数ID: 1, 3, 5, ...
        threads.emplace_back(writer_thread, buffer, config.buffer_size,
                             config.block_size, std::ref(stop_flag),
                             std::ref(thread_stats[num_readers + i]),
                             write_limiter.get(), writer_id);
      }
    } else {
      // Open the device
      fd = open(config.device_path.c_str(), O_RDWR | O_DIRECT);
      if (fd < 0) {
        std::cerr << "Failed to open device " << config.device_path << ": "
                  << strerror(errno) << std::endl;
        return 1;
      }

      if (config.use_mmap) {
        // Use mmap for device access
        mapped_area = mmap(NULL, config.buffer_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
        if (mapped_area == MAP_FAILED) {
          std::cerr << "Failed to mmap device: " << strerror(errno)
                    << std::endl;
          close(fd);
          return 1;
        }

        // Create reader and writer threads for mmap
        // 为读线程分配偶数ID
        for (int i = 0; i < num_readers; i++) {
          int reader_id = i * 2; // 生成偶数ID: 0, 2, 4, ...
          threads.emplace_back(mmap_reader_thread, mapped_area,
                               config.buffer_size, config.block_size,
                               std::ref(stop_flag), std::ref(thread_stats[i]),
                               read_limiter.get(), reader_id);
        }

        // 为写线程分配奇数ID
        for (int i = 0; i < num_writers; i++) {
          int writer_id = i * 2 + 1; // 生成奇数ID: 1, 3, 5, ...
          threads.emplace_back(mmap_writer_thread, mapped_area,
                               config.buffer_size, config.block_size,
                               std::ref(stop_flag),
                               std::ref(thread_stats[num_readers + i]),
                               write_limiter.get(), writer_id);
        }
      } else {
        // Use read/write for device access
        // 为读线程分配偶数ID
        for (int i = 0; i < num_readers; i++) {
          int reader_id = i * 2; // 生成偶数ID: 0, 2, 4, ...
          threads.emplace_back(device_reader_thread, fd, config.buffer_size,
                               config.block_size, std::ref(stop_flag),
                               std::ref(thread_stats[i]), read_limiter.get(),
                               reader_id);
        }

        // 为写线程分配奇数ID
        for (int i = 0; i < num_writers; i++) {
          int writer_id = i * 2 + 1; // 生成奇数ID: 1, 3, 5, ...
          threads.emplace_back(device_writer_thread, fd, config.buffer_size,
                               config.block_size, std::ref(stop_flag),
                               std::ref(thread_stats[num_readers + i]),
                               write_limiter.get(), writer_id);
        }
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

    // Calculate total stats
    double elapsed_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    size_t total_read_bytes = 0;
    size_t total_read_ops = 0;
    size_t total_write_bytes = 0;
    size_t total_write_ops = 0;

    for (int i = 0; i < num_readers; i++) {
      total_read_bytes += thread_stats[i].bytes_processed;
      total_read_ops += thread_stats[i].operations;
    }

    for (int i = 0; i < num_writers; i++) {
      total_write_bytes += thread_stats[num_readers + i].bytes_processed;
      total_write_ops += thread_stats[num_readers + i].operations;
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

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }

  // Clean up resources
  if (mapped_area && mapped_area != MAP_FAILED) {
    munmap(mapped_area, config.buffer_size);
  }

  if (fd >= 0) {
    close(fd);
  }

  if (buffer) {
    free(buffer);
  }

  return 0;
}
