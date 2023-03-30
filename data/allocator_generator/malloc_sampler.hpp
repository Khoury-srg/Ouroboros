#define _GNU_SOURCE
#include <chrono>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <vector>
using namespace std;

inline int thread_rand() {
    thread_local int g_seed = time(0);
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

void print_stack(FILE *f) {
    fflush(f);
    void *callstack[128];
    int i, frames = backtrace(callstack, 128);
    backtrace_symbols_fd(callstack+3, frames, fileno(f));
}

uint64_t get_timestamp() {
    using namespace std::chrono;
    microseconds us =
        duration_cast<microseconds>(system_clock::now().time_since_epoch());
    return us.count();
}

template <int BUFFER_SIZE = 128> class lifetime_sampler {
  public:
    void *addresses[BUFFER_SIZE];
    uint64_t start_time[BUFFER_SIZE];
    int sample_rate; // out of 100
    FILE *output;

    lifetime_sampler(int sample_rate = 100, FILE *file_output = stdout)
        : sample_rate(sample_rate), output(file_output) {
        memset(addresses, 0, sizeof(addresses));
        memset(start_time, 0, sizeof(addresses));
    }

    void set_output(FILE *file_output) { output = file_output; }

    void sample_malloc(void *addr, size_t sz) {
        bool sample = (thread_rand() % 100) < sample_rate;
        if (!sample) {
            return;
        }
        for (uint i = 0; i < BUFFER_SIZE; i++) {
            if (addresses[i] == 0) {
                uint64_t alloc_time = get_timestamp();
                fprintf(output, "malloc %p %lu\n", addr, sz);
                print_stack(output);
                fprintf(output, "end\n");
                addresses[i] = addr;
                start_time[i] = alloc_time;
                return;
            }
        }
        fprintf(stderr, "Warning: sampler buffer full\n");
    }

    void sample_free(void *addr) {
        uint64_t free_time = get_timestamp();
        for (uint i = 0; i < BUFFER_SIZE; i++) {
            if (addresses[i] == addr) {
                fprintf(output, "free %p %llu\n", addr,
                        free_time - start_time[i]);
                addresses[i] = 0;
                start_time[i] = 0;
                return;
            }
        }
    }
};
lifetime_sampler<> sampler(100);

void *malloc(size_t sz) {
    void *(*libc_malloc)(size_t) =
        (void *(*)(size_t))dlsym(RTLD_NEXT, "malloc");
    void *addr = libc_malloc(sz);
    sampler.sample_malloc(addr, sz);
    return addr;
}

void free(void *p) {
    void (*libc_free)(void *) = (void (*)(void *))dlsym(RTLD_NEXT, "free");
    libc_free(p);
    sampler.sample_free(p);
}

void *operator new(size_t size) throw(std::bad_alloc) {
    void *p = malloc(size);
    return p;
}

void operator delete(void *p) throw() { free(p); }