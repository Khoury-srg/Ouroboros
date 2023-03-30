#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int msleep(unsigned int tms) { return usleep(tms * 1000); }

void func_1_s() {
    void *x = malloc(64);
    sleep(1);
    free(x);
}

void func_100_ms() {
    void *x = malloc(64);
    msleep(100);
    free(x);
}

void func_1_ms() {
    void *x = malloc(64);
    msleep(1);
    free(x);
}

void func_10_us() {
    void *x = malloc(64);
    usleep(10);
    free(x);
}

void func_fastest() {
    void *x = malloc(64);
    free(x);
}

void (*funcs[])() = {func_100_ms, func_1_ms, func_10_us, func_fastest};

int main(int argc, char **argv) {
    srand(time(0));
    int func_n = sizeof(funcs) / sizeof(void *);
    for (int i = 0; i < 1000; ++i) {
        int r = rand() % func_n;
        funcs[r]();
    }
    return 0;
}