#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
__global__ void k(int* p) { p[threadIdx.x] = threadIdx.x; }
int main() {
    int nd = 0; hipError_t e = hipGetDeviceCount(&nd); std::printf("devices %d (%s)\n", nd, hipGetErrorString(e));
    for (int d = 0; d < nd; ++d) {
        hipDeviceProp_t pr{}; hipGetDeviceProperties(&pr, d);
        if (std::strncmp(pr.gcnArchName, "gfx1030", 7)) continue;
        std::printf("device %d %s\n", d, pr.name);
        e = hipSetDevice(d); std::printf("setdevice %s\n", hipGetErrorString(e));
        size_t fr = 0, tot = 0; e = hipMemGetInfo(&fr, &tot); std::printf("meminfo %s free %.2f GB of %.2f\n", hipGetErrorString(e), fr / 1e9, tot / 1e9);
        int* p = nullptr; e = hipMalloc(&p, 1 << 20); std::printf("malloc 1 MB %s\n", hipGetErrorString(e));
        if (p) { k<<<1, 64>>>(p); e = hipDeviceSynchronize(); std::printf("kernel %s\n", hipGetErrorString(e)); }
        void* h = nullptr; e = hipHostMalloc(&h, 256 << 20, 0); std::printf("pinned 256 MB %s\n", hipGetErrorString(e));
    }
}
