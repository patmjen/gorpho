#ifndef MISC_CUH__
#define MISC_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudablockproc.cuh"
#include "helper_math.cuh"

#include "util.cuh"
#include "view.cuh"

namespace gpho {

enum MathOp {
    MATH_ADD,
    MATH_SUB,
    MATH_MUL,
    MATH_DIV
};

namespace kernel {

// TODO: Can these kernels be merged to one with a switch on a template MathOp switch without any perf. loss?

template <class Ty>
__global__ void elemWiseAdd(DeviceView<Ty> res, DeviceView<const Ty> a, DeviceView<const Ty> b)
{
    const size_t idx = res.idx(globalPos3d());
    if (idx < res.numel()) {
        res[idx] = a[idx] + b[idx];
    }
}

template <class Ty>
__global__ void elemWiseSub(DeviceView<Ty> res, DeviceView<const Ty> a, DeviceView<const Ty> b)
{
    const size_t idx = res.idx(globalPos3d());
    if (idx < res.numel()) {
        res[idx] = a[idx] - b[idx];
    }
}

template <class Ty>
__global__ void elemWiseMul(DeviceView<Ty> res, DeviceView<const Ty> a, DeviceView<const Ty> b)
{
    const size_t idx = res.idx(globalPos3d());
    if (idx < res.numel()) {
        res[idx] = a[idx] * b[idx];
    }
}

template <class Ty>
__global__ void elemWiseDiv(DeviceView<Ty> res, DeviceView<const Ty> a, DeviceView<const Ty> b)
{
    const size_t idx = res.idx(globalPos3d());
    if (idx < res.numel()) {
        res[idx] = a[idx] / b[idx];
    }
}

} // namespace kernel

template <MathOp op, class Ty>
void elemWiseOp(DeviceView<Ty> res, DeviceView<const Ty> a, DeviceView<const Ty> b, cudaStream_t stream = 0)
{
    size_t threads = 1024;
    size_t blocks = gridAxisBlocks(threads, res.numel());
    switch (op) {
    case MATH_ADD:
        kernel::elemWiseAdd<<<blocks, threads, 0, stream>>>(res, a, b);
        break;
    case MATH_SUB:
        kernel::elemWiseSub<<<blocks, threads, 0, stream>>>(res, a, b);
        break;
    case MATH_MUL:
        kernel::elemWiseMul<<<blocks, threads, 0, stream>>>(res, a, b);
        break;
    case MATH_DIV:
        kernel::elemWiseDiv<<<blocks, threads, 0, stream>>>(res, a, b);
        break;
    }
}

} // namespace gpho

#endif // MISC_CUH__