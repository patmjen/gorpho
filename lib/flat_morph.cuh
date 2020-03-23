#ifndef FLAT_MORPH_CUH__
#define FLAT_MORPH_CUH__

#include <limits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudablockproc.cuh"
#include "helper_math.cuh"

#include "volume.cuh"
#include "morph.cuh"
#include "misc.cuh"

namespace gpho {

namespace kernel {

template <MorphOp op, class Ty>
__global__ void flatDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const bool> strel)
{
    static_assert(op == MORPH_DILATE || op == MORPH_ERODE, "op must be MORPH_DILATE or MORPH_ERODE");
    const int3 pos = globalPos3d();

    // Make sure we are within array bounds
    if (pos < vol.size()) {
        // Precompute start- and endpoints
        const int3 rstart = pos - strel.size() / 2;
        const int3 start = max(make_int3(0, 0, 0), rstart);
        const int3 end = min(vol.size() - 1, pos + (strel.size() - 1) / 2);

        // Find and store value for this position
        Ty val;
        if (op == MORPH_DILATE) {
            val = std::numeric_limits<Ty>::lowest();
        } else if (op == MORPH_ERODE) {
            val = std::numeric_limits<Ty>::max();
        }
        for (int iz = start.z; iz <= end.z; iz++) {
            for (int iy = start.y; iy <= end.y; iy++) {
                for (int ix = start.x; ix <= end.x; ix++) {
                    const int3 vidx = make_int3(ix, iy, iz);
                    Ty newVal;
                    if (op == MORPH_DILATE) {
                        newVal = max(val, vol[vidx]);
                    } else if (op == MORPH_ERODE) {
                        newVal = min(val, vol[vidx]);
                    }
                    // Update value if strel is true at this position
                    const Ty s = zeroOrOne<Ty>(strel[vidx - rstart]);
                    val = s * newVal + (Ty(1) - s) * val;
                }
            }
        }
        res[pos] = val;
    }
}

} // namespace kernel

template <MorphOp op, class Ty>
void flatDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    static_assert(op == MORPH_DILATE || op == MORPH_ERODE, "op must be MORPH_DILATE or MORPH_ERODE");
    dim3 threads = dim3(8, 8, 8);
    dim3 blocks = gridBlocks(threads, vol.size());
    kernel::flatDilateErode<op><<<blocks, threads, 0, stream>>>(res, vol, strel);
}

template <MorphOp op, class Ty>
void flatDilateErode(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    static_assert(op == MORPH_DILATE || op == MORPH_ERODE, "op must be MORPH_DILATE or MORPH_ERODE");
    auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void *buf)
    {
        const int3 size = block.blockSizeBorder();
        DeviceView<const Ty> volBlk(volVec[0], size);
        DeviceView<Ty> resBlk(resVec[0], size);
        flatDilateErode<op>(resBlk, volBlk, strel, stream);
    };

    const int3 borderSize = strel.size() / 2;
    cbp::BlockIndexIterator blockIter(vol.size(), blockSize, borderSize);
    cbp::CbpResult bpres = cbp::blockProc(processBlock, vol.data(), res.data(), blockIter);
    ensureCudaSuccess(cudaDeviceSynchronize());
    if (bpres != cbp::CBP_SUCCESS) {
        // TODO: Better error message
        throw std::runtime_error("Error during block processing");
    }
}

template <MorphOp op, class Ty>
void flatDilateErode(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    static_assert(op == MORPH_DILATE || op == MORPH_ERODE, "op must be MORPH_DILATE or MORPH_ERODE");
    DeviceVolume<bool> dstrel = makeDeviceVolume<bool>(strel.size());
    transfer(dstrel.view(), strel);
    flatDilateErode<op>(res, vol, dstrel.constView(), blockSize);
}

template <class Ty>
void flatDilate(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    flatDilateErode<MORPH_DILATE>(res, vol, strel, stream);
}

template <class Ty>
void flatDilate(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatDilateErode<MORPH_DILATE>(res, vol, strel, blockSize);
}

template <class Ty>
void flatDilate(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatDilateErode<MORPH_DILATE>(res, vol, strel, blockSize);
}

template <class Ty>
void flatErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    flatDilateErode<MORPH_ERODE>(res, vol, strel, stream);
}

template <class Ty>
void flatErode(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatDilateErode<MORPH_ERODE>(res, vol, strel, blockSize);
}

template <class Ty>
void flatErode(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatDilateErode<MORPH_ERODE>(res, vol, strel, blockSize);
}

template <MorphOp op, class Ty>
void flatOpenClose(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol, 
    DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    static_assert(op == MORPH_OPEN || op == MORPH_CLOSE, "op must be MORPH_OPEN or MORPH_CLOSE");
    constexpr MorphOp op1 = op == MORPH_OPEN ? MORPH_ERODE : MORPH_DILATE;
    constexpr MorphOp op2 = op == MORPH_OPEN ? MORPH_DILATE : MORPH_ERODE;
    flatDilateErode<op1>(res, vol, strel, stream);
    // We use a copy here, since resBuffer might point to the same data as vol.
    // Also, this copy is takes very little time, so the overhead is negligible.
    cudaMemcpyAsync(resBuffer.data(), res.data(), res.numel() * sizeof(Ty), cudaMemcpyDeviceToDevice, stream);
    flatDilateErode<op2, Ty>(res, resBuffer, strel, stream);
}

template <MorphOp op, class Ty>
void flatOpenClose(DeviceView<Ty> res, DeviceView<Ty> vol, DeviceView<const bool> strel,
    cudaStream_t stream = 0)
{
    static_assert(op == MORPH_OPEN || op == MORPH_CLOSE, "op must be MORPH_OPEN or MORPH_CLOSE");
    flatOpenClose<op, Ty>(res, vol, vol, strel, stream);
}

template <MorphOp op, class Ty>
void flatOpenClose(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    static_assert(op == MORPH_OPEN || op == MORPH_CLOSE, "op must be MORPH_OPEN or MORPH_CLOSE");
    auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void* buf)
    {
        const int3 size = block.blockSizeBorder();
        DeviceView<Ty> volBlk(volVec[0], size);
        DeviceView<Ty> resBlk(resVec[0], size);
        // Since the vol block will be overwritten next iteration, we use it as the result buffer.
        flatOpenClose<op, Ty>(resBlk, volBlk, strel, stream);
    };

    const int3 borderSize = 2 * (strel.size() / 2); // Need double border as we do two operations
    cbp::BlockIndexIterator blockIter(vol.size(), blockSize, borderSize);
    cbp::CbpResult bpres = cbp::blockProc(processBlock, vol.data(), res.data(), blockIter);
    ensureCudaSuccess(cudaDeviceSynchronize());
    if (bpres != cbp::CBP_SUCCESS) {
        // TODO: Better error message
        throw std::runtime_error("Error during block processing");
    }
}

template <MorphOp op, class Ty>
void flatOpenClose(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    static_assert(op == MORPH_OPEN || op == MORPH_CLOSE, "op must be MORPH_OPEN or MORPH_CLOSE");
    DeviceVolume<bool> dstrel = makeDeviceVolume<bool>(strel.size());
    transfer(dstrel.view(), strel);
    flatOpenClose<op>(res, vol, dstrel.constView(), blockSize);
}

template <class Ty>
void flatOpen(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol,
    DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    flatOpenClose<MORPH_OPEN>(res, resBuffer, vol, strel, stream);
}

template <class Ty>
void flatOpen(DeviceView<Ty> res, DeviceView<Ty> vol, DeviceView<const bool> strel,
    cudaStream_t stream = 0)
{
    flatOpenClose<MORPH_OPEN>(res, vol, strel, stream);
}

template <class Ty>
void flatOpen(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatOpenClose<MORPH_OPEN>(res, vol, strel, blockSize);
}

template <class Ty>
void flatOpen(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatOpenClose<MORPH_OPEN>(res, vol, strel, blockSize);
}

template <class Ty>
void flatClose(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol,
    DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    flatOpenClose<MORPH_CLOSE>(res, resBuffer, vol, strel, stream);
}

template <class Ty>
void flatClose(DeviceView<Ty> res, DeviceView<Ty> vol, DeviceView<const bool> strel,
    cudaStream_t stream = 0)
{
    flatOpenClose<MORPH_CLOSE>(res, vol, strel, stream);
}

template <class Ty>
void flatClose(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatOpenClose<MORPH_CLOSE>(res, vol, strel, blockSize);
}

template <class Ty>
void flatClose(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatOpenClose<MORPH_CLOSE>(res, vol, strel, blockSize);
}

template <class Ty>
void flatTophat(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol,
    DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    flatOpen(res, resBuffer, vol, strel, stream);
    elemWiseOp<MATH_SUB, Ty>(res, vol, res, stream);
}

template <class Ty>
void flatBothat(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol,
    DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    flatClose(res, resBuffer, vol, strel, stream);
    elemWiseOp<MATH_SUB, Ty>(res, res, vol, stream);
}

template <MorphOp op, class Ty>
void flatTophatBothat(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol,
    DeviceView<const bool> strel, cudaStream_t stream = 0)
{
    static_assert(op == MORPH_TOPHAT || op == MORPH_BOTHAT, "op must be MORPH_TOPHAT or MORPH_BOTHAT");
    if (op == MORPH_TOPHAT) {
        flatTophat(res, resBuffer, vol, strel, stream);
    } else {
        flatBothat(res, resBuffer, vol, strel, stream);
    }
}

template <MorphOp op, class Ty>
void flatTophatBothat(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    static_assert(op == MORPH_TOPHAT || op == MORPH_BOTHAT, "op must be MORPH_TOPHAT or MORPH_BOTHAT");
    auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void* buf)
    {
        const int3 size = block.blockSizeBorder();
        DeviceView<Ty> volBlk(volVec[0], size);
        DeviceView<Ty> resBlk(resVec[0], size);
        DeviceView<Ty> resBufBlk(static_cast<Ty *>(buf), size);
        // Since the vol block will be overwritten next iteration, we use it as the result buffer.
        flatTophatBothat<op, Ty>(resBlk, resBufBlk, volBlk, strel, stream);
    };

    const int3 borderSize = 2 * (strel.size() / 2); // Need double border as we do two operations
    size_t tmpSize = prod(blockSize + 2 * borderSize) * sizeof(Ty); // Need an extra block as result buffer
    cbp::BlockIndexIterator blockIter(vol.size(), blockSize, borderSize);
    cbp::CbpResult bpres = cbp::blockProc(processBlock, vol.data(), res.data(), blockIter, tmpSize);
    ensureCudaSuccess(cudaDeviceSynchronize());
    if (bpres != cbp::CBP_SUCCESS) {
        // TODO: Better error message
        throw std::runtime_error("Error during block processing");
    }
}

template <MorphOp op, class Ty>
void flatTophatBothat(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    static_assert(op == MORPH_TOPHAT || op == MORPH_BOTHAT, "op must be MORPH_TOPHAT or MORPH_BOTHAT");
    DeviceVolume<bool> dstrel = makeDeviceVolume<bool>(strel.size());
    transfer(dstrel.view(), strel);
    flatTophatBothat<op>(res, vol, dstrel.constView(), blockSize);
}

template <class Ty>
void flatTophat(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatTophatBothat<MORPH_TOPHAT>(res, vol, strel, blockSize);
}

template <class Ty>
void flatBothat(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatTophatBothat<MORPH_BOTHAT>(res, vol, strel, blockSize);
}

template <class Ty>
void flatTophat(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatTophatBothat<MORPH_TOPHAT>(res, vol, strel, blockSize);
}

template <class Ty>
void flatBothat(HostView<Ty> res, HostView<const Ty> vol, HostView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    flatTophatBothat<MORPH_BOTHAT>(res, vol, strel, blockSize);
}

} // namespace gpho

#endif // FLAT_MORPH_CUH__