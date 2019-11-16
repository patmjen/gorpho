#ifndef FLAT_MORPH_CUH__
#define FLAT_MORPH_CUH__

#include <limits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudablockproc.cuh"
#include "helper_math.cuh"

#include "volume.cuh"
#include "morph.cuh"

namespace gpho {

namespace kernel {

template <MorphOp op, class Ty>
__global__ void flatDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const bool> strel)
{
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
    dim3 threads = dim3(8, 8, 8);
    dim3 blocks = gridBlocks(threads, vol.size());
    kernel::flatDilateErode<op><<<blocks, threads, 0, stream>>>(res, vol, strel);
}

template <MorphOp op, class Ty>
void flatDilateErode(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const bool> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
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

} // namespace gpho

#endif // FLAT_MORPH_CUH__