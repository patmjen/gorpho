#ifndef GENERAL_MORPH_CUH__
#define GENERAL_MORPH_CUH__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudablockproc.cuh"
#include "cusatarit.cuh"
#include "helper_math.cuh"

#include "volume.cuh"
#include "morph.cuh"

namespace gpho {

namespace kernel {

template <MorphOp op, class Ty>
__global__ void genDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const Ty> strel)
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
            val = minusInfOrMin<Ty>();
        } else if (op == MORPH_ERODE) {
            val = infOrMax<Ty>();
        }
        for (int iz = start.z; iz <= end.z; iz++) {
            for (int iy = start.y; iy <= end.y; iy++) {
                for (int ix = start.x; ix <= end.x; ix++) {
                    const int3 vidx = make_int3(ix, iy, iz);
                    if (op == MORPH_DILATE) {
                        val = max(val, csa::satPlus(vol[vidx], strel[vidx - rstart]));
                    } else if (op == MORPH_ERODE) {
                        val = min(val, csa::satMinus(vol[vidx], strel[vidx - rstart]));
                    }
                }
            }
        }
        res[pos] = val;
    }
}

} // namespace kernel

template <MorphOp op, class Ty>
void genDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const Ty> strel, cudaStream_t stream = 0)
{
    dim3 threads = dim3(8, 8, 8);
    dim3 blocks = gridBlocks(threads, vol.size());
    kernel::genDilateErode<op><<<blocks, threads, 0, stream>>>(res, vol, strel);
}

template <MorphOp op, class Ty>
void genDilateErode(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const Ty> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void *buf)
    {
        const int3 size = block.blockSizeBorder();
        DeviceView<const Ty> volBlk(volVec[0], size);
        DeviceView<Ty> resBlk(resVec[0], size);
        genDilateErode<op>(resBlk, volBlk, strel, stream);
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
void genDilateErode(HostView<Ty> res, HostView<const Ty> vol, HostView<const Ty> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    DeviceVolume<Ty> dstrel = makeDeviceVolume<Ty>(strel.size());
    transfer(dstrel.view(), strel);
    genDilateErode<op>(res, vol, dstrel.constView(), blockSize);
}

template <class Ty>
void genDilate(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const Ty> strel, cudaStream_t stream = 0)
{
    genDilateErode<MORPH_DILATE>(res, vol, strel, stream);
}

template <class Ty>
void genDilate(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const Ty> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    genDilateErode<MORPH_DILATE>(res, vol, strel, blockSize);
}

template <class Ty>
void genDilate(HostView<Ty> res, HostView<const Ty> vol, HostView<const Ty> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    genDilateErode<MORPH_DILATE>(res, vol, strel, blockSize);
}

template <class Ty>
void genErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<const Ty> strel, cudaStream_t stream = 0)
{
    genDilateErode<MORPH_ERODE>(res, vol, strel, stream);
}

template <class Ty>
void genErode(HostView<Ty> res, HostView<const Ty> vol, DeviceView<const Ty> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    genDilateErode<MORPH_ERODE>(res, vol, strel, blockSize);
}

template <class Ty>
void genErode(HostView<Ty> res, HostView<const Ty> vol, HostView<const Ty> strel,
    int3 blockSize = make_int3(256, 256, 256))
{
    genDilateErode<MORPH_ERODE>(res, vol, strel, blockSize);
}

} // namespace gpho

#endif // GENERAL_MORPH_CUH__