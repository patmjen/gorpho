/** @file
 * Contains functions used for block processing on CUDA
*/

#ifndef CUDABLOCKPROC_CUH__
#define CUDABLOCKPROC_CUH__

#include <array>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <exception>
#include "blockindexiter.cuh"

/** @namespace cbp @brief CUDA block processing */
namespace cbp {

/** Physical memory location */
enum MemLocation {
    HOST_NORMAL = 0x01, /**< Normal host memory as returned by malloc. */
    HOST_PINNED = 0x02, /**< Pinned (page-locked) host memory as returned by cudaMallocHost. */
    DEVICE      = 0x10  /**< Device memory as returned by cudaMalloc. */
};

/** Block transfer direction */
enum BlockTransferKind {
    VOL_TO_BLOCK, /**< Transfer memory from volume to block. */
    BLOCK_TO_VOL  /**< Transfer memory from block to volume. */
};

/** CUDA block processing result */
enum CbpResult : int {
    CBP_SUCCESS               = 0x0, /**< Success */
    CBP_INVALID_VALUE         = 0x1, /**< Invalid value passed as input. */
    CBP_INVALID_MEM_LOC       = 0x2, /**< Invalid memory location. */
    CBP_HOST_MEM_ALLOC_FAIL   = 0x4, /**< Host memory allocation failed. */
    CBP_DEVICE_MEM_ALLOC_FAIL = 0x8  /**< Device memory allocation failed. */
};

/** Block process function with single input- and output volume.
 * @param func Callable with signature void(BlockIndex, cudaStream_t, std::vector<InTy>, std::vector<OutTy>, void *).
 * @param inVol Pointer to input volume.
 * @param outVol Pointer to output volume.
 * @param blockIter BlockIndexIterator describing the blocks.
 * @param tmpSize Size of temporary device buffer (default: 0).
 * @return Result of processing.
 */
template <class InTy, class OutTy, class Func>
inline CbpResult blockProc(Func func, InTy *inVol, OutTy *outVol,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize=0);

/*! Block process function with multiple input- and output volumes.
 * @param func Callable with signature void(BlockIndex, cudaStream_t, std::vector<InTy>, std::vector<OutTy>, void *).
 * @param inVols Container-like with pointers to input volumes.
 * @param outVols Container-like with pointers to output volumes.
 * @param blockIter BlockIndexIterator describing the blocks.
 * @param tmpSize Size of temporary device buffer (default: 0).
 * @return Result of processing.
 */
template <class InArr, class OutArr, class Func>
inline CbpResult blockProcMultiple(Func func, const InArr& inVols, const OutArr& outVols,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize=0);

/** Block process function with multiple input- and output volumes.
 * @param func Callable with signature void(BlockIndex, cudaStream_t, InDBlkArr, OutDBlkArr, void *).
 * @param inVols Container-like with pointers to input volumes.
 * @param outVols Container-like with pointers to output volumes.
 * @param inBlocks Container-like with pointers to pinned host staging buffer for input volumes.
 * @param outBlocks Container-like with pointers to pinned host staging buffer for output volumes.
 * @param d_inBlocks Container-like with pointers to device buffers for input blocks.
 * @param d_outBlocks Container-like with pointers to device buffers for output blocks.
 * @param blockIter BlockIndexIterator describing the blocks.
 * @param d_tmpMem Pointer to temporary device buffer (default: nullptr).
 * @return Result of processing.
 */
template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
inline CbpResult blockProcMultiple(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem=nullptr);

/**
 * Block process function with multiple input- and output volumes without validating input.
 * @param func Callable with signature void(BlockIndex, cudaStream_t, InDBlkArr, OutDBlkArr, void *).
 * @param inVols Container-like with pointers to input volumes.
 * @param outVols Container-like with pointers to output volumes.
 * @param inBlocks Container-like with pointers to pinned host staging buffer for input volumes.
 * @param outBlocks Container-like with pointers to pinned host staging buffer for output volumes.
 * @param d_inBlocks Container-like with pointers to device buffers for input blocks.
 * @param d_outBlocks Container-like with pointers to device buffers for output blocks.
 * @param blockIter BlockIndexIterator describing the blocks.
 * @param d_tmpMem Pointer to temporary device buffer (default: nullptr).
 * @return Result of processing.
 */
template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
inline CbpResult blockProcMultipleNoValidate(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem=nullptr);

/**
 * Get physical location of memory pointed to by ptr.
 * @param ptr Pointer to memory.
 * @return Physical location of memory pointed to by ptr.
 */
inline MemLocation getMemLocation(const void *ptr);

/**
 * Test location of memory pointed to by ptr.
 * @param ptr Pointer to memory.
 * @return True if loc is the physical location of memory pointed to by ptr.
 */
template <MemLocation loc>
inline bool memLocationIs(const void *ptr);

/**
 * Transfer memory from volume to block or reverse. All memory must be on the host.
 * @param vol Pointer to full volume.
 * @param block Pointer to block.
 * @param bi Block index giving the location of the block in vol.
 * @param volSize Size of full volume.
 * @param kind Whether to transfer from volume to block or block to volume.
 * @param stream Current CUDA stream.
 */
template <typename Ty>
inline void blockVolumeTransfer(Ty *vol, Ty *block, const BlockIndex& bi, int3 volSize,
    BlockTransferKind kind, cudaStream_t stream);

/**
 * Transfer memory from multiple volumes to multiple blocks or reverse.  All memory must be on the host.
 * @param volArray Container-like with pointers to volumes.
 * @param blockArray Container-like with pointers to blocks.
 * @param blkIdx Block index giving the location and size of the block in the full volumes.
 * @param volSize Size of full volume.
 * @param kind Whether to transfer from volume to block or block to volume.
 * @param stream Current CUDA stream.
 */
template <typename VolArr, typename BlkArr>
inline void blockVolumeTransferAll(const VolArr& volArray, const BlkArr& blockArray, const BlockIndex& blkIdx,
    int3 volSize, BlockTransferKind kind, cudaStream_t stream);

/**
 * Copy blocks to/from device.
 * @param dstArray Container-like with pointers to destination blocks.
 * @param srcArray Container-like with pointers to source blocks.
 * @param blkIdx Block index giving the location and size of the block in the full volumes.
 * @param kind Type of copy. See documentation for `cudaMemcpyAsync`.
 * @param stream Current CUDA stream.
 */
template <typename DstArr, typename SrcArr>
inline void hostDeviceTransferAll(const DstArr& dstArray, const SrcArr& srcArray, const BlockIndex& blkIdx,
    cudaMemcpyKind kind, cudaStream_t stream);

template <typename Ty>
inline CbpResult allocBlocks(std::vector<Ty *>& blocks, const size_t n, const MemLocation loc,
    const int3 blockSize, const int3 borderSize=make_int3(0)) noexcept;

} // namespace cbp

#include "cudablockproc.inl"

#endif // CUDABLOCKPROC_CUH__