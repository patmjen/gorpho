#ifndef BLOCKINDEXITER_CUH__
#define BLOCKINDEXITER_CUH__

#include <iterator>
#include "helper_math.cuh"

namespace cbp {

namespace detail {

inline __host__ __device__ int gridLineBlocks(const int bs, const int siz)
{
    return siz/bs + ((siz % bs != 0) ? 1 : 0);
}

} // namespace detail

struct BlockIndex {
    int3 startIdx;
    int3 endIdx;
    int3 startIdxBorder;
    int3 endIdxBorder;

    explicit BlockIndex() = default;
    BlockIndex(int endIdx) :
        startIdx(make_int3(0)),
        endIdx(make_int3(endIdx)),
        startIdxBorder(make_int3(0)),
        endIdxBorder(make_int3(endIdx)) {}
    BlockIndex(int3 endIdx) :
        startIdx(make_int3(0)),
        endIdx(endIdx),
        startIdxBorder(make_int3(0)),
        endIdxBorder(endIdx) {}
    explicit BlockIndex(int3 startIdx, int3 endIdx) :
        startIdx(startIdx),
        endIdx(endIdx),
        startIdxBorder(startIdx),
        endIdxBorder(endIdx) {}
    explicit BlockIndex(int startIdx, int endIdx) :
        startIdx(make_int3(startIdx)),
        endIdx(make_int3(endIdx)),
        startIdxBorder(make_int3(startIdx)),
        endIdxBorder(make_int3(endIdx)) {}
    explicit BlockIndex(int3 startIdx, int3 endIdx, int3 startIdxBorder, int3 endIdxBorder) :
        startIdx(startIdx),
        endIdx(endIdx),
        startIdxBorder(startIdxBorder),
        endIdxBorder(endIdxBorder) {}
    explicit BlockIndex(int startIdx, int endIdx, int startIdxBorder, int endIdxBorder) :
        startIdx(make_int3(startIdx)),
        endIdx(make_int3(endIdx)),
        startIdxBorder(make_int3(startIdxBorder)),
        endIdxBorder(make_int3(endIdxBorder)) {}

    inline int3 blockSizeBorder() const;

    inline int3 blockSize() const;

    inline int3 startBorder() const;

    inline int3 endBorder() const;

    inline int numel() const;

    inline bool operator==(const BlockIndex& rhs) const;

    inline bool operator!=(const BlockIndex& rhs) const;
};

class BlockIndexIterator : public std::iterator<
    std::input_iterator_tag, BlockIndex, int, const BlockIndex*, const BlockIndex&> {
    // NOTE: In many ways, this iterator behaves exactly like a random access iterator, as it can jump to any
    // position in constant time. However, it *cannot* return references when dereferenced which remain valid
    // once the iterator has been moved, which disqualifies it as a ForwardIterator.
    int3 blockSize_;
    int3 borderSize_;
    int3 volSize_;

    int3 numBlocks_;
    int maxBlkIdx_;

    int linBlkIdx_;
    BlockIndex blkIdx_;

public:
    explicit BlockIndexIterator() = default;
    explicit BlockIndexIterator(const int3 volSize, const int3 blockSize,
        const int3 borderSize=make_int3(0)) :
        blockSize_(blockSize),
        borderSize_(borderSize),
        volSize_(volSize),
        numBlocks_(make_int3(
            cbp::detail::gridLineBlocks(blockSize.x, volSize.x),
            cbp::detail::gridLineBlocks(blockSize.y, volSize.y),
            cbp::detail::gridLineBlocks(blockSize.z, volSize.z))),
        maxBlkIdx_(-1),
        linBlkIdx_(0),
        blkIdx_()
    {
        maxBlkIdx_ = numBlocks_.x * numBlocks_.y * numBlocks_.z - 1;
        updateBlockIndex_();
    }

    inline bool operator==(const BlockIndexIterator& rhs) const;

    inline bool operator!=(const BlockIndexIterator& rhs) const;

    inline bool operator<=(const BlockIndexIterator& rhs) const;

    inline bool operator>=(const BlockIndexIterator& rhs) const;

    inline bool operator<(const BlockIndexIterator& rhs) const;

    inline bool operator>(const BlockIndexIterator& rhs) const;

    inline BlockIndexIterator& operator++();
    inline BlockIndexIterator operator++(int);

    inline BlockIndexIterator& operator--();
    inline BlockIndexIterator operator--(int);

    inline BlockIndexIterator operator+=(int n);

    inline BlockIndexIterator operator-=(int n);

    inline int operator-(const BlockIndexIterator& rhs);

    inline const BlockIndex& operator*() const;

    inline const BlockIndex *operator->();

    inline const BlockIndex operator[](const int i);

    inline int maxLinearIndex() const;

    inline int3 numBlocks() const;

    inline int3 blockSize() const;

    inline int3 volSize() const;

    inline int3 borderSize() const;

    inline int linearIndex() const;

    inline BlockIndexIterator begin() const;

    inline BlockIndexIterator end() const;

    inline BlockIndex blockIndexAt(const int i) const;

    inline const BlockIndex& blockIndex() const;

private:
    inline BlockIndex calcBlockIndex_(const int i) const;

    inline void updateBlockIndex_();

    inline void updateBlockIndex_(const int bi);
};

inline BlockIndexIterator operator+(const BlockIndexIterator& it, const int n);

inline BlockIndexIterator operator+(const int n, const BlockIndexIterator& it);

inline BlockIndexIterator operator-(const BlockIndexIterator& it, const int n);

inline BlockIndexIterator operator-(const int n, const BlockIndexIterator& it);

inline void swap(BlockIndexIterator& a, BlockIndexIterator& b);

} // namespace cbp

#include "blockindexiter.inl"

#endif // BLOCKINDEXITER_CUH__