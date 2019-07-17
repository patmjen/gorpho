#include "blockindexiter.cuh"

namespace cbp {

// BlockIndex

int3 BlockIndex::blockSizeBorder() const
{
    return endIdxBorder - startIdxBorder;
}

int3 BlockIndex::blockSize() const
{
    return endIdx - startIdx;
}

int3 BlockIndex::startBorder() const
{
    return startIdx - startIdxBorder;
}

int3 BlockIndex::endBorder() const
{
    return endIdxBorder - endIdx;
}

int BlockIndex::numel() const
{
    const int3 bs = blockSizeBorder();
    return bs.x*bs.y*bs.z;
}

bool BlockIndex::operator==(const BlockIndex& rhs) const
{
    return startIdx == rhs.startIdx && endIdx == rhs.endIdx &&
        startIdxBorder == rhs.startIdxBorder && endIdxBorder == rhs.endIdxBorder;
}

bool BlockIndex::operator!=(const BlockIndex& rhs) const
{
    return !(*this == rhs);
}

// BlockIndexIterator

bool BlockIndexIterator::operator==(const BlockIndexIterator& rhs) const
{
    // Since numBlocks and maxBlkIdx are computed from the other values,
    // there is no reason to compare these
    return blockSize_ == rhs.blockSize_ && borderSize_ == rhs.borderSize_ && volSize_ == rhs.volSize_ &&
        linBlkIdx_ == rhs.linBlkIdx_;
}

bool BlockIndexIterator::operator!=(const BlockIndexIterator& rhs) const
{
    return !(*this == rhs);
}

bool BlockIndexIterator::operator<=(const BlockIndexIterator& rhs) const
{
    return linBlkIdx_ <= rhs.linBlkIdx_;
}

bool BlockIndexIterator::operator>=(const BlockIndexIterator& rhs) const
{
    return linBlkIdx_ >= rhs.linBlkIdx_;
}

bool BlockIndexIterator::operator<(const BlockIndexIterator& rhs) const
{
    return linBlkIdx_ < rhs.linBlkIdx_;
}

bool BlockIndexIterator::operator>(const BlockIndexIterator& rhs) const
{
    return linBlkIdx_ > rhs.linBlkIdx_;
}

BlockIndexIterator& BlockIndexIterator::operator++()
{
    if (linBlkIdx_ <= maxBlkIdx_) {
        linBlkIdx_++;
    }
    updateBlockIndex_();
    return *this;
}

BlockIndexIterator BlockIndexIterator::operator++(int)
{
    BlockIndexIterator out = *this;
    if (linBlkIdx_ <= maxBlkIdx_) {
        linBlkIdx_++;
    }
    updateBlockIndex_();
    return out;
}

BlockIndexIterator&BlockIndexIterator:: operator--()
{
    if (linBlkIdx_ > 0 ) {
        linBlkIdx_--;
    }
    updateBlockIndex_();
    return *this;
}

BlockIndexIterator BlockIndexIterator::operator--(int)
{
    BlockIndexIterator out = *this;
    if (linBlkIdx_ > 0) {
        linBlkIdx_--;
    }
    updateBlockIndex_();
    return out;
}

BlockIndexIterator BlockIndexIterator::operator+=(int n)
{
    linBlkIdx_ += n;
    if (linBlkIdx_ > maxBlkIdx_ + 1) {
        linBlkIdx_ = maxBlkIdx_ + 1;
    }
    updateBlockIndex_();
    return *this;
}

BlockIndexIterator BlockIndexIterator::operator-=(int n)
{
    linBlkIdx_ -= n;
    if (linBlkIdx_ < 0) {
        linBlkIdx_ = 0;
    }
    updateBlockIndex_();
    return *this;
}

int BlockIndexIterator::operator-(const BlockIndexIterator& rhs)
{
    return linBlkIdx_ - rhs.linBlkIdx_;
}

const BlockIndex& BlockIndexIterator::operator*() const
{
    return blockIndex();
}

const BlockIndex *BlockIndexIterator::operator->()
{
    return &(this->blkIdx_);
}

const BlockIndex BlockIndexIterator::operator[](const int i)
{
    return blockIndexAt(i);
}

int BlockIndexIterator::maxLinearIndex() const
{
    return maxBlkIdx_;
}

int3 BlockIndexIterator::numBlocks() const
{
    return numBlocks_;
}

int3 BlockIndexIterator::blockSize() const
{
    return blockSize_;
}

int3 BlockIndexIterator::volSize() const
{
    return volSize_;
}

int3 BlockIndexIterator::borderSize() const
{
    return borderSize_;
}

int BlockIndexIterator::linearIndex() const
{
    return linBlkIdx_;
}

BlockIndexIterator BlockIndexIterator::begin() const
{
    BlockIndexIterator out = *this;
    out.updateBlockIndex_(0);
    return out;
}

BlockIndexIterator BlockIndexIterator::end() const
{
    BlockIndexIterator out = *this;
    out.updateBlockIndex_(maxBlkIdx_ + 1);
    return out;
}

BlockIndex BlockIndexIterator::blockIndexAt(const int i) const
{
    return calcBlockIndex_(i);
}

const BlockIndex& BlockIndexIterator::blockIndex() const
{
    return blkIdx_;
}

BlockIndex BlockIndexIterator::calcBlockIndex_(const int i) const
{
    // TODO: Allow iterating over the axes in different order
    const int xi = i % numBlocks_.x;
    const int yi = (i / numBlocks_.x) % numBlocks_.y;
    const int zi = i / (numBlocks_.x * numBlocks_.y);

    BlockIndex out;
    out.startIdx = make_int3(xi, yi, zi) * blockSize_;
    out.endIdx = min(out.startIdx + blockSize_, volSize_);
    out.startIdxBorder = max(out.startIdx - borderSize_, make_int3(0));
    out.endIdxBorder = min(out.endIdx + borderSize_, volSize_);

    return out;
}

void BlockIndexIterator::updateBlockIndex_()
{
    blkIdx_ = calcBlockIndex_(linBlkIdx_);
}

void BlockIndexIterator::updateBlockIndex_(const int bi)
{
    linBlkIdx_ = bi;
    updateBlockIndex_();
}

BlockIndexIterator operator+(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out += n;
    return out;
}

BlockIndexIterator operator+(const int n, const BlockIndexIterator& it)
{
    BlockIndexIterator out = it;
    out += n;
    return out;
}

BlockIndexIterator operator-(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out -= n;
    return out;
}

BlockIndexIterator operator-(const int n, const BlockIndexIterator& it)
{
    BlockIndexIterator out = it.begin();
    out += (n - it.linearIndex());
    return out;
}

} // namespace cbp
