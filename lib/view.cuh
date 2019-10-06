#ifndef VIEW_CUH__
#define VIEW_CUH__

#include <cuda_runtime.h>
#include <type_traits>

#include "helper_math.cuh"
#include "util.cuh"

namespace gpho {

namespace detail {

class SizedBase {
    int3 size_ = make_int3(0, 0, 0);

public:
    explicit SizedBase() = default;

    __host__ __device__
    explicit SizedBase(int3 size) noexcept :
        size_(size) {}

    __host__ __device__
    SizedBase(const SizedBase& other) noexcept :
        size_(other.size_) {}

    __host__ __device__
    SizedBase(SizedBase&& other) noexcept :
        size_(other.size_)
    {
        other.size_ = make_int3(0, 0, 0);
    }

    __host__ __device__
    SizedBase& operator=(const SizedBase& rhs) noexcept
    {
        if (this != &rhs) {
            size_ = rhs.size_;
        }
        return *this;
    }

    __host__ __device__
    SizedBase& operator=(SizedBase&& rhs) noexcept
    {
        if (this != &rhs) {
            size_ = rhs.size_;
            rhs.size_ = make_int3(0, 0, 0);
        }
        return *this;
    }

    __host__ __device__
    bool operator==(const SizedBase& rhs) const noexcept
    {
        return size_ == rhs.size_;
    }

    __host__ __device__
    bool operator!=(const SizedBase& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    __host__ __device__
    const int3& size() const noexcept
    {
        return size_;
    }

    __host__ __device__
    int numel() const noexcept
    {
        return prod(size_);
    }

    __host__ __device__
    void reshape(const int3 newSize)
    {
        if (prod(newSize) != prod(size_)) {
#ifdef __CUDA_ARCH__
            // TODO: Maybe better to return an error state of some kind
            // Device version
            return;
#else
            // Host version
            throw std::length_error("Reshape cannot alter number of elements");
#endif
        }
        size_ = newSize;
    }

    __host__ __device__
    void reshape(int nx, int ny, int nz)
    {
        reshape(make_int3(nx, ny, nz));
    }

    __host__ __device__
    size_t idx(int3 pos) const noexcept
    {
        return gpho::idx(pos, size());
    }

    __host__ __device__
    size_t idx(int x, int y, int z) const noexcept
    {
        return gpho::idx(x, y, z, size());
    }
};

template <class Ty>
class ViewBase : public SizedBase {
    Ty *data_ = nullptr;

public:
    using Type = Ty;
    using ConstView = ViewBase<const typename std::remove_const<Ty>::type>;

    explicit ViewBase() = default;

    __host__ __device__
    explicit ViewBase(Ty *data, int3 size) noexcept :
        SizedBase(size),
        data_(data) {}

    __host__ __device__
    explicit ViewBase(Ty *data, int nx, int ny, int nz) noexcept :
        ViewBase(data, make_int3(nx, ny, nz)) {}

    __host__ __device__
    ViewBase(const ViewBase& other) noexcept :
        SizedBase(other),
        data_(other.data_) {}

    template <class Ty2>
    __host__ __device__
    ViewBase(const ViewBase<Ty2>& other) noexcept :
        SizedBase(other),
        data_(other.data()) {}

    __host__ __device__
    ViewBase& operator=(const ViewBase& rhs) noexcept
    {
        if (this != &rhs) {
            SizedBase::operator=(rhs);
            data_ = rhs.data_;
        }
        return *this;
    }

    __host__ __device__
    bool operator==(const ViewBase& rhs) const noexcept
    {
        return SizedBase::operator==(rhs) && data_ == rhs.data_;
    }

    __host__ __device__
    bool operator!=(const ViewBase& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    __host__ __device__
    Ty *data() noexcept
    {
        return data_;
    }

    __host__ __device__
    const Ty *data() const noexcept
    {
        return data_;
    }

    __host__ __device__
    Ty& operator[](size_t idx)
    {
        return data_[idx];
    }

    __host__ __device__
    const Ty& operator[](size_t idx) const
    {
        return data_[idx];
    }

    __host__ __device__
    Ty& operator[](int3 idx3)
    {
        return data_[idx(idx3)];
    }

    __host__ __device__
    const Ty& operator[](int3 idx3) const
    {
        return data_[idx(idx3)];
    }
};

template <class DstTy, class SrcTy>
void cudaCopy(ViewBase<DstTy> dst, const ViewBase<SrcTy> src, cudaMemcpyKind copyKind)
{
    static_assert(std::is_same<typename std::remove_cv<DstTy>::type, typename std::remove_cv<SrcTy>::type>::value,
        "Underlying type for views must be the same");
    if (dst.numel() != src.numel()) {
        throw std::length_error("Source and destination must have same number of elements");
    }
    ensureCudaSuccess(cudaMemcpy(dst.data(), src.data(), dst.numel() * sizeof(DstTy), copyKind));
}

} // namespace detail

template <class Ty>
class DeviceView : public detail::ViewBase<Ty> {
public:
    using ConstView = DeviceView<const typename std::remove_const<Ty>::type>;

    using detail::ViewBase<Ty>::ViewBase; // Inherit constructors
    DeviceView() = default;
    DeviceView(const DeviceView&) = default;
    DeviceView& operator=(const DeviceView&) = default;
};

template <class Ty>
class HostView : public detail::ViewBase<Ty> {
public:
    using ConstView = HostView<const typename std::remove_const<Ty>::type>;

    using detail::ViewBase<Ty>::ViewBase; // Inherit constructors
    HostView() = default;
    HostView(const HostView&) = default;
    HostView& operator=(const HostView&) = default;
};

template <class Ty>
class PinnedView : public HostView<Ty> {
public:
    using ConstView = PinnedView<const typename std::remove_const<Ty>::type>;

    using HostView<Ty>::HostView; // Inherit constructors
    PinnedView() = default;
    PinnedView(const PinnedView&) = default;
    PinnedView& operator=(const PinnedView&) = default;
};

template <class DstTy, class SrcTy>
void transfer(DeviceView<DstTy> dst, const HostView<SrcTy> src)
{
    static_assert(std::is_same<typename std::remove_const<DstTy>::type,
        typename std::remove_const<SrcTy>::type>::value,
        "Destination and source must have same fundamental type");
    detail::cudaCopy(dst, src, cudaMemcpyHostToDevice);
}

template <class DstTy, class SrcTy>
void transfer(HostView<DstTy> dst, const DeviceView<SrcTy> src)
{
    static_assert(std::is_same<typename std::remove_const<DstTy>::type,
        typename std::remove_const<SrcTy>::type>::value,
        "Destination and source must have same fundamental type");
    detail::cudaCopy(dst, src, cudaMemcpyDeviceToHost);
}

template <class DstTy, class SrcTy>
void copy(HostView<DstTy>& dst, const HostView<SrcTy>& src)
{
    static_assert(std::is_same<typename std::remove_const<DstTy>::type,
        typename std::remove_const<SrcTy>::type>::value,
        "Destination and source must have same fundamental type");
    detail::cudaCopy(dst, src, cudaMemcpyHostToHost);
}

template <class DstTy, class SrcTy>
void copy(DeviceView<DstTy>& dst, const DeviceView<SrcTy>& src)
{
    static_assert(std::is_same<typename std::remove_const<DstTy>::type,
        typename std::remove_const<SrcTy>::type>::value,
        "Destination and source must have same fundamental type");
    detail::cudaCopy(dst, src, cudaMemcpyDeviceToDevice);
}

} // namespace gpho

#endif // VIEW_CUH__
