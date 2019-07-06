#ifndef VIEW_CUH__
#define VIEW_CUH__

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

	explicit ViewBase() = default;

	__host__ __device__
	explicit ViewBase(Ty *data, int3 size) noexcept :
		SizedBase(size),
		data_(data) {}

	__host__ __device__
	ViewBase(const ViewBase& other) noexcept :
		SizedBase(other),
		data_(other.data_) {}

	template <class V>
	__host__ __device__
	ViewBase(const V& other) noexcept :
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
	Ty *data() noexcept
	{
		return data_;
	}

	__host__ __device__
	const Ty *data() const noexcept
	{
		return data_;
	}
};

} // namespace detail

template <class Ty>
class DeviceView : public detail::ViewBase<Ty> {
public:
	using detail::ViewBase<Ty>::ViewBase; // Inherit constructors
	DeviceView() = default;
	DeviceView(const DeviceView&) = default;
	DeviceView& operator=(const DeviceView&) = default;
};

template <class Ty>
class HostView : public detail::ViewBase<Ty> {
public:
	using detail::ViewBase<Ty>::ViewBase; // Inherit constructors
	HostView() = default;
	HostView(const HostView&) = default;
	HostView& operator=(const HostView&) = default;
};

template <class Ty>
class PinnedView : public detail::ViewBase<Ty> {
public:
	using detail::ViewBase<Ty>::ViewBase; // Inherit constructors
	PinnedView() = default;
	PinnedView(const PinnedView&) = default;
	PinnedView& operator=(const PinnedView&) = default;
};

} // namespace gpho

#endif // VIEW_CUH__