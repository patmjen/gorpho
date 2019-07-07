#ifndef VOLUME_CUH__
#define VOLUME_CUH__

#include <memory>
#include <type_traits>
#include <new>
#include <stdexcept>
#include <string>

#include "util.cuh"
#include "view.cuh"

namespace gpho {

namespace detail {

template <class Ty>
class VolumeBase : public SizedBase {
	std::shared_ptr<Ty> data_ = nullptr;

public:
	using Type = Ty;

	explicit VolumeBase() = default;

	__host__
	VolumeBase(const VolumeBase& other) noexcept :
		SizedBase(other),
		data_(other.data_) {}

	__host__
	VolumeBase(VolumeBase&& other) noexcept :
		SizedBase(std::move(other)),
		data_(std::move(other.data_)) {}

	__host__
	VolumeBase& operator=(const VolumeBase& rhs) noexcept
	{
		if (this != &rhs) {
			SizedBase::operator=(rhs);
			data_ = rhs.data_;
		}
		return *this;
	}

	__host__
	VolumeBase& operator=(VolumeBase&& rhs) noexcept
	{
		if (this != &rhs) {
			SizedBase::operator=(std::move(rhs));
			data_ = std::move(rhs.data_);
		}
		return *this;
	}

	__host__
	long useCount() const noexcept
	{
		return data_.use_count();
	}

	__host__
	Ty *data() noexcept
	{
		return data_.get();
	}

	__host__
	const Ty *data() const noexcept
	{
		return data_.get();
	}	

protected:
	__host__
	explicit VolumeBase(const std::shared_ptr<Ty>& data, int3 size) noexcept :
		SizedBase(size),
		data_(data) {}

	__host__
	explicit VolumeBase(std::shared_ptr<Ty>&& data, int3 size) noexcept :
		SizedBase(size),
		data_(std::move(data)) {}

	__host__
	explicit VolumeBase(const std::shared_ptr<Ty>& data, int nx, int ny, int nz) noexcept :
		VolumeBase(data, make_int3(nx, ny, nz)) {}

	__host__
	explicit VolumeBase(std::shared_ptr<Ty>&& data, int nx, int ny, int nz) noexcept :
		VolumeBase(std::move(data), make_int3(nx, ny, nz)) {}

	void copyTo_(VolumeBase<Ty>& dst, cudaMemcpyKind copyKind) const
	{
		cudaError_t res = cudaMemcpy(dst.data(), data(), numel() * sizeof(Ty), copyKind);
		if (res != cudaSuccess) {
			std::string msg = "Error while copying: ";
			msg += cudaGetErrorString(res);
			throw std::runtime_error(msg);
		}
	}
};

} // namespace detail

// Forward decls
template <class Ty> class DeviceVolume;
template <class Ty> class HostVolume;
template <class Ty> class PinnedVolume;
template <class Ty> inline DeviceVolume<Ty> makeDeviceVolume(int3 size);
template <class Ty> inline DeviceVolume<Ty> makeDeviceVolume(int nx, int ny, int nz);
template <class Ty> inline HostVolume<Ty> makeHostVolume(int3 size);
template <class Ty> inline HostVolume<Ty> makeHostVolume(int nx, int ny, int nz);
template <class Ty> inline PinnedVolume<Ty> makePinnedVolume(int3 size);
template <class Ty> inline PinnedVolume<Ty> makePinnedVolume(int nx, int ny, int nz);

// We need this ugly enable_if soup (I think!), as C++ does not allow us to overload on return type.

template <class Vol>
std::enable_if<std::is_same<Vol, DeviceVolume<typename Vol::Type>>::value, Vol>::type makeVolume(int3 size)
{
	return makeDeviceVolume<typename Vol::Type>(size);
}

template <class Vol>
std::enable_if<std::is_same<Vol, HostVolume<typename Vol::Type>>::value, Vol>::type makeVolume(int3 size)
{
	return makeHostVolume<typename Vol::Type>(size);
}

template <class Vol>
std::enable_if<std::is_same<Vol, PinnedVolume<typename Vol::Type>>::value, Vol>::type makeVolume(int3 size)
{
	return makePinnedVolume<typename Vol::Type>(size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Ty>
class DeviceVolume : public detail::VolumeBase<Ty> {
public:
	using View = DeviceView<Ty>;
	using ConstView = DeviceView<Ty>::ConstView;

	explicit DeviceVolume() = default;

	__host__
	explicit DeviceVolume(Ty *data, int3 size) noexcept :
		VolumeBase(std::shared_ptr<Ty>(data, cudaFree), size) {}

	__host__
	explicit DeviceVolume(Ty *data, int nx, int ny, int nz) noexcept :
		DeviceVolume(data, make_int3(nx, ny, nz)) {}

	__host__ 
	DeviceVolume(const DeviceVolume& other) :
		VolumeBase(other) {}

	__host__
	DeviceVolume(DeviceVolume&& other) :
		VolumeBase(std::move(other)) {}

	__host__
	DeviceVolume& operator=(const DeviceVolume& other)
	{
		VolumeBase::operator=(other);
		return *this;
	}

	__host__
	DeviceVolume& operator=(DeviceVolume&& other)
	{
		VolumeBase::operator=(std::move(other));
		return *this;
	}

	///
	/// Perform deep copy to new DeviceVolume.
	__host__
	DeviceVolume<Ty> copy() const
	{
		DeviceVolume<Ty> out = makeDeviceVolume(size());
		copyTo_(out, cudaMemcpyDeviceToDevice);
		return out;
	}

	///
	/// Copy contents to new host volume.
	__host__
	HostVolume<Ty> copyToHost() const
	{
		HostVolume<Ty> out = makeHostVolume<Ty>(size());
		copyToHost(out);
		return out;
	}

	///
	/// Copy contents to new pinned host volume.
	__host__
	PinnedVolume<Ty> copyToPinned() const
	{
		PinnedVolume<Ty> out = makePinnedVolume<Ty>(size());
		copyToHost(out);
		return out;
	}

	///
	/// Copy contents to supplied (pinned or not pinned) host volume. 
	__host__
	HostVolume<Ty>& copyToHost(HostVolume<Ty>& dst) const 
	{
		if (numel() != dst.numel()) {
			throw std::length_error("Must copy to host volume with same number of elements");
		}
		copyTo_(dst, cudaMemcpyDeviceToHost);
		return dst;
	}

	///
	/// Return view of device volume
	View view() noexcept
	{
		return View(data(), size());
	}

	///
	/// Return const view of device volume
	ConstView view() const noexcept
	{
		return ConstView(data(), size());
	}
};

template <class Ty>
__host__
inline DeviceVolume<Ty> makeDeviceVolume(int3 size)
{
	Ty *deviceMem = nullptr;
	if (cudaMalloc(&deviceMem, prod(size) * sizeof(Ty)) != cudaSuccess) {
		// TODO: Maybe throw another exception which better describes the error
		throw std::bad_alloc();
	}
	return DeviceVolume<Ty>(deviceMem, size);
}

template <class Ty>
__host__
inline DeviceVolume<Ty> makeDeviceVolume(int nx, int ny, int nz)
{
	return makeDeviceVolume<Ty>(make_int3(nx, ny, nz));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Host
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Ty>
class HostVolume : public detail::VolumeBase<Ty> {
public:
	using View = HostView<Ty>;
	using ConstView = HostView<Ty>::ConstView;

	explicit HostVolume() = default;

	template <class Del>
	__host__
	explicit HostVolume(Ty *data, int3 size, Del deleter) noexcept :
		VolumeBase(std::shared_ptr<Ty>(data, deleter), size) {}

	__host__
	explicit HostVolume(Ty *data, int3 size) noexcept :
		HostVolume(data, size, std::default_delete<Ty[]>()) {}

	template <class Del>
	__host__
	explicit HostVolume(Ty *data, int nx, int ny, int nz, Del deleter) noexcept :
		HostVolume(data, make_int3(nx, ny, nz), deleter) {}

	__host__
	explicit HostVolume(Ty *data, int nx, int ny, int nz) noexcept :
		HostVolume(data, nx, ny, nz, std::default_delete<Ty[]>()) {}

	__host__
	HostVolume(const HostVolume& other) :
		VolumeBase(other) {}

	__host__
	HostVolume(HostVolume&& other) :
		VolumeBase(std::move(other)) {}

	__host__
	HostVolume& operator=(const HostVolume& other)
	{
		VolumeBase::operator=(other);
		return *this;
	}

	__host__
	HostVolume& operator=(HostVolume&& other)
	{
		VolumeBase::operator=(std::move(other));
		return *this;
	}

	__host__
	HostVolume<Ty> copy() const
	{
		HostVolume<Ty> out = makeHostVolume(size());
		copyTo_(out, cudaMemcpyHostToHost);
		return out;
	}

	__host__
	DeviceVolume<Ty> copyToDevice() const
	{
		DeviceVolume<Ty> out = makeDeviceVolume<Ty>(size());
		copyToDevice(out);
		return out;
	}

	__host__
	DeviceVolume<Ty>& copyToDevice(DeviceVolume<Ty>& dst) const {
		if (numel() != dst.numel()) {
			throw std::length_error("Must copy to DeviceVolume with same number of elements");
		}
		copyTo_(dst, cudaMemcpyHostToDevice);
		return dst;
	}

	///
	/// Return view of host volume
	View view() noexcept
	{
		return View(data(), size());
	}

	///
	/// Return const view of host volume
	ConstView view() const noexcept
	{
		return ConstView(data(), size());
	}
};

template <class Ty>
__host__
inline HostVolume<Ty> makeHostVolume(int3 size)
{
	Ty *hostMem = new Ty[prod(size)];
	return HostVolume<Ty>(hostMem, size);
}

template <class Ty>
__host__
inline HostVolume<Ty> makeHostVolume(int nx, int ny, int nz)
{
	return makeHostVolume<Ty>(make_int3(nx, ny, nz));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pinned
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Ty>
class PinnedVolume : public HostVolume<Ty> {
public:
	using View = PinnedView<Ty>;
	using ConstView = PinnedView<Ty>::ConstView;

	explicit PinnedVolume() = default;

	__host__
	explicit PinnedVolume(Ty *data, int3 size) noexcept :
		HostVolume(data, size, cudaFreeHost) {}

	__host__
	explicit PinnedVolume(Ty *data, int nx, int ny, int nz) noexcept :
		PinnedVolume(data, make_int3(nx, ny, nz), cudaFreeHost) {}

	__host__
	PinnedVolume(const PinnedVolume& other) :
		HostVolume(other) {}

	__host__
	PinnedVolume(PinnedVolume&& other) :
		HostVolume(std::move(other)) {}

	__host__
	PinnedVolume& operator=(const PinnedVolume& other)
	{
		HostVolume::operator=(other);
		return *this;
	}

	__host__
	PinnedVolume& operator=(PinnedVolume&& other)
	{
		HostVolume::operator=(std::move(other));
		return *this;
	}

	__host__
	PinnedVolume<Ty> copy() const
	{
		PinnedVolume<Ty> out = makePinnedVolume(size());
		copyTo_(out, cudaMemcpyHostToHost);
		return out;
	}

	///
	/// Return view of pinned volume
	View view() noexcept
	{
		return View(data(), size());
	}

	///
	/// Return const view of pinned volume
	ConstView view() const noexcept
	{
		return ConstView(data(), size());
	}
};

template <class Ty>
__host__
inline PinnedVolume<Ty> makePinnedVolume(int3 size)
{
	Ty *pinnedMem = nullptr;
	if (cudaMallocHost(&pinnedMem, prod(size) * sizeof(Ty)) != cudaSuccess) {
		// TODO: Maybe throw another exception which better describes the error
		throw std::bad_alloc();
	}
	return PinnedVolume<Ty>(pinnedMem, size);
}

template <class Ty>
__host__
inline PinnedVolume<Ty> makePinnedVolume(int nx, int ny, int nz)
{
	return makePinnedVolume<Ty>(make_int3(nx, ny, nz));
}

} // namespace gpho

#endif // VOLUME_CUH__