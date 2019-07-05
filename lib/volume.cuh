#ifndef VOLUME_CUH__
#define VOLUME_CUH__

#include <memory>
#include <type_traits>
#include <new>
#include <stdexcept>
#include <string>

#include "util.cuh"

namespace gpho {

template <class Ty>
class Volume {
	std::shared_ptr<Ty> data_ = nullptr;
	int3 size_ = make_int3(0, 0, 0);

public:
	explicit Volume() = default;

	__host__
	explicit Volume(const std::shared_ptr<Ty>& data, int3 size) noexcept :
		data_(data),
		size_(size) {}

	__host__
	explicit Volume(std::shared_ptr<Ty>&& data, int3 size) noexcept :
		data_(std::move(data)),
		size_(size) {}

	__host__
	Volume(const Volume& other) noexcept :
		data_(other.data_),
		size_(other.size_) {}

	__host__
	Volume(Volume&& other) noexcept :
		data_(std::move(other.data_)),
		size_(other.size_)
	{
		other.size_ = make_int3(0, 0, 0);
	}

	__host__
	Volume& operator=(const Volume& rhs) noexcept
	{
		if (this != &rhs) {
			data_ = rhs.data_;
			size_ = rhs.size_;
		}
		return *this;
	}

	__host__
	Volume& operator=(Volume&& rhs) noexcept
	{
		if (this != &rhs) {
			data_ = std::move(rhs.data_);
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

template <class Ty>
class DeviceVolume : public Volume<Ty> {
	// Since calling get() on the shared_ptr owned by the Volume class means we are calling a __host__ function,
	// we need to keep a raw non-owning pointer around
	Ty *data_ptr_ = nullptr;

public:
	explicit DeviceVolume() = default;

	__host__
	explicit DeviceVolume(Ty *data, int3 size) noexcept :
		Volume(std::shared_ptr<Ty>(data, cudaFree), size)
	{
		data_ptr_ = data;
	}

	__host__
	explicit DeviceVolume(Ty *data, int nx, int ny, int nz) noexcept :
		Volume(std::shared_ptr<Ty>(data, cudaFree), make_int3(nx, ny, nz))
	{
		data_ptr_ = data;
	}

	__host__
	explicit DeviceVolume(const std::shared_ptr<Ty>& data, int3 size) noexcept :
		Volume(data, size)
	{
		data_ptr_ = data.get();
	}

	__host__
	explicit DeviceVolume(const std::shared_ptr<Ty>& data, int nx, int ny, int nz) noexcept :
		Volume(data, make_int3(nx, ny, nz))
	{
		data_ptr_ = data.get();
	}

	__host__ 
	DeviceVolume(const DeviceVolume& other) :
		Volume(other)
	{
		// Make sure we call data function from base class, since it knows the actual pointer
		data_ptr_ = Volume::data();
	}

	__host__
	DeviceVolume(DeviceVolume&& other) :
		Volume(std::move(other))
	{
		// Make sure we call data function from base class, since it knows the actual pointer
		data_ptr_ = Volume::data();
	}

	__host__
	DeviceVolume& operator=(const DeviceVolume& rhs)
	{
		if (this != &rhs) {
			data_ptr_ = rhs.data();
			Volume::operator=(rhs);
		}
		return *this;
	}

	__host__
	DeviceVolume& operator=(DeviceVolume&& rhs)
	{
		if (this != &rhs) {
			data_ptr_ = rhs.data();
			Volume::operator=(std::move(rhs));
		}
		return *this;
	}

	__host__
	HostVolume<Ty> copyToHost() const
	{
		HostVolume<Ty> out = makeHostVolume<Ty>(size());
		copyToHost(out);
		return out;
	}

	__host__
	PinnedVolume<Ty> copyToPinned() const
	{
		PinnedVolume<Ty> out = makePinnedVolume<Ty>(size());
		copyToHost(out);
		return out;
	}

	__host__
	HostVolume<Ty>& copyToHost(HostVolume<Ty>& dst) const 
	{
		if (numel() != dst.numel()) {
			throw std::length_error("Must copy to host volume with same number of elements");
		}
		cudaError_t res = cudaMemcpy(dst.data(), data(), numel() * sizeof(Ty), cudaMemcpyDeviceToHost);
		if (res != cudaSuccess) {
			std::string msg = "Error while copying to host: ";
			msg += cudaGetErrorString(res);
			throw std::runtime_error(msg);
		}
		return dst;
	}

	__host__ __device__
	Ty *data() noexcept
	{
		return data_ptr_;
	}

	__host__ __device__
	const Ty *data() const noexcept
	{
		return data_ptr_;
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

template <class Ty>
class HostVolume : public Volume<Ty> {
public:
	explicit HostVolume() = default;

	template <class Del>
	__host__
	explicit HostVolume(Ty *data, int3 size, Del deleter = std::default_delete<Ty[]>()) noexcept :
		Volume(std::shared_ptr<Ty>(data, deleter), size) {}

	template <class Del>
	__host__
	explicit HostVolume(Ty *data, int nx, int ny, int nz, Del deleter = std::default_delete<Ty[]>()) noexcept :
		Volume(std::shared_ptr<Ty>(data, deleter), make_int3(nx, ny, nz)) {}

	__host__
	explicit HostVolume(const std::shared_ptr<Ty>& data, int3 size) noexcept :
		Volume(data, size) {}

	__host__
	explicit HostVolume(const std::shared_ptr<Ty>& data, int nx, int ny, int nz) noexcept :
		Volume(data, make_int3(nx, ny, nz)) {}

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
		cudaError_t res = cudaMemcpy(dst.data(), data(), numel() * sizeof(Ty), cudaMemcpyHostToDevice);
		if (res != cudaSuccess) {
			std::string msg = "Error while copying to device: ";
			msg += cudaGetErrorString(res);
			throw std::runtime_error(msg);
		}
		return dst;
	}
};

template <class Ty>
__host__
inline HostVolume<Ty> makeHostVolume(int3 size)
{
	Ty *hostMem = new Ty[numel()];
	return HostVolume<Ty>(hostMem, size);
}

template <class Ty>
__host__
inline HostVolume<Ty> makeHostVolume(int nx, int ny, int nz)
{
	return makeHostVolume<Ty>(make_int3(nx, ny, nz));
}

template <class Ty>
class PinnedVolume : public HostVolume<Ty> {
public:
	explicit PinnedVolume() = default;

	__host__
	explicit PinnedVolume(Ty *data, int3 size) noexcept :
		HostVolume(std::shared_ptr<Ty>(data, cudaFreeHost), size) {}

	__host__
	explicit PinnedVolume(Ty *data, int nx, int ny, int nz) noexcept :
		HostVolume(std::shared_ptr<Ty>(data, cudaFreeHost), make_int3(nx, ny, nz)) {}

	__host__
	explicit PinnedVolume(const std::shared_ptr<Ty>& data, int3 size) noexcept :
		HostVolume(data, size) {}

	__host__
	explicit PinnedVolume(const std::shared_ptr<Ty>& data, int nx, int ny, int nz) noexcept :
		HostVolume(data, make_int3(nx, ny, nz)) {}
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