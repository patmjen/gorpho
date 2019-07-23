#ifndef UTIL_CUH__
#define UTIL_CUH__

#include <limits>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

namespace gpho {

inline void ensureCudaSuccess(cudaError_t res)
{
	if (res != cudaSuccess) {
		std::string msg("CUDA error: ");
		msg += cudaGetErrorString(res);
		throw std::runtime_error(msg);
	}
}

template <class Ty> void nonDeleter(Ty *x) {};

__host__ __device__
inline int prod(const int3 a)
{
	return a.x * a.y * a.z;
}

__host__ __device__
inline size_t idx(int x, int y, int z, int3 size)
{
	return (size_t)x + (size_t)y * (size_t)size.x + (size_t)z * (size_t)size.x * (size_t)size.y;
}

__host__ __device__
inline size_t idx(const int3 pos, const int3 size)
{
	return idx(pos.x, pos.y, pos.z, size);
}

inline unsigned int gridAxisBlocks(unsigned int nthr, int len)
{
	return len / nthr + ((len % nthr != 0) ? 1 : 0);
}

inline dim3 gridBlocks(const dim3 thrConfig, const int3 size)
{
	return dim3(
		gridAxisBlocks(thrConfig.x, size.x),
		gridAxisBlocks(thrConfig.y, size.y),
		gridAxisBlocks(thrConfig.z, size.z)
	);
}

namespace kernel {

__device__
inline int3 globalPos3d()
{
	return make_int3(
		threadIdx.x + blockDim.x*blockIdx.x,
		threadIdx.y + blockDim.y*blockIdx.y,
		threadIdx.z + blockDim.z*blockIdx.z
	);
}

template <typename T>
__host__ __device__
inline T infOrMax()
{
	return std::numeric_limits<T>::max();
}

template<>
__host__ __device__
inline float infOrMax<float>()
{
#ifdef __CUDA_ARCH__
	return CUDART_INF_F;
#else
	return std::numeric_limits<float>::infinity();
#endif
}

template<>
__host__ __device__
inline double infOrMax<double>()
{
#ifdef __CUDA_ARCH__
	return CUDART_INF;
#else
	return std::numeric_limits<double>::infinity();
#endif
}

template <typename T>
__host__ __device__
inline T minusInfOrMin()
{
	return std::numeric_limits<T>::min();
}

template<>
__host__ __device__
inline float minusInfOrMin<float>()
{
#ifdef __CUDA_ARCH__
	return -CUDART_INF_F;
#else
	return -std::numeric_limits<float>::infinity();
#endif
}

template<>
__host__ __device__
inline double minusInfOrMin<double>()
{
#ifdef __CUDA_ARCH__
	return -CUDART_INF;
#else
	return -std::numeric_limits<double>::infinity();
#endif
}

} // namespace kernel

} // namespace gpho

#endif // UTIL_CUH__