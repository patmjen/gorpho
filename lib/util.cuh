#ifndef UTIL_CUH__
#define UTIL_CUH__

#include <limits>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace gpho {

__host__
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
__device__
inline T infOrMax()
{
	// TODO: Maybe do specialization for long double?
	return std::numeric_limits<T>::max();
}

template<>
__device__
inline float infOrMax<float>()
{
	return CUDART_INF_F;
}

template<>
__device__
inline double infOrMax<double>()
{
	return CUDART_INF;
}

template <typename T>
__device__
inline T minusInfOrMin()
{
	// TODO: Maybe do specialization for long double?
	return std::numeric_limits<T>::min();
}

template<>
__device__
inline float minusInfOrMin<float>()
{
	return -CUDART_INF_F;
}

template<>
__device__
inline double minusInfOrMin<double>()
{
	return -CUDART_INF;
}

} // namespace gpho

#endif // UTIL_CUH__