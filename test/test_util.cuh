#ifndef TEST_UTIL_CUH__
#define TEST_UTIL_CUH__

#define ASSERT_CUDA_SUCCESS(expr) do { \
	cudaError_t res__ = (expr); \
	ASSERT_EQ(res__, cudaSuccess) << "CUDA error: " << cudaGetErrorString(res__); \
} while(false)

#define EXPECT_CUDA_SUCCESS(expr) do { \
	cudaError_t res__ = (expr); \
	EXPECT_EQ(res__, cudaSuccess) << "CUDA error: " << cudaGetErrorString(res__); \
} while (false)

const int3 int3_0 = make_int3(0, 0, 0);
const int3 int3_1 = make_int3(1, 1, 1);

#endif // TEST_UTIL_CUH__