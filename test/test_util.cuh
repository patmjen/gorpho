#ifndef TEST_UTIL_CUH__
#define TEST_UTIL_CUH__

#include <cuda_runtime.h>

#define ASSERT_CUDA_SUCCESS(expr) do { \
	cudaError_t res__ = (expr); \
	ASSERT_EQ(res__, cudaSuccess) << "Expression '" #expr "' resulted in CUDA error: " << cudaGetErrorString(res__); \
} while(false)

#define EXPECT_CUDA_SUCCESS(expr) do { \
	cudaError_t res__ = (expr); \
	EXPECT_EQ(res__, cudaSuccess) << "Expression '" #expr "' resulted in CUDA error: " << cudaGetErrorString(res__); \
} while (false)

const int3 int3_0 = make_int3(0, 0, 0);
const int3 int3_1 = make_int3(1, 1, 1);

template <class Ty>
__global__ void setDevicePtrKernel(Ty *ptr, Ty val)
{
	*ptr = val;
}

template <class Ty>
void setDevicePtr(Ty *ptr, Ty val)
{
	// TODO: Find way to forcefully propagate error state
	setDevicePtrKernel<Ty><<<1, 1 >>>(ptr, val);
	ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
}

#define EXPECT_CUDA_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return assertDevicePtrEqual(e1, e2, a1, a2); }, expected, actual)

#define ASSERT_CUDA_EQ(expected, actual) \
    ASSERT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return assertDevicePtrEqual(e1, e2, a1, a2); }, expected, actual)

#define EXPECT_CUDA_NE(expected, actual) \
    EXPECT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return !assertDevicePtrEqual(e1, e2, a1, a2); }, expected, actual)

#define ASSERT_CUDA_NE(expected, actual) \
    ASSERT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return !assertDevicePtrEqual(e1, e2, a1, a2); }, expected, actual)

template <class Ty>
::testing::AssertionResult assertDevicePtrEqual(const char *exprPtr, const char *exprExpected, 
	const Ty *ptr, Ty expected)
{
	Ty actual;
	cudaError_t res = cudaMemcpy(&actual, ptr, sizeof(Ty), cudaMemcpyDeviceToHost);
	if (res != cudaSuccess) {
		auto out = ::testing::AssertionFailure() << "could not transfer " << exprPtr << " to host: ";
		out << cudaGetErrorString(res);
		return out;
	}
	if (actual != expected) {
		auto out = ::testing::AssertionFailure() << "device value does not match expected value.\n";
		out << "Device value was pointed by:\n  " << exprPtr << "\n  which was:\n  " << actual << "\n";
		out << "Expected value:\n  " << exprExpected << "\nwhich was:\n  " << expected;
		return out;
	}
	auto out = ::testing::AssertionSuccess() << "device value matches value.\n";
	out << "Device value was pointed by:\n  " << exprPtr << "\n  which was:\n  " << actual << "\n";
	out << "Expected value:\n  " << exprExpected << "\nwhich was:\n  " << expected;
	return out;
}

#endif // TEST_UTIL_CUH__