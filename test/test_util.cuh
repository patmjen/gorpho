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

#define EXPECT_VOL_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return assertVolumeEqual(e1, e2, a1, a2); }, expected, actual)

#define ASSERT_VOL_EQ(expected, actual) \
    ASSERT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return assertVolumeEqual(e1, e2, a1, a2); }, expected, actual)

#define EXPECT_VOL_NE(expected, actual) \
    EXPECT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return !assertVolumeEqual(e1, e2, a1, a2); }, expected, actual)

#define ASSERT_VOL_NE(expected, actual) \
    ASSERT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return !assertVolumeEqual(e1, e2, a1, a2); }, expected, actual)

template <class Ty1, class Ty2>
::testing::AssertionResult assertVolumeEqual(const char *expr1, const char *expr2,
	gpho::detail::ViewBase<Ty1> a1, gpho::detail::ViewBase<Ty2> a2)
{
	if (a1.size() != a2.size()) {
		return ::testing::AssertionFailure() << expr1 << " and " << expr2 <<
			" have different number of elements.\nExpected: (" <<
			a1.size().x << "," << a1.size().y << "," << a1.size().z << ")\nActual: (" <<
			a2.size().x << "," << a2.size().y << "," << a2.size().z << ")";
	}
	if (a1.numel() <= 0) {
		return ::testing::AssertionSuccess() << "Volumes are empty.";
	}
	int numFail = 0;
	for (int i = 0; i < a1.numel(); i++) {
		if (a1[i] != a2[i]) {
			numFail++;
		}
	}
	if (numFail == 0) {
		return ::testing::AssertionSuccess() << expr1 << " and " << expr2 << " are equal.";
	}
	auto out = ::testing::AssertionFailure() << expr1 << " and " << expr2 << " differ in " << numFail <<
		" elements.\nExpected: [" << a1[0];
	for (int i = 1; i < a1.numel(); i++) {
		out << ", " << a1[i];
	}
	out << "]\nActual:   [" << a2[0];
	for (int i = 1; i < a2.numel(); i++) {
		out << ", " << a2[i];
	}
	out << "]";
	return out;
}

template <class Ty>
__global__ void setDevicePtrKernel(Ty *ptr, Ty val)
{
	*ptr = val;
}

class CudaTest : public ::testing::Test {
public:
	static void TearDownTestCase()
	{
		cudaDeviceReset();
	}

protected:
	void SetUp() override
	{
		cudaDeviceReset();
	}

	template <class Ty>
	void setDevicePtr(Ty *ptr, Ty val)
	{
		// TODO: Find way to forcefully propagate error state
		setDevicePtrKernel<Ty><<<1, 1>>>(ptr, val);
		syncAndAssertCudaSuccess();
	}

	void assertCudaSuccess() const
	{
		ASSERT_CUDA_SUCCESS(cudaPeekAtLastError());
	}

	void syncAndAssertCudaSuccess() const
	{
		ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
	}

	void expectCudaSuccess() const
	{
		EXPECT_CUDA_SUCCESS(cudaPeekAtLastError());
	}

	void syncAndExpectCudaSuccess() const
	{
		EXPECT_CUDA_SUCCESS(cudaDeviceSynchronize());
	}
};

#endif // TEST_UTIL_CUH__