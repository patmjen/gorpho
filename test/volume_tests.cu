#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;

template <class VolTy>
class AllVolumesTest : public ::testing::Test {
public:
	using VolumeType = VolTy;
};

// TODO: Also parameterize on the type contained in the volumes
using AllVolumeTypes = ::testing::Types<Volume<float>, DeviceVolume<float>, HostVolume<float>, PinnedVolume<float>>;
TYPED_TEST_SUITE(AllVolumesTest, AllVolumeTypes);

TYPED_TEST(AllVolumesTest, DefaultInit)
{
	typename TestFixture::VolumeType vol;
	EXPECT_EQ(vol.data(), nullptr);
	EXPECT_EQ(vol.size(), int3_0);
}

template <class VolTy, class Ty>
void verifyCopiedState(const VolTy& vol, const Ty *expectedPtr, const int3& expectedSize,
	const std::shared_ptr<Ty>& outerSharedPtr, int expectedUseCount, std::string postfix = "")
{
	EXPECT_EQ(vol.data(), expectedPtr) << postfix;
	EXPECT_EQ(vol.size(), expectedSize) << postfix;
	EXPECT_EQ(outerSharedPtr.use_count(), expectedUseCount) << postfix;
}

TYPED_TEST(AllVolumesTest, CopySharedPtrConstructor)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	ASSERT_EQ(xPtr.use_count(), 1) << "Test pre-condition";

	{
		typename TestFixture::VolumeType vol(xPtr, size);
		verifyCopiedState(vol, &x, size, xPtr, 2);
	}
	ASSERT_EQ(xPtr.use_count(), 1);

	{
		typename TestFixture::VolumeType vol(xPtr, size.x, size.y, size.z);
		verifyCopiedState(vol, &x, size, xPtr, 2);
	}
	EXPECT_EQ(xPtr.use_count(), 1);
}

TYPED_TEST(AllVolumesTest, Copy)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	typename TestFixture::VolumeType vol1(xPtr, size);
	ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

	// Copy construction
	{
		typename TestFixture::VolumeType vol2(vol1);
		verifyCopiedState(vol2, vol1.data(), size, xPtr, 3);
	}
	ASSERT_EQ(xPtr.use_count(), 2) << "Construction post-condition";

	// Copy assignment
	{
		typename TestFixture::VolumeType vol3 = vol1;
		verifyCopiedState(vol3, vol1.data(), size, xPtr, 3);
	}
	EXPECT_EQ(xPtr.use_count(), 2) << "Assignment post-condition";
}

TYPED_TEST(AllVolumesTest, MoveSharedPtrConstructor)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	ASSERT_EQ(xPtr.use_count(), 1) << "Test pre-condition";

	auto verifyState = [&](const auto& vol, const auto& xPtrCopy) {
		verifyCopiedState(vol, &x, size, xPtr, 2);

		EXPECT_EQ(xPtrCopy.get(), nullptr);
		EXPECT_EQ(xPtrCopy.use_count(), 0);
	};

	{
		std::shared_ptr<float> xPtrCopy(xPtr);
		ASSERT_EQ(xPtrCopy.get(), xPtr.get()) << "Test pre-condition";
		ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

		typename TestFixture::VolumeType vol(std::move(xPtrCopy), size);
		verifyState(vol, xPtrCopy);
	}
	EXPECT_EQ(xPtr.use_count(), 1);

	{
		std::shared_ptr<float> xPtrCopy(xPtr);
		ASSERT_EQ(xPtrCopy.get(), xPtr.get()) << "Test pre-condition";
		ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

		typename TestFixture::VolumeType vol(std::move(xPtrCopy), size.x, size.y, size.z);
		verifyState(vol, xPtrCopy);
	}
	EXPECT_EQ(xPtr.use_count(), 1);
}

template <class VolTy, class Ty>
void verifyMovedState(const VolTy& vol1, const VolTy& vol2, const Ty *expectedPtr, const int3& expectedSize,
	const std::shared_ptr<Ty>& outerSharedPtr, int expectedUseCount, std::string postfix = "")
{
	verifyCopiedState(vol2, expectedPtr, expectedSize, outerSharedPtr, expectedUseCount, postfix);

	EXPECT_EQ(vol1.data(), nullptr);
	EXPECT_EQ(vol1.size(), int3_0);
}

TYPED_TEST(AllVolumesTest, MoveConstruction)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	typename TestFixture::VolumeType vol1(xPtr, size);
	ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

	{
		typename TestFixture::VolumeType vol2(std::move(vol1));
		verifyMovedState(vol1, vol2, &x, size, xPtr, 2);
	}
	EXPECT_EQ(xPtr.use_count(), 1);
}

TYPED_TEST(AllVolumesTest, MoveAssignment)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	typename TestFixture::VolumeType vol1(xPtr, size);
	ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

	{
		typename TestFixture::VolumeType vol2 = std::move(vol1);
		verifyMovedState(vol1, vol2, &x, size, xPtr, 2);
	}
	EXPECT_EQ(xPtr.use_count(), 1);
}

TEST(VolumeTest, Numel)
{
	int3 size1 = make_int3(2, 3, 5);
	Volume<float> vol1(std::shared_ptr<float>(), size1);
	ASSERT_EQ(vol1.size(), size1) << "Test pre-condition";
	EXPECT_EQ(vol1.numel(), 2 * 3 * 5);

	int3 size2 = make_int3(2, 3, 0);
	Volume<float> vol2(std::shared_ptr<float>(), size2);
	ASSERT_EQ(vol2.size(), size2) << "Test pre-condition";
	EXPECT_EQ(vol2.numel(), 0);
}

TEST(VolumeTest, Reshape)
{
	int3 size = make_int3(2, 4, 8);
	Volume<float> vol(std::shared_ptr<float>(), size);
	ASSERT_EQ(vol.size(), size) << "Test pre-condition";

	int3 newSize = make_int3(16, 2, 2);
	EXPECT_NO_THROW(vol.reshape(newSize));
	ASSERT_EQ(vol.size(), newSize);

	EXPECT_NO_THROW(vol.reshape(size.x, size.y, size.z));
	EXPECT_EQ(vol.size(), size);

	EXPECT_THROW(vol.reshape(1, 2, 3), std::length_error);
}

TEST(VolumeTest, Idx)
{
	// NOTE: The point of this test is **only** to verify that the output matches the idx function
	int3 size = make_int3(2, 3, 5);
	Volume<float> vol(std::shared_ptr<float>(), size);
	ASSERT_EQ(vol.size(), size) << "Test pre-condition";

	EXPECT_EQ(vol.idx(0, 0, 0), idx(0, 0, 0, vol.size()));
	EXPECT_EQ(vol.idx(1, 0, 0), idx(1, 0, 0, vol.size()));
	EXPECT_EQ(vol.idx(0, 1, 0), idx(0, 1, 0, vol.size()));
	EXPECT_EQ(vol.idx(0, 0, 1), idx(0, 0, 1, vol.size()));
	EXPECT_EQ(vol.idx(1, 1, 3), idx(1, 1, 3, vol.size()));
	EXPECT_EQ(vol.idx(1, 2, 4), idx(1, 2, 4, vol.size()));
}

TEST(DeviceVolumeTest, Copy)
{
	// NOTE: We only test the parts we are specific to DeviceVolume
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	DeviceVolume<float> vol1(xPtr, int3_1);

	auto verifyData = [&](const auto& vol1, const auto& vol2, std::string postfix) {
		// We convert to a Volume reference so we actually get the shared_ptr pointer
		const Volume<float>& v1 = vol1;
		const Volume<float>& v2 = vol2;
		EXPECT_EQ(v1.data(), v2.data());
	};

	// Copy construction
	DeviceVolume<float> vol2(vol1);
	verifyData(vol1, vol2, "Construction");

	// Copy assignment
	DeviceVolume<float> vol3 = vol1;
	verifyData(vol1, vol3, "Assignment");
}

TEST(DeviceVolumeTest, Move)
{
	// NOTE: We only test the parts we are specific to DeviceVolume
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	DeviceVolume<float> vol1(xPtr, size);
	ASSERT_EQ(vol1.data(), &x);

	auto verifyData = [&](const auto& vol, std::string postfix) {
		const Volume<float>& v = vol;
		ASSERT_EQ(v.data(), &x) << postfix;
	};

	DeviceVolume<float> vol2(std::move(vol1));
	ASSERT_NO_FATAL_FAILURE(verifyData(vol2, "Construction"));

	DeviceVolume<float> vol3 = std::move(vol2);
	ASSERT_NO_FATAL_FAILURE(verifyData(vol3, "Assignment"));
}

template <class VolTy, class Ty>
void verifyConstruction(const VolTy& vol, Ty *data, int3 size)
{
	::testing::StaticAssertTypeEq<typename VolTy::Type, Ty>();
	EXPECT_EQ(vol.data(), data);
	EXPECT_EQ(vol.size(), size);
}

TEST(DeviceVolumeTest, PtrConstructor)
{
	int3 size = int3_1;
	float *x = nullptr;
	ASSERT_CUDA_SUCCESS(cudaMalloc(&x, prod(size) * sizeof(*x)));

	verifyConstruction(DeviceVolume<float>(x, size), x, size);
	EXPECT_CUDA_SUCCESS(cudaGetLastError());

	x = nullptr;
	ASSERT_CUDA_SUCCESS(cudaMalloc(&x, prod(size) * sizeof(*x)));
	verifyConstruction(DeviceVolume<float>(x, size.x, size.y, size.z), x, size);
	EXPECT_CUDA_SUCCESS(cudaGetLastError());
}

TEST(HostVolumeTest, PtrConstructor)
{
	int3 size = int3_1;
	float *x = nullptr;
	ASSERT_NO_THROW(x = new float[prod(size)]);

	verifyConstruction(HostVolume<float>(x, size), x, size);

	ASSERT_NO_THROW(x = new float[prod(size)]);
	verifyConstruction(HostVolume<float>(x, size.x, size.y, size.z), x, size);
}

TEST(HostVolumeTest, DifferentAllocator)
{
	int3 size = int3_1;
	float *x = static_cast<float *>(malloc(prod(size) * sizeof(*x)));
	ASSERT_NE(x, nullptr) << "Allocation failed";

	// We don't really have any way to check if the call to free was successful, except to run the
	// tests under something like valgrind. We can only check that it was called, and that it did
	// not crash the program.
	bool wasFreed = false;
	{
		auto deleter = [&](float *ptr) {
			wasFreed = true;
			free(ptr);
		};
		HostVolume<float> vol(x, size, deleter);
	}
	ASSERT_TRUE(wasFreed);
}

TEST(PinnedVolumeTest, PtrConstructor)
{
	float *x = nullptr;
	int3 size = int3_1;
	ASSERT_CUDA_SUCCESS(cudaMallocHost(&x, prod(size) * sizeof(*x)));

	verifyConstruction(PinnedVolume<float>(x, size), x, size);
	EXPECT_CUDA_SUCCESS(cudaGetLastError());

	x = nullptr;
	ASSERT_CUDA_SUCCESS(cudaMallocHost(&x, prod(size) * sizeof(*x)));
	verifyConstruction(PinnedVolume<float>(x, size.x, size.y, size.z), x, size);
	EXPECT_CUDA_SUCCESS(cudaGetLastError());
}