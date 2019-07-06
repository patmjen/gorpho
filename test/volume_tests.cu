#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

template <class VolTy>
class AllVolumesTest : public ::testing::Test {
public:
	using VolumeType = VolTy;
};

// TODO: Also parameterize on the type contained in the volumes
using AllVolumeTypes = ::testing::Types<VolumeBase<float>, DeviceVolume<float>, HostVolume<float>, PinnedVolume<float>>;
TYPED_TEST_SUITE(AllVolumesTest, AllVolumeTypes);

TYPED_TEST(AllVolumesTest, DefaultInit)
{
	typename TestFixture::VolumeType vol;
	EXPECT_EQ(vol.data(), nullptr);
	EXPECT_EQ(vol.size(), int3_0);
}

template <class VolTy, class Ty>
void verifyCopiedState(const VolTy& vol, const Ty *expectedPtr, const int3& expectedSize,
	int expectedUseCount, std::string postfix = "")
{
	EXPECT_EQ(vol.data(), expectedPtr) << postfix;
	EXPECT_EQ(vol.size(), expectedSize) << postfix;
	EXPECT_EQ(vol.useCount(), expectedUseCount) << postfix;
}

template <class VolTy>
void performCopyTest(const VolTy& vol)
{
	int initialUseCount = vol.useCount();
	int3 size = vol.size();
	const typename VolTy::Type *data_ptr = vol.data();

	// Copy construction
	{
		VolTy vol2(vol);
		verifyCopiedState(vol2, data_ptr, size, initialUseCount + 1);
	}
	ASSERT_EQ(vol.useCount(), 1) << "Construction post-condition";

	// Copy assignment
	{
		VolTy vol3 = vol;
		verifyCopiedState(vol3, data_ptr, size, initialUseCount + 1);
	}
	EXPECT_EQ(vol.useCount(), 1) << "Assignment post-condition";
}

TEST(DeviceVolumeTest, Copy)
{
	DeviceVolume<float> vol;
	ASSERT_NO_THROW(vol = makeDeviceVolume<float>(1, 1, 1));
	performCopyTest<DeviceVolume<float>>(vol);
	ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

TEST(HostVolumeTest, Copy)
{
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(1, 1, 1));
	performCopyTest<HostVolume<float>>(vol);
}

TEST(PinnedVolumeTest, Copy)
{
	PinnedVolume<float> vol;
	ASSERT_NO_THROW(vol = makePinnedVolume<float>(1, 1, 1));
	performCopyTest<PinnedVolume<float>>(vol);
	ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

template <class VolTy, class Ty>
void verifyMovedState(const VolTy& vol1, const VolTy& vol2, const Ty *expectedPtr, const int3& expectedSize,
	int expectedUseCount, std::string postfix = "")
{
	verifyCopiedState(vol2, expectedPtr, expectedSize, expectedUseCount, postfix);

	EXPECT_EQ(vol1.data(), nullptr);
	EXPECT_EQ(vol1.size(), int3_0);
}

template <class VolTy>
void performMoveConstructionTest(VolTy vol)
{
	int initialUseCount = vol.useCount();
	const typename VolTy::Type *dataPtr = vol.data();
	int3 size = vol.size();

	{
		VolTy vol2(std::move(vol));
		verifyMovedState(vol, vol2, dataPtr, size, initialUseCount);
	}
	EXPECT_EQ(vol.useCount(), 0);
}

TEST(DeviceVolumeTest, MoveConstruction)
{
	DeviceVolume<float> vol;
	ASSERT_NO_THROW(vol = makeDeviceVolume<float>(1, 1, 1));
	performMoveConstructionTest<DeviceVolume<float>>(vol);
	ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

TEST(HostVolumeTest, MoveConstruction)
{
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(1, 1, 1));
	performMoveConstructionTest<HostVolume<float>>(vol);
}

TEST(PinnedVolumeTest, MoveConstruction)
{
	PinnedVolume<float> vol;
	ASSERT_NO_THROW(vol = makePinnedVolume<float>(1, 1, 1));
	performMoveConstructionTest<PinnedVolume<float>>(vol);
	ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

template <class VolTy>
void performMoveAssignmentTest(VolTy vol)
{
	int initialUseCount = vol.useCount();
	const typename VolTy::Type *dataPtr = vol.data();
	int3 size = vol.size();

	{
		VolTy vol2 = std::move(vol);
		verifyMovedState(vol, vol2, dataPtr, size, initialUseCount);
	}
	EXPECT_EQ(vol.useCount(), 0);
}

TEST(DeviceVolumeTest, MoveAssignment)
{
	DeviceVolume<float> vol;
	ASSERT_NO_THROW(vol = makeDeviceVolume<float>(1, 1, 1));
	performMoveAssignmentTest<DeviceVolume<float>>(vol);
	ASSERT_CUDA_SUCCESS(cudaGetLastError());
}

TEST(HostVolumeTest, MoveAssignment)
{
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(1, 1, 1));
	performMoveAssignmentTest<HostVolume<float>>(vol);
}

TEST(PinnedVolumeTest, MoveAssignment)
{
	PinnedVolume<float> vol;
	ASSERT_NO_THROW(vol = makePinnedVolume<float>(1, 1, 1));
	performMoveAssignmentTest<PinnedVolume<float>>(vol);
	ASSERT_CUDA_SUCCESS(cudaGetLastError());
}


template <class VolTy, class Ty>
void verifyConstruction(const VolTy& vol, Ty *data, int3 size)
{
	EXPECT_EQ(vol.data(), data);
	EXPECT_EQ(vol.size(), size);
}

TEST(SizedBaseTest, Numel)
{
	int3 size1 = make_int3(2, 3, 5);
	SizedBase sb1(size1);
	ASSERT_EQ(sb1.size(), size1) << "Test pre-condition";
	EXPECT_EQ(sb1.numel(), 2 * 3 * 5);

	int3 size2 = make_int3(2, 3, 0);
	SizedBase sb2(size2);
	ASSERT_EQ(sb2.size(), size2) << "Test pre-condition";
	EXPECT_EQ(sb2.numel(), 0);
}

TEST(SizedBaseTest, Reshape)
{
	// TODO: Test device version
	int3 size = make_int3(2, 4, 8);
	SizedBase sb(size);
	ASSERT_EQ(sb.size(), size) << "Test pre-condition";

	int3 newSize = make_int3(16, 2, 2);
	EXPECT_NO_THROW(sb.reshape(newSize));
	ASSERT_EQ(sb.size(), newSize);

	EXPECT_NO_THROW(sb.reshape(size.x, size.y, size.z));
	EXPECT_EQ(sb.size(), size);

	EXPECT_THROW(sb.reshape(1, 2, 3), std::length_error);
}

TEST(SizedBaseTest, Idx)
{
	// NOTE: The point of this test is **only** to verify that the output matches the idx function
	int3 size = make_int3(2, 3, 5);
	SizedBase sb(size);
	ASSERT_EQ(sb.size(), size) << "Test pre-condition";

	EXPECT_EQ(sb.idx(0, 0, 0), idx(0, 0, 0, sb.size()));
	EXPECT_EQ(sb.idx(1, 0, 0), idx(1, 0, 0, sb.size()));
	EXPECT_EQ(sb.idx(0, 1, 0), idx(0, 1, 0, sb.size()));
	EXPECT_EQ(sb.idx(0, 0, 1), idx(0, 0, 1, sb.size()));
	EXPECT_EQ(sb.idx(1, 1, 3), idx(1, 1, 3, sb.size()));
	EXPECT_EQ(sb.idx(1, 2, 4), idx(1, 2, 4, sb.size()));
}