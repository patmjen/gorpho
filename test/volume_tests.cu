#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "view.cuh"
#include "volume.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

template <class Vol>
class AllVolumesTest : public ::testing::Test {
public:
	using Volume = Vol;
};

class DeviceVolumeTest : public CudaTest {};
class PinnedVolumeTest : public CudaTest {};

// TODO: Also parameterize on the type contained in the volumes
using AllVolumeTypes = ::testing::Types<VolumeBase<float>, DeviceVolume<float>, HostVolume<float>, PinnedVolume<float>>;
TYPED_TEST_SUITE(AllVolumesTest, AllVolumeTypes);

TYPED_TEST(AllVolumesTest, DefaultInit)
{
	typename TestFixture::Volume vol;
	EXPECT_EQ(vol.data(), nullptr);
	EXPECT_EQ(vol.size(), int3_0);
}

template <class Vol, class Ty>
void verifyCopiedState(const Vol& vol, const Ty *expectedPtr, const int3& expectedSize,
	int expectedUseCount, std::string postfix = "")
{
	EXPECT_EQ(vol.data(), expectedPtr) << postfix;
	EXPECT_EQ(vol.size(), expectedSize) << postfix;
	EXPECT_EQ(vol.useCount(), expectedUseCount) << postfix;
}

template <class Vol>
void performCopyTest(const Vol& vol)
{
	int initialUseCount = vol.useCount();
	int3 size = vol.size();
	const typename Vol::Type *data_ptr = vol.data();

	// Copy construction
	{
		Vol vol2(vol);
		verifyCopiedState(vol2, data_ptr, size, initialUseCount + 1);
	}
	ASSERT_EQ(vol.useCount(), 1) << "Construction post-condition";

	// Copy assignment
	{
		Vol vol3 = vol;
		verifyCopiedState(vol3, data_ptr, size, initialUseCount + 1);
	}
	EXPECT_EQ(vol.useCount(), 1) << "Assignment post-condition";
}

TEST_F(DeviceVolumeTest, Copy)
{
	DeviceVolume<float> vol;
	ASSERT_NO_THROW(vol = makeDeviceVolume<float>(1, 1, 1));
	performCopyTest<DeviceVolume<float>>(vol);
	assertCudaSuccess();
}

TEST(HostVolumeTest, Copy)
{
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(1, 1, 1));
	performCopyTest<HostVolume<float>>(vol);
}

TEST_F(PinnedVolumeTest, Copy)
{
	PinnedVolume<float> vol;
	ASSERT_NO_THROW(vol = makePinnedVolume<float>(1, 1, 1));
	performCopyTest<PinnedVolume<float>>(vol);
	assertCudaSuccess();
}

template <class Vol, class Ty>
void verifyMovedState(const Vol& vol1, const Vol& vol2, const Ty *expectedPtr, const int3& expectedSize,
	int expectedUseCount, std::string postfix = "")
{
	verifyCopiedState(vol2, expectedPtr, expectedSize, expectedUseCount, postfix);

	EXPECT_EQ(vol1.data(), nullptr);
	EXPECT_EQ(vol1.size(), int3_0);
}

template <class Vol>
void performMoveConstructionTest(Vol vol)
{
	int initialUseCount = vol.useCount();
	const typename Vol::Type *dataPtr = vol.data();
	int3 size = vol.size();

	{
		Vol vol2(std::move(vol));
		verifyMovedState(vol, vol2, dataPtr, size, initialUseCount);
	}
	EXPECT_EQ(vol.useCount(), 0);
}

TEST_F(DeviceVolumeTest, MoveConstruction)
{
	DeviceVolume<float> vol;
	ASSERT_NO_THROW(vol = makeDeviceVolume<float>(1, 1, 1));
	performMoveConstructionTest<DeviceVolume<float>>(vol);
	assertCudaSuccess();
}

TEST(HostVolumeTest, MoveConstruction)
{
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(1, 1, 1));
	performMoveConstructionTest<HostVolume<float>>(vol);
}

TEST_F(PinnedVolumeTest, MoveConstruction)
{
	PinnedVolume<float> vol;
	ASSERT_NO_THROW(vol = makePinnedVolume<float>(1, 1, 1));
	performMoveConstructionTest<PinnedVolume<float>>(vol);
	assertCudaSuccess();
}

template <class Vol>
void performMoveAssignmentTest(Vol vol)
{
	int initialUseCount = vol.useCount();
	const typename Vol::Type *dataPtr = vol.data();
	int3 size = vol.size();

	{
		Vol vol2 = std::move(vol);
		verifyMovedState(vol, vol2, dataPtr, size, initialUseCount);
	}
	EXPECT_EQ(vol.useCount(), 0);
}

TEST_F(DeviceVolumeTest, MoveAssignment)
{
	DeviceVolume<float> vol;
	ASSERT_NO_THROW(vol = makeDeviceVolume<float>(1, 1, 1));
	performMoveAssignmentTest<DeviceVolume<float>>(vol);
	assertCudaSuccess();
}

TEST(HostVolumeTest, MoveAssignment)
{
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(1, 1, 1));
	performMoveAssignmentTest<HostVolume<float>>(vol);
}

TEST_F(PinnedVolumeTest, MoveAssignment)
{
	PinnedVolume<float> vol;
	ASSERT_NO_THROW(vol = makePinnedVolume<float>(1, 1, 1));
	performMoveAssignmentTest<PinnedVolume<float>>(vol);
	assertCudaSuccess();
}

TEST_F(DeviceVolumeTest, HostTransfer)
{
	float expected = 2.3f;
	DeviceVolume<float> dvol = makeDeviceVolume<float>(1, 1, 1);
	ASSERT_NO_FATAL_FAILURE(setDevicePtr(dvol.data(), expected));
	ASSERT_CUDA_EQ(dvol.data(), expected); // TODO: Write separate tests for setDevicePtr

	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = dvol.copyToHost());
	EXPECT_EQ(*vol.data(), expected);
}

TEST(HostVolumeTest, DeviceTransfer)
{
	float expected = 2.3f;
	HostVolume<float> vol = makeHostVolume<float>(1, 1, 1);
	*vol.data() = expected;
	ASSERT_EQ(*vol.data(), expected);
	
	DeviceVolume<float> dvol;
	ASSERT_NO_THROW(dvol = vol.copyToDevice());
	EXPECT_CUDA_EQ(dvol.data(), expected);
}

TEST(HostVolumeTest, Indexing)
{
	int3 size = make_int3(2, 3, 4);
	HostVolume<float> vol;
	ASSERT_NO_THROW(vol = makeHostVolume<float>(size));
	for (int i = 0; i < prod(size); ++i) {
		vol.data()[i] = i;
	}

	for (int i = 0; i < vol.numel(); ++i) {
		EXPECT_EQ(vol[i], i);
	}
	for (int x = 0; x < vol.size().x; ++x) {
		for (int y = 0; y < vol.size().y; ++y) {
			for (int z = 0; z < vol.size().z; ++z) {
				EXPECT_EQ(vol[make_int3(x, y, z)], vol.idx(x, y, z));
			}
		}
	}
}

template <class Vol>
class AllDerivedVolumesTest : public ::testing::Test {
public:
	using Volume = Vol;
};

// TODO: Also parameterize on the type contained in the volumes
// TODO: Maybe need to inherit from CudaTest or find way to mock memory allocations
using AllDerivedVolumeTypes = ::testing::Types<DeviceVolume<float>, HostVolume<float>, PinnedVolume<float>>;
TYPED_TEST_SUITE(AllDerivedVolumesTest, AllDerivedVolumeTypes);

TYPED_TEST(AllDerivedVolumesTest, GetView)
{
	using Volume = typename TestFixture::Volume;
	using Type = typename Volume::Type;
	Volume vol;
	ASSERT_NO_THROW(vol = makeVolume<Volume>(int3_1));
	auto vw = vol.view();

	EXPECT_EQ(vw.data(), vol.data());
	EXPECT_EQ(vw.size(), vol.size());

	// Make sure a const reference returns a const view
	const Volume& constRef = vol;
	auto cvw = constRef.view();
	::testing::StaticAssertTypeEq<decltype(cvw)::Type, const Type>();
}

TYPED_TEST(AllDerivedVolumesTest, ViewConversion)
{
	using Volume = typename TestFixture::Volume;
	using Type = typename Volume::Type;
	using View = typename Volume::View;
	using ConstView = typename Volume::ConstView;

	Volume vol;
	ASSERT_NO_THROW(vol = makeVolume<Volume>(int3_1));
	View vw = vol;

	EXPECT_EQ(vw.data(), vol.data());
	EXPECT_EQ(vw.size(), vol.size());

	// Make sure a const reference returns a const view
	const Volume& constRef = vol;
	ConstView cvw = constRef.view();
	::testing::StaticAssertTypeEq<decltype(cvw)::Type, const Type>();
}

TYPED_TEST(AllDerivedVolumesTest, Comparison)
{
	using Volume = typename TestFixture::Volume;
	int3 size1 = make_int3(2, 1, 3);
	int3 size2 = make_int3(4, 3, 1);

	Volume vol1, vol2, vol3, vol4, vol5;
	ASSERT_NO_THROW(vol1 = makeVolume<Volume>(size1));
	ASSERT_NO_THROW(vol3 = makeVolume<Volume>(size1));
	ASSERT_NO_THROW(vol4 = makeVolume<Volume>(size2));
	vol2 = vol1;

	EXPECT_EQ(vol1, vol1);
	EXPECT_EQ(vol1, vol2);
	EXPECT_NE(vol1, vol3);
	EXPECT_NE(vol1, vol4);
	vol2.reshape(1, 2, 3);
	ASSERT_NE(vol2.size(), vol1.size());
	EXPECT_NE(vol1, vol2);
}