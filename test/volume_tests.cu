#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::test;

TEST(VolumeTest, DefaultInit)
{
	Volume<float> vol;
	EXPECT_EQ(vol.data(), nullptr);
	EXPECT_EQ(vol.size(), int3_0);
}

TEST(VolumeTest, ParamConstructor)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	ASSERT_EQ(xPtr.use_count(), 1) << "Test pre-condition";

	// Copy construct shared_ptr
	{
		Volume<float> vol1(xPtr, size);
		EXPECT_EQ(vol1.data(), &x);
		EXPECT_EQ(vol1.size(), size);
		EXPECT_EQ(xPtr.use_count(), 2);
	}
	ASSERT_EQ(xPtr.use_count(), 1) << "Copy post-condition";

	// Move construct shared_ptr
	std::shared_ptr<float> xPtrCopy(xPtr);
	ASSERT_EQ(xPtrCopy.get(), xPtr.get()) << "Move pre-condition";
	ASSERT_EQ(xPtr.use_count(), 2) << "Move pre-condition";

	{
		Volume<float> vol2(std::move(xPtrCopy), size);
		EXPECT_EQ(vol2.data(), &x);
		EXPECT_EQ(vol2.size(), size);

		EXPECT_EQ(xPtrCopy.get(), nullptr);
		EXPECT_EQ(xPtrCopy.use_count(), 0);

		EXPECT_EQ(xPtr.use_count(), 2);
	}
	EXPECT_EQ(xPtr.use_count(), 1);
}

TEST(VolumeTest, Copy)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	Volume<float> vol1(xPtr, int3_1);
	ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

	// Copy construction
	{
		Volume<float> vol2(vol1);
		EXPECT_EQ(vol2.data(), vol1.data()) << "Construction";
		EXPECT_EQ(vol2.size(), vol1.size()) << "Construction";

		EXPECT_EQ(xPtr.use_count(), 3) << "Construction";
	}
	ASSERT_EQ(xPtr.use_count(), 2) << "Construction post-condition";

	// Copy assignment
	{
		Volume<float> vol3 = vol1;
		EXPECT_EQ(vol3.data(), vol1.data()) << "Assignment";
		EXPECT_EQ(vol3.size(), vol1.size()) << "Assignment";

		EXPECT_EQ(xPtr.use_count(), 3) << "Assignment";
	}
	EXPECT_EQ(xPtr.use_count(), 2) << "Assignment post-condition";
}

TEST(VolumeTest, MoveConstruction)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	Volume<float> vol1(xPtr, size);
	ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

	{
		Volume<float> vol2(std::move(vol1));
		EXPECT_EQ(vol2.data(), &x);
		EXPECT_EQ(vol2.size(), size);

		EXPECT_EQ(vol1.data(), nullptr);
		EXPECT_EQ(vol1.size(), int3_0);

		EXPECT_EQ(xPtr.use_count(), 2);
	}
	EXPECT_EQ(xPtr.use_count(), 1);
}

TEST(VolumeTest, MoveAssignment)
{
	float x = 1.0f;
	std::shared_ptr<float> xPtr(&x, nonDeleter<float>);
	int3 size = int3_1;
	Volume<float> vol1(xPtr, size);
	ASSERT_EQ(xPtr.use_count(), 2) << "Test pre-condition";

	{
		Volume<float> vol2 = std::move(vol1);
		EXPECT_EQ(vol2.data(), &x);
		EXPECT_EQ(vol2.size(), size);

		EXPECT_EQ(vol1.data(), nullptr);
		EXPECT_EQ(vol1.size(), int3_0);

		EXPECT_EQ(xPtr.use_count(), 2);
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
	int3 size = make_int3(2, 3, 5);
	Volume<float> vol(std::shared_ptr<float>(), size);
	ASSERT_EQ(vol.size(), size) << "Test pre-condition";

	// NOTE: The point of this test is **only** to verify that the output matches the idx function
	EXPECT_EQ(vol.idx(0, 0, 0), idx(0, 0, 0, vol.size()));
	EXPECT_EQ(vol.idx(1, 0, 0), idx(1, 0, 0, vol.size()));
	EXPECT_EQ(vol.idx(0, 1, 0), idx(0, 1, 0, vol.size()));
	EXPECT_EQ(vol.idx(0, 0, 1), idx(0, 0, 1, vol.size()));
	EXPECT_EQ(vol.idx(1, 1, 3), idx(1, 1, 3, vol.size()));
	EXPECT_EQ(vol.idx(1, 2, 4), idx(1, 2, 4, vol.size()));
}