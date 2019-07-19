#include <cuda_runtime.h>
#include <algorithm>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "view.cuh"
#include "morph.cuh"
#include "flat_linear_morph.cuh"
#include "strel.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

class FlatLinearMorphTest : public CudaTest {};

void performLinesTest(HostView<float> expectedRes, HostView<const float> vol, const std::vector<LineSeg>& lines,
	bool onlyUseFirstLineSeg = false)
{
	size_t bufSize = minTotalBufferSize(minRSBufferSize(lines), vol.size());
	DeviceVolume<float> dvol, dres, rBuffer, sBuffer, dresBuffer;
	ASSERT_NO_THROW(dvol = makeDeviceVolume<float>(vol.size()));
	ASSERT_NO_THROW(dres = makeDeviceVolume<float>(vol.size()));
	ASSERT_NO_THROW(rBuffer = makeDeviceVolume<float>(bufSize, 1, 1));
	ASSERT_NO_THROW(sBuffer = makeDeviceVolume<float>(bufSize, 1, 1));

	transfer(dvol.view(), vol);

	if (onlyUseFirstLineSeg) {
		flatLinearDilateErode<MORPH_DILATE, float>(dres, dvol, rBuffer, sBuffer, lines[0]);
	} else {
		ASSERT_NO_THROW(dresBuffer = makeDeviceVolume<float>(vol.size()));
		flatLinearDilateErode<MORPH_DILATE, float>(dres, dresBuffer, dvol, rBuffer, sBuffer, lines);
	}
	ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

	HostVolume<float> res;
	ASSERT_NO_THROW(res = dres.copyToHost());

	EXPECT_VOL_EQ(expectedRes, res.view());
}

void performSingleLineTest(HostView<float> expectedRes, HostView<const float> vol, LineSeg line)
{
	performLinesTest(expectedRes, vol, { line }, true);
}

TEST_F(FlatLinearMorphTest, SingleLineAxisAligned)
{
	int3 volSize = make_int3(7, 3, 3);
	float expectedResData[7 * 3 * 3];
	float volData[7 * 3 * 3];
	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> vol(volData, volSize);

	{
		SCOPED_TRACE("Line aligned with 1st axis");

		std::fill(vol.data(), vol.data() + vol.numel(), 0.0f);
		vol[make_int3(3, 1, 1)] = 1.0f;

		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		for (int i = 1; i < expectedRes.size().x - 1; ++i) {
			expectedRes[make_int3(i, 1, 1)] = 1.0f;
		}
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(1, 0, 0), 5));
	}
	{
		SCOPED_TRACE("Line aligned with 2nd axis");
		vol.reshape(3, 7, 3);
		std::fill(vol.data(), vol.data() + vol.numel(), 0.0f);
		vol[make_int3(1, 3, 1)] = 1.0f;

		expectedRes.reshape(3, 7, 3);
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		for (int i = 1; i < expectedRes.size().y - 1; ++i) {
			expectedRes[make_int3(1, i, 1)] = 1.0f;
		}
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(0, 1, 0), 5));
	}
	{
		SCOPED_TRACE("Line aligned with 3rd axis");
		vol.reshape(3, 3, 7);
		std::fill(vol.data(), vol.data() + vol.numel(), 0.0f);
		vol[make_int3(1, 1, 3)] = 1.0f;

		expectedRes.reshape(3, 3, 7);
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		for (int i = 1; i < expectedRes.size().z - 1; ++i) {
			expectedRes[make_int3(1, 1, i)] = 1.0f;
		}
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(0, 0, 1), 5));
	}

	assertCudaSuccess();
}

TEST_F(FlatLinearMorphTest, SingleLineAxisAlignedEvenLength)
{
	int3 volSize = make_int3(5, 3, 3);
	float expectedResData[5 * 3 * 3];
	float volData[5 * 3 * 3] = { 0.0f };
	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> vol(volData, volSize);

	vol[make_int3(2, 1, 1)] = 1.0f;

	{
		SCOPED_TRACE("Step = (1, 0, 0)");
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		expectedRes[make_int3(2, 1, 1)] = 1.0f;
		expectedRes[make_int3(3, 1, 1)] = 1.0f;
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(1, 0, 0), 2));
	}
	{
		SCOPED_TRACE("Step = (-1, 0, 0)");
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		expectedRes[make_int3(2, 1, 1)] = 1.0f;
		expectedRes[make_int3(1, 1, 1)] = 1.0f;
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(-1, 0, 0), 2));
	}

	assertCudaSuccess();
}

TEST_F(FlatLinearMorphTest, SingleLineNotAxisAligned)
{
	int3 volSize = make_int3(5, 5, 5);
	float expectedResData[5 * 5 * 5];
	float volData[5 * 5 * 5] = { 0.0f };
	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> vol(volData, volSize);

	vol[make_int3(2, 2, 2)] = 1.0f;

	{
		SCOPED_TRACE("Step = (1, 1, 0)");
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		for (int i = 1; i < 4; ++i) {
			expectedRes[make_int3(i, i, 2)] = 1.0f;
		}
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(1, 1, 0), 3));
	}
	{
		SCOPED_TRACE("Step = (1, 1, 1)");
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		for (int i = 1; i < 4; ++i) {
			expectedRes[make_int3(i, i, i)] = 1.0f;
		}
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(1, 1, 1), 3));
	}
	{
		SCOPED_TRACE("Step = (-1, 1, 0)");
		std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0.0f);
		for (int i = 1; i < 4; ++i) {
			expectedRes[make_int3(4 - i, i, 2)] = 1.0f;
		}
		performSingleLineTest(expectedRes, vol, LineSeg(make_int3(-1, 1, 0), 3));
	}

	assertCudaSuccess();
}

TEST_F(FlatLinearMorphTest, MutipleLines)
{
	int3 volSize = make_int3(9, 9, 9);
	float expectedResData[9 * 9 * 9] = { 0.0f };
	float volData[9 * 9 * 9] = { 0.0f };
	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> vol(volData, volSize);

	vol[make_int3(4, 4, 4)] = 1.0f;

	for (int x = 0; x < 3; ++x) {
		for (int y = 0; y < 5; ++y) {
			for (int z = 0; z < 7; ++z) {
				expectedRes[make_int3(x + 3, y + 2, z + 1)] = 1.0f;
			}
		}
	}
	std::vector<LineSeg> lines = {
		LineSeg(make_int3(1, 0, 0), 3),
		LineSeg(make_int3(0, 1, 0), 5),
		LineSeg(make_int3(0, 0, 1), 7)
	};

	performLinesTest(expectedRes, vol, lines);

	assertCudaSuccess();
}

TEST_F(FlatLinearMorphTest, HostInput)
{

	int3 volSize = make_int3(9, 9, 9);
	float expectedResData[9 * 9 * 9] = { 0.0f };
	float volData[9 * 9 * 9] = { 0.0f };
	float resData[9 * 9 * 9] = { 0.0f };
	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> vol(volData, volSize);
	HostView<float> res(resData, volSize);

	vol[make_int3(4, 4, 4)] = 1.0f;

	for (int x = 0; x < 3; ++x) {
		for (int y = 0; y < 5; ++y) {
			for (int z = 0; z < 7; ++z) {
				expectedRes[make_int3(x + 3, y + 2, z + 1)] = 1.0f;
			}
		}
	}
	std::vector<LineSeg> lines = {
		LineSeg(make_int3(1, 0, 0), 3),
		LineSeg(make_int3(0, 1, 0), 5),
		LineSeg(make_int3(0, 0, 1), 7)
	};

	try {
		flatLinearDilateErode<MORPH_DILATE, float>(res, vol, lines, make_int3(2, 2, 2));
	} catch (const std::exception e) {
		FAIL() << e.what();
	} catch (...) {
		FAIL();
	}
	syncAndAssertCudaSuccess();

	ASSERT_VOL_EQ(expectedRes, res);
}

TEST_F(FlatLinearMorphTest, EmptyOp)
{
	int3 volSize = make_int3(3, 3, 3);
	float volData[3 * 3 * 3] = { 0.0f };
	HostView<float> vol(volData, volSize);

	vol[make_int3(1, 1, 1)] = 1.0f;

	{
		SCOPED_TRACE("Step = (0, 0, 0)");
		performSingleLineTest(vol, vol, LineSeg(make_int3(0, 0, 0), 5));
	}
	{
		SCOPED_TRACE("Length = 1");
		performSingleLineTest(vol, vol, LineSeg(make_int3(1, 0, 0), 1));
	}
	{
		SCOPED_TRACE("Length = 0");
		performSingleLineTest(vol, vol, LineSeg(make_int3(1, 0, 0), 0));
	}
}

TEST_F(FlatLinearMorphTest, MultipleEmptyOp)
{
	int3 volSize = make_int3(5, 5, 5);
	float volData[5 * 5 * 5] = { 0.0f };
	float expectedResData[5 * 5 * 5] = { 0.0f };
	HostView<float> vol(volData, volSize);
	HostView<float> expectedRes(expectedResData, volSize);

	vol[make_int3(2, 2, 2)] = 1.0f;

	{
		SCOPED_TRACE("All empty steps");
		// For this case, the output should be same as the input
		performLinesTest(vol, vol, {
			LineSeg(make_int3(0, 0, 0), 5),
			LineSeg(make_int3(1, 0, 0), 0)
		});
	}
	{
		SCOPED_TRACE("One empty step");
		for (int x = 1; x < 4; ++x) {
			for (int y = 1; y < 4; ++y) {
				expectedRes[make_int3(x, y, 2)] = 1.0f;
			}
		}
		performLinesTest(expectedRes, vol, {
			LineSeg(make_int3(1, 0, 0), 3),
			LineSeg(make_int3(0, 0, 1), 0),
			LineSeg(make_int3(0, 1, 0), 3)
		});
	}
}