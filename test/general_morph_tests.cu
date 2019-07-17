#include <cuda_runtime.h>
#include <stdexcept>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "view.cuh"
#include "general_morph.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

class GeneralMorphTest : public CudaTest {};

TEST_F(GeneralMorphTest, DeviceInput)
{
	float expectedResData[] = {
		2, 2, 1, 1, 1,
		2, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,

		2, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,

		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 2, 2, 2, 1,
		1, 1, 2, 1, 1,
		1, 1, 1, 1, 1,

		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,

		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 2, 2, 2, 1
	};
	float volData[] = {
		1, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 1, 0, 0
	};
	float strelData[] = {
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,

		0, 1, 0,
		1, 1, 1,
		0, 1, 0,

		0, 0, 0,
		0, 1, 0,
		0, 0, 0
	};

	int3 volSize = make_int3(5, 5, 5);
	int3 strelSize = make_int3(3, 3, 3);

	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> vol(volData, volSize);
	HostView<float> strel(strelData, strelSize);

	DeviceVolume<float> dvol, dres, dstrel;
	ASSERT_NO_THROW(dvol = makeDeviceVolume<float>(volSize));
	ASSERT_NO_THROW(dres = makeDeviceVolume<float>(volSize));
	ASSERT_NO_THROW(dstrel = makeDeviceVolume<float>(strelSize));

	ASSERT_NO_THROW(transfer(dvol.view(), vol));
	ASSERT_NO_THROW(transfer(dstrel.view(), strel));

	genDilateErode<MORPH_DILATE, float>(dres, dvol, dstrel);
	syncAndAssertCudaSuccess();

	HostVolume<float> res;
	ASSERT_NO_THROW(res = dres.copyToHost());

	EXPECT_VOL_EQ(expectedRes, res.view());

	assertCudaSuccess();
}

TEST_F(GeneralMorphTest, HostInput)
{
	float expectedResData[] = {
		2, 2, 1, 1, 1,
		2, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,

		2, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,

		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 2, 2, 2, 1,
		1, 1, 2, 1, 1,
		1, 1, 1, 1, 1,

		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,

		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 2, 1, 1,
		1, 2, 2, 2, 1
	};
	float volData[] = {
		1, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,

		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 1, 0, 0
	};
	float strelData[] = {
		0, 0, 0,
		0, 1, 0,
		0, 0, 0,

		0, 1, 0,
		1, 1, 1,
		0, 1, 0,

		0, 0, 0,
		0, 1, 0,
		0, 0, 0
	};
	float resData[5 * 5 * 5] = { -1 };

	int3 volSize = make_int3(5, 5, 5);
	int3 strelSize = make_int3(3, 3, 3);

	HostView<float> expectedRes(expectedResData, volSize);
	HostView<float> res(resData, volSize);
	HostView<const float> vol(volData, volSize);
	HostView<const float> strel(strelData, strelSize);

	try {
		genDilateErode<MORPH_DILATE, float>(res, vol, strel, make_int3(2, 2, 2));
	} catch (const std::exception& e) {
		FAIL() << e.what();
	}
	syncAndAssertCudaSuccess();

	EXPECT_VOL_EQ(expectedRes, res);

	assertCudaSuccess();
}