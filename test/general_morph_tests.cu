#include <cuda_runtime.h>
#include <stdexcept>
#include <cinttypes>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "view.cuh"
#include "general_morph.cuh"
#include "util.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

template <class Ty>
class GeneralMorphTest : public CudaTest {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(GeneralMorphTest, AllPodTypes);

TYPED_TEST(GeneralMorphTest, DeviceInput)
{
    using Type = typename TestFixture::Type;
    Type expectedResData[] = {
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
    Type volData[] = {
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
    Type strelData[] = {
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

    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);
    HostView<Type> strel(strelData, strelSize);

    DeviceVolume<Type> dvol, dres, dstrel;
    ASSERT_NO_THROW(dvol = makeDeviceVolume<Type>(volSize));
    ASSERT_NO_THROW(dres = makeDeviceVolume<Type>(volSize));
    ASSERT_NO_THROW(dstrel = makeDeviceVolume<Type>(strelSize));

    ASSERT_NO_THROW(transfer(dvol.view(), vol));
    ASSERT_NO_THROW(transfer(dstrel.view(), strel));

    genDilateErode<MORPH_DILATE, Type>(dres, dvol, dstrel);
    this->syncAndAssertCudaSuccess();

    HostVolume<Type> res;
    ASSERT_NO_THROW(res = dres.copyToHost());

    EXPECT_VOL_EQ(expectedRes, res.view());

    this->assertCudaSuccess();
}

TYPED_TEST(GeneralMorphTest, HostInput)
{
    using Type = typename TestFixture::Type;
    Type expectedResData[] = {
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
    Type volData[] = {
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
    Type strelData[] = {
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
    Type resData[5 * 5 * 5] = { Type(0) };

    int3 volSize = make_int3(5, 5, 5);
    int3 strelSize = make_int3(3, 3, 3);

    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> res(resData, volSize);
    HostView<const Type> vol(volData, volSize);
    HostView<const Type> strel(strelData, strelSize);

    try {
        genDilateErode<MORPH_DILATE, Type>(res, vol, strel, make_int3(2, 2, 2));
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
    this->syncAndAssertCudaSuccess();

    EXPECT_VOL_EQ(expectedRes, res);

    this->assertCudaSuccess();
}

template <class Ty>
class GeneralMorphEmulateFlatTest : public GeneralMorphTest<Ty> {};

// We only test this part for signed integer and float types, since we cannot emulate flat structuring
// elements when using unigned integer types.
TYPED_TEST_SUITE(GeneralMorphEmulateFlatTest, SignedIntAndFloatTypes);

TYPED_TEST(GeneralMorphEmulateFlatTest, Dilate)
{
    using Type = typename TestFixture::Type;
    const Type minVal = kernel::minusInfOrMin<Type>();
    Type expectedResData[] = {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };
    Type volData[] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    Type strelData[] = {
        minVal, 0, minVal,
        0, 0, 0,
        minVal, 0, minVal
    };
    Type resData[3 * 3 * 1] = { Type(0) };

    int3 volSize = make_int3(3, 3, 1);
    int3 strelSize = make_int3(3, 3, 1);

    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> res(resData, volSize);
    HostView<const Type> vol(volData, volSize);
    HostView<const Type> strel(strelData, strelSize);

    try {
        genDilateErode<MORPH_DILATE, Type>(res, vol, strel, make_int3(3, 3, 1));
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }

    this->syncAndAssertCudaSuccess();

    EXPECT_VOL_EQ(expectedRes, res);

    this->assertCudaSuccess();
}

TYPED_TEST(GeneralMorphEmulateFlatTest, Erode)
{
    using Type = typename TestFixture::Type;
    const Type minVal = kernel::minusInfOrMin<Type>();
    Type expectedResData[] = {
        1, 0, 1,
        0, 0, 0,
        1, 0, 1
    };
    Type volData[] = {
        1, 1, 1,
        1, 0, 1,
        1, 1, 1
    };
    Type strelData[] = {
        minVal, 0, minVal,
        0, 0, 0,
        minVal, 0, minVal
    };
    Type resData[3 * 3 * 1] = { Type(0) };

    int3 volSize = make_int3(3, 3, 1);
    int3 strelSize = make_int3(3, 3, 1);

    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> res(resData, volSize);
    HostView<const Type> vol(volData, volSize);
    HostView<const Type> strel(strelData, strelSize);

    try {
        genDilateErode<MORPH_ERODE, Type>(res, vol, strel, make_int3(3, 3, 1));
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }

    this->syncAndAssertCudaSuccess();

    EXPECT_VOL_EQ(expectedRes, res);

    this->assertCudaSuccess();
}
