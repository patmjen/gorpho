#include <cuda_runtime.h>
#include <stdexcept>
#include <cinttypes>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "view.cuh"
#include "flat_morph.cuh"
#include "util.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

template <class Ty>
class FlatMorphTest : public CudaTest {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(FlatMorphTest, AllPodTypes);

TYPED_TEST(FlatMorphTest, DeviceInput)
{
    using Type = typename TestFixture::Type;
    Type expectedResData[] = {
        1, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 2, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 2, 0, 0,
        0, 2, 2, 2, 0
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
        0, 0, 2, 0, 0
    };
    bool strelData[] = {
        false, false, false,
        false, true, false,
        false, false, false,

        false, true, false,
        true, true, true,
        false, true, false,

        false, false, false,
        false, true, false,
        false, false, false
    };

    int3 volSize = make_int3(5, 5, 5);
    int3 strelSize = make_int3(3, 3, 3);

    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);
    HostView<bool> strel(strelData, strelSize);

    DeviceVolume<Type> dvol, dres;
    DeviceVolume<bool> dstrel;
    ASSERT_NO_THROW(dvol = makeDeviceVolume<Type>(volSize));
    ASSERT_NO_THROW(dres = makeDeviceVolume<Type>(volSize));
    ASSERT_NO_THROW(dstrel = makeDeviceVolume<bool>(strelSize));

    ASSERT_NO_THROW(transfer(dvol.view(), vol));
    ASSERT_NO_THROW(transfer(dstrel.view(), strel));

    flatDilateErode<MORPH_DILATE, Type>(dres, dvol, dstrel);
    this->syncAndAssertCudaSuccess();

    HostVolume<Type> res;
    ASSERT_NO_THROW(res = dres.copyToHost());

    EXPECT_VOL_EQ(expectedRes, res.view());

    this->assertCudaSuccess();
}

TYPED_TEST(FlatMorphTest, HostInput)
{
    using Type = typename TestFixture::Type;
    Type expectedResData[] = {
        1, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        1, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,

        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0
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
    bool strelData[] = {
        false, false, false,
        false, true, false,
        false, false, false,

        false, true, false,
        true, true, true,
        false, true, false,

        false, false, false,
        false, true, false,
        false, false, false
    };
    Type resData[5 * 5 * 5] = { Type(0) };

    int3 volSize = make_int3(5, 5, 5);
    int3 strelSize = make_int3(3, 3, 3);

    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> res(resData, volSize);
    HostView<const Type> vol(volData, volSize);
    HostView<const bool> strel(strelData, strelSize);

    try {
        flatDilateErode<MORPH_DILATE, Type>(res, vol, strel, make_int3(2, 2, 2));
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }
    this->syncAndAssertCudaSuccess();

    EXPECT_VOL_EQ(expectedRes, res);

    this->assertCudaSuccess();
}