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

template <class Ty>
class FlatLinearMorphTest : public CudaTest {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(FlatLinearMorphTest, AllPodTypes);

template <class Ty>
void performLinesTest(HostView<Ty> expectedRes, HostView<const Ty> vol, const std::vector<LineSeg>& lines,
    bool onlyUseFirstLineSeg = false)
{
    int bufSize = static_cast<int>(minTotalBufferSize(minRSBufferSize(lines), vol.size()));
    DeviceVolume<Ty> dvol, dres, rBuffer, sBuffer, dresBuffer;
    ASSERT_NO_THROW(dvol = makeDeviceVolume<Ty>(vol.size()));
    ASSERT_NO_THROW(dres = makeDeviceVolume<Ty>(vol.size()));
    ASSERT_NO_THROW(rBuffer = makeDeviceVolume<Ty>(bufSize, 1, 1));
    ASSERT_NO_THROW(sBuffer = makeDeviceVolume<Ty>(bufSize, 1, 1));

    transfer(dvol.view(), vol);

    if (onlyUseFirstLineSeg) {
        flatLinearDilateErode<MORPH_DILATE, Ty>(dres, dvol, rBuffer, sBuffer, lines[0]);
    } else {
        ASSERT_NO_THROW(dresBuffer = makeDeviceVolume<Ty>(vol.size()));
        flatLinearDilateErode<MORPH_DILATE, Ty>(dres, dresBuffer, dvol, rBuffer, sBuffer, lines);
    }
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    HostVolume<Ty> res;
    ASSERT_NO_THROW(res = dres.copyToHost());

    EXPECT_VOL_EQ(expectedRes, res.view());
}

template <class Ty>
void performSingleLineTest(HostView<Ty> expectedRes, HostView<const Ty> vol, LineSeg line)
{
    performLinesTest<Ty>(expectedRes, vol, { line }, true);
}

TYPED_TEST(FlatLinearMorphTest, SingleLineAxisAligned)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(7, 3, 3);
    Type expectedResData[7 * 3 * 3];
    Type volData[7 * 3 * 3];
    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);

    {
        SCOPED_TRACE("Line aligned with 1st axis");

        std::fill(vol.data(), vol.data() + vol.numel(), 0);
        vol[make_int3(3, 1, 1)] = Type(1);

        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        for (int i = 1; i < expectedRes.size().x - 1; ++i) {
            expectedRes[make_int3(i, 1, 1)] = Type(1);
        }
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(1, 0, 0), 5));
    }
    {
        SCOPED_TRACE("Line aligned with 2nd axis");
        vol.reshape(3, 7, 3);
        std::fill(vol.data(), vol.data() + vol.numel(), 0);
        vol[make_int3(1, 3, 1)] = 1;

        expectedRes.reshape(3, 7, 3);
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        for (int i = 1; i < expectedRes.size().y - 1; ++i) {
            expectedRes[make_int3(1, i, 1)] = Type(1);
        }
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(0, 1, 0), 5));
    }
    {
        SCOPED_TRACE("Line aligned with 3rd axis");
        vol.reshape(3, 3, 7);
        std::fill(vol.data(), vol.data() + vol.numel(), 0);
        vol[make_int3(1, 1, 3)] = 1;

        expectedRes.reshape(3, 3, 7);
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        for (int i = 1; i < expectedRes.size().z - 1; ++i) {
            expectedRes[make_int3(1, 1, i)] = Type(1);
        }
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(0, 0, 1), 5));
    }

    assertCudaSuccess();
}

TYPED_TEST(FlatLinearMorphTest, SingleLineAxisAlignedEvenLength)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(5, 3, 3);
    Type expectedResData[5 * 3 * 3];
    Type volData[5 * 3 * 3] = { Type(0) };
    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);

    vol[make_int3(2, 1, 1)] = Type(1);

    {
        SCOPED_TRACE("Step = (1, 0, 0)");
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        expectedRes[make_int3(2, 1, 1)] = Type(1);
        expectedRes[make_int3(3, 1, 1)] = Type(1);
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(1, 0, 0), 2));
    }
    {
        SCOPED_TRACE("Step = (-1, 0, 0)");
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        expectedRes[make_int3(2, 1, 1)] = Type(1);
        expectedRes[make_int3(1, 1, 1)] = Type(1);
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(-1, 0, 0), 2));
    }

    assertCudaSuccess();
}

TYPED_TEST(FlatLinearMorphTest, SingleLineNotAxisAligned)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(5, 5, 5);
    Type expectedResData[5 * 5 * 5];
    Type volData[5 * 5 * 5] = { Type(0) };
    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);

    vol[make_int3(2, 2, 2)] = Type(1);

    {
        SCOPED_TRACE("Step = (1, 1, 0)");
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        for (int i = 1; i < 4; ++i) {
            expectedRes[make_int3(i, i, 2)] = Type(1);
        }
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(1, 1, 0), 3));
    }
    {
        SCOPED_TRACE("Step = (1, 1, 1)");
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        for (int i = 1; i < 4; ++i) {
            expectedRes[make_int3(i, i, i)] = Type(1);
        }
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(1, 1, 1), 3));
    }
    {
        SCOPED_TRACE("Step = (-1, 1, 0)");
        std::fill(expectedRes.data(), expectedRes.data() + expectedRes.numel(), 0);
        for (int i = 1; i < 4; ++i) {
            expectedRes[make_int3(4 - i, i, 2)] = Type(1);
        }
        performSingleLineTest<Type>(expectedRes, vol, LineSeg(make_int3(-1, 1, 0), 3));
    }

    assertCudaSuccess();
}

TYPED_TEST(FlatLinearMorphTest, MutipleLines)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(9, 9, 9);
    Type expectedResData[9 * 9 * 9] = { Type(0) };
    Type volData[9 * 9 * 9] = { Type(0) };
    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);

    vol[make_int3(4, 4, 4)] = Type(1);

    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 5; ++y) {
            for (int z = 0; z < 7; ++z) {
                expectedRes[make_int3(x + 3, y + 2, z + 1)] = Type(1);
            }
        }
    }
    std::vector<LineSeg> lines = {
        LineSeg(make_int3(1, 0, 0), 3),
        LineSeg(make_int3(0, 1, 0), 5),
        LineSeg(make_int3(0, 0, 1), 7)
    };

    performLinesTest<Type>(expectedRes, vol, lines);

    assertCudaSuccess();
}

TYPED_TEST(FlatLinearMorphTest, HostInput)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(9, 9, 9);
    Type expectedResData[9 * 9 * 9] = { Type(0) };
    Type volData[9 * 9 * 9] = { Type(0) };
    Type resData[9 * 9 * 9] = { Type(0) };
    HostView<Type> expectedRes(expectedResData, volSize);
    HostView<Type> vol(volData, volSize);
    HostView<Type> res(resData, volSize);

    vol[make_int3(4, 4, 4)] = Type(1);

    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 5; ++y) {
            for (int z = 0; z < 7; ++z) {
                expectedRes[make_int3(x + 3, y + 2, z + 1)] = Type(1);
            }
        }
    }
    std::vector<LineSeg> lines = {
        LineSeg(make_int3(1, 0, 0), 3),
        LineSeg(make_int3(0, 1, 0), 5),
        LineSeg(make_int3(0, 0, 1), 7)
    };

    try {
        flatLinearDilateErode<MORPH_DILATE, Type>(res, vol, lines, make_int3(2, 2, 2));
    } catch (const std::exception e) {
        FAIL() << e.what();
    } catch (...) {
        FAIL();
    }
    syncAndAssertCudaSuccess();

    ASSERT_VOL_EQ(expectedRes, res);
}

TYPED_TEST(FlatLinearMorphTest, EmptyOp)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(3, 3, 3);
    Type volData[3 * 3 * 3] = { Type(0) };
    HostView<Type> vol(volData, volSize);

    vol[make_int3(1, 1, 1)] = 1;

    {
        SCOPED_TRACE("Step = (0, 0, 0)");
        performSingleLineTest<Type>(vol, vol, LineSeg(make_int3(0, 0, 0), 5));
    }
    {
        SCOPED_TRACE("Length = 1");
        performSingleLineTest<Type>(vol, vol, LineSeg(make_int3(1, 0, 0), 1));
    }
    {
        SCOPED_TRACE("Length = 0");
        performSingleLineTest<Type>(vol, vol, LineSeg(make_int3(1, 0, 0), 0));
    }
}

TYPED_TEST(FlatLinearMorphTest, MultipleEmptyOp)
{
    using Type = typename TestFixture::Type;
    int3 volSize = make_int3(5, 5, 5);
    Type volData[5 * 5 * 5] = { Type(0) };
    Type expectedResData[5 * 5 * 5] = { Type(0) };
    HostView<Type> vol(volData, volSize);
    HostView<Type> expectedRes(expectedResData, volSize);

    vol[make_int3(2, 2, 2)] = Type(1);

    {
        SCOPED_TRACE("All empty steps");
        // For this case, the output should be same as the input
        performLinesTest<Type>(vol, vol, {
            LineSeg(make_int3(0, 0, 0), 5),
            LineSeg(make_int3(1, 0, 0), 0)
        });
    }
    {
        SCOPED_TRACE("One empty step");
        for (int x = 1; x < 4; ++x) {
            for (int y = 1; y < 4; ++y) {
                expectedRes[make_int3(x, y, 2)] = Type(1);
            }
        }
        performLinesTest<Type>(expectedRes, vol, {
            LineSeg(make_int3(1, 0, 0), 3),
            LineSeg(make_int3(0, 0, 1), 0),
            LineSeg(make_int3(0, 1, 0), 3)
        });
    }
}