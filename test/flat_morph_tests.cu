#include <cuda_runtime.h>
#include <stdexcept>
#include <cinttypes>
#include <cstdlib>

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
class FlatDilateErodeTest : public CudaTest {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(FlatDilateErodeTest, AllPodTypes);

TYPED_TEST(FlatDilateErodeTest, DeviceInput)
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

TYPED_TEST(FlatDilateErodeTest, HostInput)
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

// NOTE: For opening and closing we only test that open(...) = dilate(erode(...)) and vice versa for close.
template <class Ty>
class FlatOpenCloseTest : public CudaTest {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(FlatOpenCloseTest, AllPodTypes);

template <MorphOp op, class Ty>
void computeExpectedOpenCloseRes(const DeviceView<Ty>& dvol, const DeviceView<bool>& dstrel,
    HostVolume<Ty>& expectedRes, HostVolume<Ty>& expectedResBuffer)
{
    static_assert(op == MORPH_OPEN || op == MORPH_CLOSE, "op must be MORPH_OPEN or MORPH_CLOSE");
    DeviceVolume<Ty> dres, dresBuffer;
    ASSERT_NO_THROW(dres = makeDeviceVolume<Ty>(dvol.size()));
    ASSERT_NO_THROW(dresBuffer = makeDeviceVolume<Ty>(dvol.size()));

    if (op == MORPH_OPEN) {
        flatErode<Ty>(dresBuffer, dvol, dstrel);
        flatDilate<Ty>(dres, dresBuffer, dstrel);
    } else {
        flatDilate<Ty>(dresBuffer, dvol, dstrel);
        flatErode<Ty>(dres, dresBuffer, dstrel);
    }

    expectedRes = dres.copyToHost();
    expectedResBuffer = dresBuffer.copyToHost();
}

template <MorphOp op, class Ty>
void computeExpectedOpenCloseRes(const HostVolume<Ty>& vol, const HostVolume<bool>& strel,
    HostVolume<Ty>& expectedRes)
{
    static_assert(op == MORPH_OPEN || op == MORPH_CLOSE, "op must be MORPH_OPEN or MORPH_CLOSE");
    DeviceVolume<Ty> dvol;
    DeviceVolume<bool> dstrel;
    HostVolume<Ty> ignored;

    ASSERT_NO_THROW(dvol = vol.copyToDevice());
    ASSERT_NO_THROW(dstrel = strel.copyToDevice());
    computeExpectedOpenCloseRes<op, Ty>(dvol, dstrel, expectedRes, ignored);
}

TYPED_TEST(FlatOpenCloseTest, DeviceInput)
{
    // TODO: This test is very large and should probably be spread into smaller pieces.
    using Type = typename TestFixture::Type;
    
    const int3 volSize = make_int3(8, 8, 8);
    const int3 strelSize = make_int3(3, 3, 3);

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

    HostVolume<Type> vol;
    HostVolume<bool> strel;
    ASSERT_NO_THROW(vol = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(strel = makeHostVolume<bool>(strelSize));

    std::srand(7); // Lucky number seven...
    std::generate(vol.data(), vol.data() + vol.numel(), []() {
        return static_cast<Type>(100.0 * static_cast<double>(std::rand()) / RAND_MAX); });
    std::memcpy(strel.data(), strelData, strel.numel() * sizeof(bool));

    DeviceVolume<Type> dvol, dres, dresBuffer;
    DeviceVolume<bool> dstrel;
    HostVolume<Type> expectedRes, expectedResBuffer;
    HostVolume<Type> res, resBuffer;
    
    ASSERT_NO_THROW(dres = makeDeviceVolume<Type>(vol.size()));
    ASSERT_NO_THROW(dresBuffer = makeDeviceVolume<Type>(vol.size()));

    auto initTest = [&](MorphOp op)
    {
        // This always resets dresBuffer even though it might not be needed. However, the overhead
        // is negligible so the code clarity is worth it.
        ASSERT_NO_THROW(dvol = vol.copyToDevice());
        ASSERT_NO_THROW(dstrel = strel.copyToDevice());
        ASSERT_CUDA_SUCCESS(cudaMemset(dres.data(), 0, dres.numel() * sizeof(Type)));
        ASSERT_CUDA_SUCCESS(cudaMemset(dresBuffer.data(), 0, dresBuffer.numel() * sizeof(Type)));
        if (op == MORPH_OPEN) {
            computeExpectedOpenCloseRes<MORPH_OPEN, Type>(dvol, dstrel, expectedRes, expectedResBuffer);
        } else {
            computeExpectedOpenCloseRes<MORPH_CLOSE, Type>(dvol, dstrel, expectedRes, expectedResBuffer);
        }
    };

    auto checkResult = [&](auto dres, auto dresBuffer)
    {
        ASSERT_NO_THROW(res = dres.copyToHost());
        ASSERT_NO_THROW(resBuffer = dresBuffer.copyToHost());
        EXPECT_VOL_EQ(expectedRes.view(), res.view());
        EXPECT_VOL_EQ(expectedResBuffer.view(), resBuffer.view());
    };

    {
        SCOPED_TRACE("Open with separate resBuffer");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_OPEN));

        flatOpen<Type>(dres, dresBuffer, dvol, dstrel);
        this->syncAndAssertCudaSuccess();

        ASSERT_NO_FATAL_FAILURE(checkResult(dres, dresBuffer));
    }
    {
        SCOPED_TRACE("Close with separate resBuffer");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_CLOSE));

        flatClose<Type>(dres, dresBuffer, dvol, dstrel);
        this->syncAndAssertCudaSuccess();

        ASSERT_NO_FATAL_FAILURE(checkResult(dres, dresBuffer));
    }
    {
        SCOPED_TRACE("Open with vol as buffer");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_OPEN));

        flatOpen<Type>(dres, dvol, dstrel);
        this->syncAndAssertCudaSuccess();

        ASSERT_NO_FATAL_FAILURE(checkResult(dres, dvol));
    }
    {
        SCOPED_TRACE("Close with vol as buffer");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_CLOSE));

        flatClose<Type>(dres, dvol, dstrel);
        this->syncAndAssertCudaSuccess();

        ASSERT_NO_FATAL_FAILURE(checkResult(dres, dvol));
    }
}

TYPED_TEST(FlatOpenCloseTest, HostInputSingleBlock)
{
    // TODO: This test is very large and should probably be spread into smaller pieces.
    using Type = typename TestFixture::Type;
    
    const int3 volSize = make_int3(8, 8, 8);
    const int3 strelSize = make_int3(3, 3, 3);

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

    HostVolume<Type> vol, res, expectedRes;
    HostVolume<bool> strel;
    ASSERT_NO_THROW(vol = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(res = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(strel = makeHostVolume<bool>(strelSize));

    std::srand(7); // Lucky number seven...
    std::generate(vol.data(), vol.data() + vol.numel(), []() {
        return static_cast<Type>(100.0 * static_cast<double>(std::rand()) / RAND_MAX); });
    std::memcpy(strel.data(), strelData, strel.numel() * sizeof(bool));

    auto initTest = [&](MorphOp op)
    {
        // This always resets dresBuffer even though it might not be needed. However, the overhead
        // is negligible so the code clarity is worth it.
        ASSERT_NO_THROW(std::memset(res.data(), 0, res.numel() * sizeof(Type)));
        if (op == MORPH_OPEN) {
            computeExpectedOpenCloseRes<MORPH_OPEN, Type>(vol, strel, expectedRes);
        } else {
            computeExpectedOpenCloseRes<MORPH_CLOSE, Type>(vol, strel, expectedRes);
        }
    };

    {
        SCOPED_TRACE("Open");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_OPEN));

        flatOpen<Type>(res, vol, strel, 2 * volSize);
        this->syncAndAssertCudaSuccess();

        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
    {
        SCOPED_TRACE("Close");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_CLOSE));

        flatClose<Type>(res, vol, strel, 2 * volSize);
        this->syncAndAssertCudaSuccess();

        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
}

TYPED_TEST(FlatOpenCloseTest, HostInputMultipleBlocks)
{
    // TODO: This test is very large and should probably be spread into smaller pieces.
    using Type = typename TestFixture::Type;
    
    const int3 volSize = make_int3(8, 8, 8);
    const int3 strelSize = make_int3(3, 3, 3);

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

    HostVolume<Type> vol, res, expectedRes;
    HostVolume<bool> strel;
    ASSERT_NO_THROW(vol = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(res = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(strel = makeHostVolume<bool>(strelSize));

    std::srand(7); // Lucky number seven...
    std::generate(vol.data(), vol.data() + vol.numel(), []() {
        return static_cast<Type>(100.0 * static_cast<double>(std::rand()) / RAND_MAX); });
    std::memcpy(strel.data(), strelData, strel.numel() * sizeof(bool));

    auto initTest = [&](MorphOp op)
    {
        // This always resets dresBuffer even though it might not be needed. However, the overhead
        // is negligible so the code clarity is worth it.
        ASSERT_NO_THROW(std::memset(res.data(), 0, res.numel() * sizeof(Type)));
        if (op == MORPH_OPEN) {
            computeExpectedOpenCloseRes<MORPH_OPEN, Type>(vol, strel, expectedRes);
        } else {
            computeExpectedOpenCloseRes<MORPH_CLOSE, Type>(vol, strel, expectedRes);
        }
    };

    {
        SCOPED_TRACE("Open");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_OPEN));

        flatOpen<Type>(res, vol, strel, volSize / 2);
        this->syncAndAssertCudaSuccess();

        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
    {
        SCOPED_TRACE("Close");

        ASSERT_NO_FATAL_FAILURE(initTest(MORPH_CLOSE));

        flatClose<Type>(res, vol, strel, volSize / 2);
        this->syncAndAssertCudaSuccess();

        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
}