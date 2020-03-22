#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "volume.cuh"
#include "view.cuh"
#include "misc.cuh"
#include "util.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

template <class Ty>
class ElemWiseOpTest : public CudaTest {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(ElemWiseOpTest, AllPodTypes);

template <MathOp op, class Ty>
void computeExpectedElemWiseResult(HostView<Ty> expectedRes, HostView<const Ty> a, HostView<const Ty> b)
{
    for (int i = 0; i < a.numel(); ++i) {
        switch (op) {
        case MATH_ADD:
            expectedRes[i] = a[i] + b[i];
            break;
        case MATH_SUB:
            expectedRes[i] = a[i] - b[i];
            break;
        case MATH_MUL:
            expectedRes[i] = a[i] * b[i];
            break;
        case MATH_DIV:
            expectedRes[i] = a[i] / b[i];
            break;
        }
    }
}

template <class Func, class Ty>
void performElemWiseTest(Func func, HostVolume<Ty>& res, HostVolume<Ty>& a, HostVolume<Ty>& b)
{
    DeviceVolume<Ty> da, db, dres;
    ASSERT_NO_THROW(dres = makeDeviceVolume<Ty>(res.size()));    
    ASSERT_NO_THROW(da = a.copyToDevice());
    ASSERT_NO_THROW(db = b.copyToDevice());

    func(dres, da, db);
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    ASSERT_NO_THROW(dres.copyToHost(res));
    ASSERT_NO_THROW(da.copyToHost(a));
    ASSERT_NO_THROW(db.copyToHost(b));
}

// TODO: These tests are pretty much the same. Refactor so code isn't repeated as much

TYPED_TEST(ElemWiseOpTest, AllSeperateVols)
{
    using Type = typename TestFixture::Type;
    const int3 volSize = make_int3(5, 5, 5);
    HostVolume<Type> a, b, res, expectedRes;

    ASSERT_NO_THROW(a = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(b = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(res = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(expectedRes = makeHostVolume<Type>(volSize));

    srand(7); // Lucky number seven...
    randomFill<Type>(a);
    randomFill<Type>(b);
    // Dividing by zero causes SEH exceptions which make the tests fail unnecessarily
    std::replace(b.data(), b.data() + b.numel(), 0, 2);

    {
        SCOPED_TRACE("Add");
        computeExpectedElemWiseResult<MATH_ADD, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_ADD, Type>(dres, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
    {
        SCOPED_TRACE("Sub");
        computeExpectedElemWiseResult<MATH_SUB, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_SUB, Type>(dres, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
    {
        SCOPED_TRACE("Mul");
        computeExpectedElemWiseResult<MATH_MUL, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_MUL, Type>(dres, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
    {
        SCOPED_TRACE("Div");
        computeExpectedElemWiseResult<MATH_DIV, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_DIV, Type>(dres, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), res.view());
    }
}

TYPED_TEST(ElemWiseOpTest, UseAAsRes)
{
    using Type = typename TestFixture::Type;
    const int3 volSize = make_int3(5, 5, 5);
    HostVolume<Type> a, b, res, expectedRes;

    ASSERT_NO_THROW(a = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(b = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(res = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(expectedRes = makeHostVolume<Type>(volSize));

    auto initVols = [&]()
    {
        srand(7); // Lucky number seven...
        randomFill<Type>(a);
        randomFill<Type>(b);
        // Dividing by zero causes SEH exceptions which make the tests fail unnecessarily
        std::replace(b.data(), b.data() + b.numel(), 0, 2);
    };

    {
        SCOPED_TRACE("Add");
        initVols();
        computeExpectedElemWiseResult<MATH_ADD, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_ADD, Type>(da, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), a.view());
    }
    {
        SCOPED_TRACE("Sub");
        initVols();
        computeExpectedElemWiseResult<MATH_SUB, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_SUB, Type>(da, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), a.view());
    }
    {
        SCOPED_TRACE("Mul");
        initVols();
        computeExpectedElemWiseResult<MATH_MUL, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_MUL, Type>(da, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), a.view());
    }
    {
        SCOPED_TRACE("Div");
        initVols();
        computeExpectedElemWiseResult<MATH_DIV, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_DIV, Type>(da, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), a.view());
    }
}

TYPED_TEST(ElemWiseOpTest, UseBAsRes)
{
    using Type = typename TestFixture::Type;
    const int3 volSize = make_int3(5, 5, 5);
    HostVolume<Type> a, b, res, expectedRes;

    ASSERT_NO_THROW(a = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(b = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(res = makeHostVolume<Type>(volSize));
    ASSERT_NO_THROW(expectedRes = makeHostVolume<Type>(volSize));

    auto initVols = [&]()
    {
        srand(7); // Lucky number seven...
        randomFill<Type>(a);
        randomFill<Type>(b);
        // Dividing by zero causes SEH exceptions which make the tests fail unnecessarily
        std::replace(b.data(), b.data() + b.numel(), 0, 2);
    };

    {
        SCOPED_TRACE("Add");
        initVols();
        computeExpectedElemWiseResult<MATH_ADD, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_ADD, Type>(db, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), b.view());
    }
    {
        SCOPED_TRACE("Sub");
        initVols();
        computeExpectedElemWiseResult<MATH_SUB, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_SUB, Type>(db, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), b.view());
    }
    {
        SCOPED_TRACE("Mul");
        initVols();
        computeExpectedElemWiseResult<MATH_MUL, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_MUL, Type>(db, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), b.view());
    }
    {
        SCOPED_TRACE("Div");
        initVols();
        computeExpectedElemWiseResult<MATH_DIV, Type>(expectedRes, a, b);
        performElemWiseTest([](auto dres, auto da, auto db) { elemWiseOp<MATH_DIV, Type>(db, da, db); },
            res, a, b);
        if (this->HasFatalFailure()) return;
        EXPECT_VOL_EQ(expectedRes.view(), b.view());
    }
}