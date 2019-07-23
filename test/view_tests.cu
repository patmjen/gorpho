#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "view.cuh"
#include "volume.cuh"
#include "test_util.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

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

TEST(SizedBaseTest, Comparison)
{
    int3 size1 = make_int3(2, 3, 5);
    int3 size2 = make_int3(7, 1, 8);
    SizedBase sb1(size1);
    SizedBase sb2(size1);
    SizedBase sb3(size2);
    ASSERT_EQ(sb1.size(), sb2.size()) << "Test pre-condition";
    ASSERT_NE(sb1.size(), sb3.size()) << "Test pre-condition";

    EXPECT_EQ(sb1, sb1);
    EXPECT_EQ(sb1, sb2);
    EXPECT_NE(sb1, sb3);
}

template <class Vw>
class AllViewsTest : public ::testing::Test {
public:
    using View = Vw;
};

// TODO: Also parameterize on the type contained in the volumes
using AllViewTypes = ::testing::Types<ViewBase<float>, DeviceView<float>, HostView<float>, PinnedView<float>>;
TYPED_TEST_SUITE(AllViewsTest, AllViewTypes);

TYPED_TEST(AllViewsTest, DefaultInit)
{
    typename TestFixture::View vw;
    EXPECT_EQ(vw.data(), nullptr);
    EXPECT_EQ(vw.size(), int3_0);
}

TYPED_TEST(AllViewsTest, PtrConstructor)
{
    float x = 1.4f;
    int3 size = make_int3(3, 1, 2);
    typename TestFixture::View vw1(&x, size);
    EXPECT_EQ(vw1.data(), &x);
    EXPECT_EQ(*vw1.data(), x);
    EXPECT_EQ(vw1.size(), size);

    typename TestFixture::View vw2(&x, size.x, size.y, size.z);
    EXPECT_EQ(vw2.data(), &x);
    EXPECT_EQ(*vw2.data(), x);
    EXPECT_EQ(vw2.size(), size);
}

TYPED_TEST(AllViewsTest, ModifyValue)
{
    float x = 1.4f;
    float newX = 2.3f;
    int3 size = int3_1;
    typename TestFixture::View vw(&x, size);
    ASSERT_EQ(*vw.data(), x);

    *vw.data() = newX;
    EXPECT_EQ(*vw.data(), newX);
    EXPECT_EQ(x, newX);
}

TYPED_TEST(AllViewsTest, NonConstToConstView)
{
    float x = 1.2f;
    float *p = &x;
    typename TestFixture::View vw1(p, int3_1);
    typename TestFixture::View::ConstView vw2(vw1);

    EXPECT_EQ(vw2.data(), vw2.data());
    EXPECT_EQ(vw2.size(), vw2.size());
    EXPECT_EQ(*vw1.data(), x);
    EXPECT_EQ(*vw2.data(), x);
}

TYPED_TEST(AllViewsTest, Comparison)
{
    int3 size1 = make_int3(2, 3, 5);
    int3 size2 = make_int3(7, 1, 8);
    float x1 = 1.2f;
    float x2 = 3.1f;
    typename TestFixture::View vw1(&x1, size1);
    typename TestFixture::View vw2(&x1, size1);
    typename TestFixture::View vw3(&x1, size2);
    typename TestFixture::View vw4(&x2, size1);
    typename TestFixture::View vw5(&x2, size2);

    EXPECT_EQ(vw1, vw1);
    EXPECT_EQ(vw1, vw2);
    EXPECT_NE(vw1, vw3);
    EXPECT_NE(vw1, vw4);
    EXPECT_NE(vw1, vw5);
}

TYPED_TEST(AllViewsTest, Indexing)
{
    int3 size = make_int3(2, 3, 4);
    float data[2 * 3 * 4] = { 0 };
    for (int i = 0; i < prod(size); ++i) {
        data[i] = i;
    }

    typename TestFixture::View vw(data, size);
    for (int i = 0; i < vw.numel(); ++i) {
        EXPECT_EQ(vw[i], i);
    }
    for (int x = 0; x < vw.size().x; ++x) {
        for (int y = 0; y < vw.size().y; ++y) {
            for (int z = 0; z < vw.size().z; ++z) {
                EXPECT_EQ(vw[make_int3(x, y, z)], vw.idx(x, y, z));
            }
        }
    }
}