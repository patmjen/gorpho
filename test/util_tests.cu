#include "gtest/gtest.h"

#include "test_util.cuh"
#include "util.cuh"

// For convenience
using namespace gpho;

template <class Ty>
class ZeroOrOneTest : public ::testing::Test {
public:
    using Type = Ty;
};

TYPED_TEST_SUITE(ZeroOrOneTest, AllPodTypes);

TYPED_TEST(ZeroOrOneTest, DeviceInput)
{
    using Type = typename TestFixture::Type;
    ASSERT_EQ(Type(1), zeroOrOne<Type>(true));
    ASSERT_EQ(Type(0), zeroOrOne<Type>(false));
}