#ifndef TEST_UTIL_CUH__
#define TEST_UTIL_CUH__

namespace gpho {

namespace test {

const int3 int3_0 = make_int3(0, 0, 0);
const int3 int3_1 = make_int3(1, 1, 1);

template <class Ty> void nonDeleter(Ty *x){};

} // namespace test

} // namespace gpho

#endif // TEST_UTIL_CUH__