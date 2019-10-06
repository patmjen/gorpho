#ifndef CUSATARIT_CUH
#define CUSATARIT_CUH

#include <inttypes.h>
#include <type_traits>
#include <limits>

namespace csa {

namespace detail {

// Helper values for checking for signed- and unsigned integers

template <class Ty>
constexpr bool isUnsignedInt = std::is_integral<Ty>::value && !std::is_signed<Ty>::value;

template <class Ty>
constexpr bool isSignedInt = std::is_integral<Ty>::value && std::is_signed<Ty>::value;

} // namespace detail

// Functions from: http://locklessinc.com/articles/sat_arithmetic/

template <class Ty>
__host__ __device__
inline typename std::enable_if<detail::isUnsignedInt<Ty>, Ty>::type satPlus(Ty x, Ty y)
{
    Ty res = x + y;
    return res | -(res < x);
}

template <class Ty>
__host__ __device__
inline typename std::enable_if<detail::isUnsignedInt<Ty>, Ty>::type satMinus(Ty x, Ty y)
{
    Ty res = x - y;
    return res & -(res <= x);
}

template <class Ty>
__host__ __device__
inline typename std::enable_if<detail::isSignedInt<Ty>, Ty>::type satPlus(Ty x, Ty y)
{
    using numLim = std::numeric_limits<Ty>;
    typename std::make_unsigned<Ty>::type ux = x;
    typename std::make_unsigned<Ty>::type uy = y;
    typename std::make_unsigned<Ty>::type res = ux + uy;

    ux = (ux >> numLim::digits) + numLim::max();

    if ((Ty) ((ux ^ uy) | ~(uy ^ res)) >= 0) {
        res = ux;
    }

    return res;
}

template <class Ty>
__host__ __device__
inline typename std::enable_if<detail::isSignedInt<Ty>, Ty>::type satMinus(Ty x, Ty y)
{
    using numLim = std::numeric_limits<Ty>;
    typename std::make_unsigned<Ty>::type ux = x;
    typename std::make_unsigned<Ty>::type uy = y;
    typename std::make_unsigned<Ty>::type res = ux - uy;

    ux = (ux >> numLim::digits) + numLim::max();

    if ((Ty)((ux ^ uy) & (ux ^ res)) < 0) {
        res = ux;
    }

    return res;
}

#ifndef CUSATARIT_DISABLE_FLOAT
template <class Ty>
__host__ __device__
inline typename std::enable_if<std::is_floating_point<Ty>::value, Ty>::type satPlus(Ty x, Ty y)
{
    return x + y;
}

template <class Ty>
__host__ __device__
inline typename std::enable_if<std::is_floating_point<Ty>::value, Ty>::type satMinus(Ty x, Ty y)
{
    return x - y;
}
#endif

template <class Ty, class Enable=void>
struct Saturate;

template <class Ty>
struct Saturate<Ty, typename std::enable_if<std::is_arithmetic<Ty>::value>::type> {
    typedef Ty ValueType;
    typedef Saturate<Ty> Self;

    Ty val;

    explicit Saturate() = default;
    __host__ __device__
    Saturate(Ty val) : val(val) {};

    __host__ __device__
    Self operator+=(Self rhs) {
        val = satPlus(val, rhs.val);
        return *this;
    }

    __host__ __device__
    Self operator-=(Self rhs) {
        val = satMinus(val, rhs.val);
        return *this;
    }

    __host__ __device__
    bool operator==(Self rhs) {
        return val == rhs.val;
    }
};

template <class Ty>
__host__ __device__
Saturate<Ty> operator+(Saturate<Ty> lhs, Saturate<Ty> rhs)
{
    lhs += rhs;
    return lhs;
}

// TODO: Find a better way to do this
template <class Ty>
__host__ __device__
Saturate<Ty> operator+(Ty lhs, Saturate<Ty> rhs)
{
    return Saturate<Ty>(lhs) + rhs;
}

template <class Ty>
__host__ __device__
Saturate<Ty> operator+(Saturate<Ty> lhs, Ty rhs)
{
    return lhs + Saturate<Ty>(rhs);
}

template <class Ty>
__host__ __device__
Saturate<Ty> operator-(Saturate<Ty> lhs, Saturate<Ty> rhs)
{
    lhs -= rhs;
    return lhs;
}

// TODO: Find a better way to do this
template <class Ty>
__host__ __device__
Saturate<Ty> operator-(Ty lhs, Saturate<Ty> rhs)
{
    return Saturate<Ty>(lhs) - rhs;
}

template <class Ty>
__host__ __device__
Saturate<Ty> operator-(Saturate<Ty> lhs, Ty rhs)
{
    return lhs - Saturate<Ty>(rhs);
}

} // namespace csa

#endif /* CUSATARIT_CUH */
