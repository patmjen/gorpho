#ifndef STREL_CUH__
#define STREL_CUH__

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

namespace gpho {

enum ApproxType {
    APPROX_INSIDE,
    APPROX_BEST,
    APPROX_OUTSIDE
};

namespace detail {

inline int3 flatBallApproxZonoCoefs(int radius, ApproxType type)
{
    static short coefs_inside[3 * 1000] = {
#include "zono_ball_coef_inside.inl"
    };
    static short coefs_best[3 * 1000] = {
#include "zono_ball_coef_best.inl"
    };
    static short coefs_outside[3 * 1000] = {
#include "zono_ball_coef_outside.inl"
    };
    if (radius > 0) {
        const short* coefs;
        switch (type) {
        case APPROX_INSIDE:
            coefs = coefs_inside;
            break;
        case APPROX_BEST:
            coefs = coefs_best;
            break;
        case APPROX_OUTSIDE:
            coefs = coefs_outside;
            break;
        default:
            throw std::invalid_argument("invalid approximation type");
        }

        return make_int3(
            static_cast<int>(coefs[3 * (radius - 1) + 0]),
            static_cast<int>(coefs[3 * (radius - 1) + 1]),
            static_cast<int>(coefs[3 * (radius - 1) + 2])
        );
    } else {
        return make_int3(0, 0, 0);
    }
}

} // namespace detail

struct LineSeg {
    int3 step;
    int length;

    LineSeg() = default;
    LineSeg(int3 step, int length) :
        step(step),
        length(length) {}
};

inline bool nonEmptyLineSeg(const LineSeg& line)
{
    return line.step != make_int3(0, 0, 0) && line.length > 1;
}

inline std::vector<LineSeg> flatBallApprox(int radius, ApproxType type = APPROX_BEST)
{
    static int3 steps[] = {
        { 1,  0,  0 },
        { 0, -1,  0 },
        { 0,  0,  1 },

        { 1,  1,  0 },
        {-1,  1,  0 },
        {-1,  0, -1 },
        { 1,  0, -1 },
        { 0,  1,  1 },
        { 0, -1,  1 },

        {-1, -1, -1 },
        { 1,  1, -1 },
        { 1, -1,  1 },
        {-1,  1,  1 }
    };
    int3 coefs = detail::flatBallApproxZonoCoefs(radius, type);
    return {
        LineSeg(steps[ 0], coefs.x),
        LineSeg(steps[ 1], coefs.x),
        LineSeg(steps[ 2], coefs.x),

        LineSeg(steps[ 3], coefs.y),
        LineSeg(steps[ 4], coefs.y),
        LineSeg(steps[ 5], coefs.y),
        LineSeg(steps[ 6], coefs.y),
        LineSeg(steps[ 7], coefs.y),
        LineSeg(steps[ 8], coefs.y),

        LineSeg(steps[ 9], coefs.z),
        LineSeg(steps[10], coefs.z),
        LineSeg(steps[11], coefs.z),
        LineSeg(steps[12], coefs.z)
    };
}

} // namespace gpho

#endif // STREL_CUH__