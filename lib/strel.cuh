#ifndef STREL_CUH__
#define STREL_CUH__

#include <cuda_runtime.h>
#include <vector>

namespace gpho {

namespace detail {

inline int3 flatBallApproxZonoCoefs(int radius)
{
	static short coefs[3*1000] = {
#include "zono_ball_coefs.inl"
	};
	if (radius > 0) {
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

inline std::vector<LineSeg> flatBallApprox(int radius)
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
	int3 coefs = detail::flatBallApproxZonoCoefs(radius);
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