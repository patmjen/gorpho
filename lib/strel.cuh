#ifndef STREL_CUH__
#define STREL_CUH__

#include <vector>

namespace gpho {

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

} // namespace gpho

#endif // STREL_CUH__