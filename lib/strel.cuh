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

} // namespace gpho

#endif // STREL_CUH__