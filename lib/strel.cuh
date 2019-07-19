#ifndef STREL_CUH__
#define STREL_CUH__

#include <vector>

namespace gpho {

struct LineSeg {
	int3 step;
	int numSteps;

	LineSeg() = default;
	LineSeg(int3 step, int numSteps) :
		step(step),
		numSteps(numSteps) {}
};

} // namespace gpho

#endif // STREL_CUH__