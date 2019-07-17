#ifndef MORPH_CUH__
#define MORPH_CUH__

#include <functional>

namespace gpho {

enum MorphOp {
	MORPH_DILATE,
	MORPH_ERODE
};

} // namespace gpho

#include "general_morph.cuh"
#include "flat_linear_morph.cuh"

#endif // MORPH_CUH__