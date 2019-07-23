#ifndef MORPH_CUH__
#define MORPH_CUH__

#include <functional>

namespace gpho {

enum MorphOp {
    MORPH_DILATE,
    MORPH_ERODE
};

enum AxisDir : int {
    AXIS_DIR_1     = 0x0010,
    AXIS_DIR_1_NEG = 0x0011,
    AXIS_DIR_1_POS = 0x0012,
    AXIS_DIR_2     = 0x0100,
    AXIS_DIR_2_POS = 0x0101,
    AXIS_DIR_2_NEG = 0x0102,
    AXIS_DIR_3     = 0x1000,
    AXIS_DIR_3_POS = 0x1001,
    AXIS_DIR_3_NEG = 0x1002
};

} // namespace gpho

#include "strel.cuh"
#include "general_morph.cuh"
#include "flat_linear_morph.cuh"

#endif // MORPH_CUH__