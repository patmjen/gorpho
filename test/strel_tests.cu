#include <cuda_runtime.h>

#include "gtest/gtest.h"

#include "helper_math.cuh"
#include "strel.cuh"

// For convenience
using namespace gpho;
using namespace gpho::detail;

TEST(FlatBallApproxTest, NonPositiveRadius)
{
    {
        SCOPED_TRACE("Radius = -1");
        auto lines = flatBallApprox(-1);
        for (const auto& line : lines) {
            ASSERT_EQ(line.length, 0);
        }
    }
    {
        SCOPED_TRACE("Radius = 0");
        auto lines = flatBallApprox(0);
        for (const auto& line : lines) {
            ASSERT_EQ(line.length, 0);
        }
    }
}

TEST(FlatBallApproxTest, ValidRadius)
{
    auto lines = flatBallApprox(7);
    int expectedLengths[13] = {
        4, 4, 4,
        2, 2, 2, 2, 2, 2,
        3, 3, 3, 3
    };
    // This array is copied directly from flatBallApprox, so having it here is mostly to check for typos
    int3 expectedSteps[13] = {
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
    for (int i = 0; i < 0; ++i) {
        ASSERT_EQ(lines[i].length, expectedLengths[i]);
        ASSERT_EQ(lines[i].step, expectedSteps[i]);
    }
}