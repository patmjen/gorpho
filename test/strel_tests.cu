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
        3, 3, 3,
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
    for (int i = 0; i < 13; ++i) {
        ASSERT_EQ(lines[i].length, expectedLengths[i]);
        ASSERT_EQ(lines[i].step, expectedSteps[i]);
    }
}

TEST(FlatBallApproxTest, ApproximationTypes)
{
    {
        SCOPED_TRACE("Inside");
        auto lines = flatBallApprox(23, APPROX_INSIDE);
        int expectedLengths[13] = {
            7, 7, 7,
            6, 6, 6, 6, 6, 6,
            5, 5, 5, 5
        };
        for (int i = 0; i < 13; ++i) {
            ASSERT_EQ(lines[i].length, expectedLengths[i]);
        }
    }
    {
        SCOPED_TRACE("Best");
        auto lines = flatBallApprox(23, APPROX_BEST);
        int expectedLengths[13] = {
            9, 9, 9,
            6, 6, 6, 6, 6, 6,
            5, 5, 5, 5
        };
        for (int i = 0; i < 13; ++i) {
            ASSERT_EQ(lines[i].length, expectedLengths[i]);
        }
    }
    {
        SCOPED_TRACE("Outside");
        auto lines = flatBallApprox(23, APPROX_OUTSIDE);
        int expectedLengths[13] = {
            11, 11, 11,
            6, 6, 6, 6, 6, 6,
            5, 5, 5, 5
        };
        for (int i = 0; i < 13; ++i) {
            ASSERT_EQ(lines[i].length, expectedLengths[i]);
        }
    }
}