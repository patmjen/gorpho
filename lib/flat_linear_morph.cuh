#ifndef FLAT_LINEAR_MORPH_CUH__
#define FLAT_LINEAR_MORPH_CUH__

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_math.cuh"

#include "view.cuh"
#include "morph.cuh"
#include "strel.cuh"
#include "util.cuh"

namespace gpho {

namespace kernel {

__host__ __device__
inline int3 getStartPos(const int3 gridPos, const AxisDir dir,
	const int3 step, const int3 volSize)
{
	if (dir & AXIS_DIR_1) {
		return make_int3((step.x > 0) ? 0 : volSize.x - 1, gridPos.y, gridPos.z);
	} else if (dir & AXIS_DIR_2) {
		return make_int3(gridPos.y, (step.y > 0) ? 0 : volSize.y - 1, gridPos.z);
	} else { // dir & AXIS_DIR_3
		return make_int3(gridPos.y, gridPos.z, (step.z > 0) ? 0 : volSize.z - 1);
	}
}

__host__ __device__
inline size_t getBufferIdx(const int3 gridPos, const AxisDir dir,
	const int3 volSize)
{
	if (dir & AXIS_DIR_1) {
		return (gridPos.y + gridPos.z*volSize.y);
	} else if (dir & AXIS_DIR_2) {
		return (gridPos.y + gridPos.z*volSize.x);
	} else { // dir & AXIS_DIR_3
		return (gridPos.y + gridPos.z*volSize.x);
	}
}

__host__ __device__
inline size_t getBufferSize(const AxisDir dir, const int3 volSize)
{
	if (dir & AXIS_DIR_1) {
		return volSize.y*volSize.z;
	} else if (dir & AXIS_DIR_2) {
		return volSize.x*volSize.z;
	} else { // dir & AXIS_DIR_3
		return volSize.x*volSize.y;
	}
}

template <MorphOp op, class Ty>
__global__ void flatLinearDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol,
	DeviceView<Ty> rBuffer, DeviceView<Ty> sBuffer, const LineSeg line, const AxisDir dir)
{
	const int halfNumSteps = line.length / 2;
	const int3 gridPos = globalPos3d();
	const int3 start = getStartPos(gridPos, dir, line.step, vol.size());
	const int bufOffset = getBufferIdx(gridPos, dir, vol.size()); // Offset into rBuffer and sBuffer
	const int bufStep = getBufferSize(dir, vol.size());
	const Ty padVal = (op == MORPH_ERODE) ? infOrMax<Ty>() : minusInfOrMin<Ty>();

	DeviceView<Ty> rsBuffer = (gridPos.x == 0) ? sBuffer : rBuffer;
	const int stepDir = (gridPos.x == 0) ? 1 : -1;

	if (start >= 0 && start < vol.size()) {
		// Initial boundary roll - only fill sBuffer
		if (gridPos.x == 0) {
			sBuffer[bufOffset] = vol[start];
			for (int k = 1; k < line.length; k++) {
				const int3 posk = start + k * line.step;
				const Ty sv = (posk >= 0 && posk < vol.size()) ? vol[posk] : padVal;
				if (op == MORPH_ERODE) {
					sBuffer[bufOffset + k * bufStep] = min(sv, sBuffer[bufOffset + (k - 1)*bufStep]);
				} else if (op == MORPH_DILATE) {
					sBuffer[bufOffset + k * bufStep] = max(sv, sBuffer[bufOffset + (k - 1)*bufStep]);
				}
			}
			for (int k = 0; k <= halfNumSteps; k++) {
				const int3 posk = start + line.step * (halfNumSteps - k);
				if (posk >= 0 && posk < vol.size()) {
					const size_t ridx = vol.idx(posk);
					res[ridx] = sBuffer[bufOffset + (line.length - k - 1)*bufStep];
				}
			}
		}
		__syncthreads();

		// Normal van Herk Gil Werman roll
		int3 pos = start + line.length * line.step;
		for (; pos >= -line.step * line.length && pos < vol.size() - line.step * line.length; pos += line.step * line.length) {
			rsBuffer[bufOffset] = vol[pos];
			int k = 1;
			// Loop unrolled for speed. Using #pragma unroll does not seem to have any effect here.
			// NOTE: Loop unrolling only seems to have an effect when moving along the x-direction
			int posRs = bufOffset;
			for (; k + 3 < line.length; k += 4) {
				const Ty rsv1 = vol[pos + stepDir * (k + 0)*line.step];
				const Ty rsv2 = vol[pos + stepDir * (k + 1)*line.step];
				const Ty rsv3 = vol[pos + stepDir * (k + 2)*line.step];
				const Ty rsv4 = vol[pos + stepDir * (k + 3)*line.step];
				if (op == MORPH_ERODE) {
					rsBuffer[posRs + 1 * bufStep] = min(rsv1, rsBuffer[posRs + 0 * bufStep]);
					rsBuffer[posRs + 2 * bufStep] = min(rsv2, rsBuffer[posRs + 1 * bufStep]);
					rsBuffer[posRs + 3 * bufStep] = min(rsv3, rsBuffer[posRs + 2 * bufStep]);
					rsBuffer[posRs + 4 * bufStep] = min(rsv4, rsBuffer[posRs + 3 * bufStep]);
				} else if (op == MORPH_DILATE) {
					rsBuffer[posRs + 1 * bufStep] = max(rsv1, rsBuffer[posRs + 0 * bufStep]);
					rsBuffer[posRs + 2 * bufStep] = max(rsv2, rsBuffer[posRs + 1 * bufStep]);
					rsBuffer[posRs + 3 * bufStep] = max(rsv3, rsBuffer[posRs + 2 * bufStep]);
					rsBuffer[posRs + 4 * bufStep] = max(rsv4, rsBuffer[posRs + 3 * bufStep]);
				}
				posRs += 4 * bufStep;
			}
			// Handle leftovers
			for (; k < line.length; k++) {
				// TODO: Use posRs
				const Ty rsv1 = vol[pos + stepDir * k*line.step];
				if (op == MORPH_ERODE) {
					rsBuffer[bufOffset + k * bufStep] = min(rsv1, rsBuffer[bufOffset + (k - 1)*bufStep]);
				} else if (op == MORPH_DILATE) {
					rsBuffer[bufOffset + k * bufStep] = max(rsv1, rsBuffer[bufOffset + (k - 1)*bufStep]);
				}
			}
			__syncthreads();
			int posR = bufOffset + gridPos.x*bufStep;
			int posS = bufOffset + (line.length - gridPos.x - 1)*bufStep;
			for (int k = gridPos.x; k < line.length; k += 2) {
				// For some reason it is faster to precompute a linear index here
				const int ridx1 = vol.idx(pos + line.step * (halfNumSteps - k));
				if (op == MORPH_ERODE) {
					res[ridx1] = min(rBuffer[posR], sBuffer[posS]);
				} else if (op == MORPH_DILATE) {
					res[ridx1] = max(rBuffer[posR], sBuffer[posS]);
				}
				posR += 2 * bufStep;
				posS -= 2 * bufStep;
			}
			__syncthreads();
		}

		// End boundary roll
		// This has a lot of branching code, so we want it in a seperate loop
		for (; pos >= line.step * line.length && pos <= vol.size() + line.step * line.length; pos += line.step * line.length) {
			if (pos >= 0 && pos < vol.size()) {
				rsBuffer[bufOffset] = vol[pos];
			} else {
				rsBuffer[bufOffset] = padVal;
			}
			for (int k = 1; k < line.length; k++) {
				const int3 posk = pos + stepDir * k*line.step;
				const Ty rsv = (posk >= 0 && posk < vol.size()) ? vol[posk] : padVal;
				if (op == MORPH_ERODE) {
					rsBuffer[bufOffset + k * bufStep] = min(rsv, rsBuffer[bufOffset + (k - 1)*bufStep]);
				} else if (op == MORPH_DILATE) {
					rsBuffer[bufOffset + k * bufStep] = max(rsv, rsBuffer[bufOffset + (k - 1)*bufStep]);
				}
			}
			__syncthreads();
			for (int k = halfNumSteps * gridPos.x; k < halfNumSteps + (line.length - halfNumSteps)*gridPos.x; k++) {
				const int3 posk = pos + line.step * (halfNumSteps - k);
				if (posk >= 0 && posk < vol.size()) {
					const size_t ridx = vol.idx(posk);
					if (op == MORPH_ERODE) {
						res[ridx] = min(rBuffer[bufOffset + k * bufStep], sBuffer[bufOffset + (line.length - k - 1)*bufStep]);
					} else if (op == MORPH_DILATE) {
						res[ridx] = max(rBuffer[bufOffset + k * bufStep], sBuffer[bufOffset + (line.length - k - 1)*bufStep]);
					}
				}
			}
			__syncthreads();
		}
	}
}

} // namespace kernel

///
/// Compute minimum needed size for R and S buffer along each axis
inline int3 minRSBufferSize(const std::vector<LineSeg>& lines)
{
    int3 stepSumPos = make_int3(0, 0, 0);
    int3 stepSumNeg = make_int3(0, 0, 0);
	for (const auto& line : lines) {
        if (line.step.x > 0) {
            stepSumPos.x += line.step.x * line.length;
        } else {
            stepSumNeg.x -= line.step.x * line.length;
        }
        if (line.step.y > 0) {
            stepSumPos.y += line.step.y * line.length;
        } else {
            stepSumNeg.y -= line.step.y * line.length;
        }
        if (line.step.z > 0) {
            stepSumPos.z += line.step.z * line.length;
        } else {
            stepSumNeg.z -= line.step.z * line.length;
        }
	}
	return max(stepSumNeg, stepSumPos);
}

///
/// Compute minimum needed size for all R and S buffers
inline size_t minTotalBufferSize(int3 minBuffer, int3 volSize)
{
	size_t bufSize = 0;
	if (minBuffer.x != 0) {
        bufSize = static_cast<size_t>(volSize.y*volSize.z*minBuffer.x);
    }
    if (minBuffer.y != 0) {
        bufSize = max(bufSize, static_cast<size_t>(volSize.x*volSize.z*minBuffer.y));
    }
    if (minBuffer.z != 0) {
        bufSize = max(bufSize, static_cast<size_t>(volSize.x*volSize.y*minBuffer.z));
    }
	return bufSize;
}

template <MorphOp op, class Ty>
inline void flatLinearDilateErode(DeviceView<Ty> res, DeviceView<const Ty> vol, DeviceView<Ty> rBuffer, 
	DeviceView<Ty> sBuffer, const LineSeg line, cudaStream_t stream = 0)
{
	if (line.step == make_int3(0, 0, 0) || line.length <= 1) {
		// Operation won't do anything, so just copy the input to the output and return
		cudaMemcpyAsync(res.data(), vol.data(), vol.numel() * sizeof(Ty), cudaMemcpyDeviceToDevice, stream);
		return;
	}
	const dim3 threads = dim3(2, 16, 16);
	if (line.step.x != 0) {
		const dim3 blocks = dim3(
			1,
			gridAxisBlocks(threads.y, vol.size().y),
			gridAxisBlocks(threads.z, vol.size().z)
		);
		kernel::flatLinearDilateErode<op><<<blocks, threads, 0, stream>>>(res, vol, rBuffer, sBuffer,
			line, AXIS_DIR_1);
	}
	if (line.step.y != 0) {
		const dim3 blocks = dim3(
			1,
			gridAxisBlocks(threads.y, vol.size().x),
			gridAxisBlocks(threads.z, vol.size().z)
		);
		kernel::flatLinearDilateErode<op><<<blocks, threads, 0, stream>>>(res, vol, rBuffer, sBuffer,
			line, AXIS_DIR_2);
	}
	if (line.step.z != 0) {
		const dim3 blocks = dim3(
			1,
			gridAxisBlocks(threads.y, vol.size().x),
			gridAxisBlocks(threads.z, vol.size().y)
		);
		kernel::flatLinearDilateErode<op><<<blocks, threads, 0, stream>>>(res, vol, rBuffer, sBuffer,
			line, AXIS_DIR_3);
	}
}


template <MorphOp op, class Ty>
inline void flatLinearDilateErode(DeviceView<Ty> res, DeviceView<Ty> resBuffer, DeviceView<const Ty> vol, 
	DeviceView<Ty> rBuffer, DeviceView<Ty> sBuffer, const std::vector<LineSeg>& lines, cudaStream_t stream = 0)
{
	DeviceView<const Ty> crntIn = vol;
	DeviceView<Ty> crntOut = (lines.size() % 2 == 0) ? resBuffer : res; // Make sure we end on res
	for (const auto& line : lines) {
		flatLinearDilateErode<op>(crntOut, crntIn, rBuffer, sBuffer, line, stream);
		crntIn = crntOut;
		crntOut = (crntOut == res) ? resBuffer : res;
	}
}

template <MorphOp op, class Ty>
inline void flatLinearDilateErode(HostView<Ty> res, HostView<const Ty> vol, const std::vector<LineSeg>& lines,
	int3 blockSize = make_int3(512,128,128))
{
	int3 minBuffer = minRSBufferSize(lines);
	int3 borderSize = blockSize >= vol.size() ? make_int3(0, 0, 0) : minBuffer; // Only use border if needed
	size_t bufSize = minTotalBufferSize(minBuffer, blockSize + 2 * borderSize);

	auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void *bufAddr)
	{
		const int3 size = block.blockSizeBorder();
		Ty *buf = static_cast<Ty *>(bufAddr);

		DeviceView<const Ty> volBlk(volVec[0], size);
		DeviceView<Ty> resBlk(resVec[0], size);
		DeviceView<Ty> rBlk(buf, make_int3(bufSize, 1, 1));
		DeviceView<Ty> sBlk(buf + bufSize, make_int3(bufSize, 1, 1));
		DeviceView<Ty> resBufferBlk(buf + 2 * bufSize, size);  // Only used if lines.size() > 1

		flatLinearDilateErode<op>(resBlk, resBufferBlk, volBlk, rBlk, sBlk, lines, stream);
	};

	size_t tmpSize = 2 * bufSize * sizeof(Ty);
	if (lines.size() > 1) {
        // If there is more than one line we need an extra output volume for intermediary results
		tmpSize += prod(blockSize + 2 * borderSize) * sizeof(Ty);
	}	
	cbp::BlockIndexIterator blockIter(vol.size(), blockSize, borderSize);
	cbp::CbpResult bpres = cbp::blockProc(processBlock, vol.data(), res.data(), blockIter, tmpSize);
	ensureCudaSuccess(cudaDeviceSynchronize());
	if (bpres != cbp::CBP_SUCCESS) {
		// TODO: Better error message
		throw std::runtime_error("Error during block processing");
	}
}

template <MorphOp op, class Ty>
inline void flatLinearDilateErode(HostView<Ty> res, HostView<const Ty> vol, const LineSeg line,
	int3 blockSize = make_int3(512,128,128))
{
	std::vector<LineSeg> lines{ line };
	flatLinearDilateErode<op>(res, vol, lines, blockSize);
}

} // namespace gpho

#endif // FLAT_LINEAR_MORPH_CUH__