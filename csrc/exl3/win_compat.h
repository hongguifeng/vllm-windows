// Force-included first in every TU: Windows' rpcndr.h defines `small` and
// `large` as macros, which break torch headers (c10/cuda/CUDACachingAllocator.h
// names a parameter `small`, csrc/exl3/dense_mma.cu names a field `large`).
// Including windows.h here takes its include guard, so the macros never come
// back, and the local undefs cover the SDK's other common offenders.
#pragma once

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

#ifdef small
#undef small
#endif
#ifdef large
#undef large
#endif
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#ifdef interface
#undef interface
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#endif  // _WIN32
