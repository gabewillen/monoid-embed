#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <new>
#include <tuple>
#include <map>
#include <vector>

#include <pybind11/stl.h>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif
#if defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

namespace {

constexpr int64_t kCacheLineBytes = 64;

using Clock = std::chrono::steady_clock;

struct ProfileTiming {
    double setup_ms = 0.0;
    double recurrence_ms = 0.0;
    double butterfly_ms = 0.0;
    double activation_ms = 0.0;
    double exchange_ms = 0.0;
    double pooling_ms = 0.0;
    double layer_norm_ms = 0.0;
    double proj_ms = 0.0;
};

inline double elapsed_ms(const Clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <bool Profiled, typename Fn>
inline void profile_block(ProfileTiming* timing, double ProfileTiming::*field, Fn&& fn) {
    if constexpr (Profiled) {
        auto start = Clock::now();
        fn();
        timing->*field += elapsed_ms(start);
    } else {
        fn();
    }
}

inline std::map<std::string, double> profile_to_map(const ProfileTiming& timing) {
    const double total = timing.setup_ms
        + timing.recurrence_ms
        + timing.butterfly_ms
        + timing.activation_ms
        + timing.exchange_ms
        + timing.pooling_ms
        + timing.layer_norm_ms
        + timing.proj_ms;
    return {
        {"setup_ms", timing.setup_ms},
        {"recurrence_ms", timing.recurrence_ms},
        {"butterfly_ms", timing.butterfly_ms},
        {"activation_ms", timing.activation_ms},
        {"exchange_ms", timing.exchange_ms},
        {"pooling_ms", timing.pooling_ms},
        {"layer_norm_ms", timing.layer_norm_ms},
        {"proj_ms", timing.proj_ms},
        {"total_ms", total},
    };
}

template <typename T>
inline int64_t cacheline_pad_elems(int64_t count) {
    if (count <= 0) {
        return count;
    }
    const int64_t elems = std::max<int64_t>(1, kCacheLineBytes / static_cast<int64_t>(sizeof(T)));
    return ((count + elems - 1) / elems) * elems;
}

struct CpuFeatures {
    bool avx512 = false;
    bool avx512bw = false;
    bool avx2 = false;
    bool neon = false;
};

inline CpuFeatures detect_cpu_features() {
    CpuFeatures features;
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    features.avx512 = __builtin_cpu_supports("avx512f");
    features.avx512bw = __builtin_cpu_supports("avx512bw");
    features.avx2 = __builtin_cpu_supports("avx2");
#endif
#endif
#if defined(__ARM_NEON)
    features.neon = true;
#endif
    return features;
}

inline const CpuFeatures& cpu_features() {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}

struct KernelOps {
    using DotF32Fn = float (*)(const float*, const float*, int64_t);
    using DotI8Fn = int32_t (*)(const int8_t*, const int32_t*, int64_t);
    using RecF32Fn = void (*)(float*, const float*, const float*, int64_t);
    using RecQ15Fn = void (*)(int32_t*, const int16_t*, const int8_t*, int64_t, int32_t);
    using ReduceSumFn = float (*)(const float*, int64_t);
    using ReduceVarFn = float (*)(const float*, int64_t, float);

    DotF32Fn dot_f32 = nullptr;
    DotI8Fn dot_i8_i32 = nullptr;
    RecF32Fn rec_f32 = nullptr;
    RecQ15Fn rec_q15 = nullptr;
    ReduceSumFn reduce_sum_f32 = nullptr;
    ReduceVarFn reduce_var_f32 = nullptr;
};

const KernelOps& kernel_ops();

template <typename T, std::size_t Alignment = 64>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Alignment));
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, std::size_t Alignment = 64>
using AlignedVector = std::vector<T, AlignedAllocator<T, Alignment>>;

inline bool require_aligned_ptrs() {
    static const bool require = []() {
        const char* env = std::getenv("MONOID_CPU_REQUIRE_ALIGNED");
        return env && env[0] == '1';
    }();
    return require;
}

inline bool force_proj_intrinsics() {
    static const bool enable = []() {
        const char* env = std::getenv("MONOID_CPU_FORCE_PROJ_INTRINSICS");
        return env && env[0] == '1';
    }();
    return enable;
}

inline bool is_aligned_ptr(const void* ptr, std::size_t alignment) {
    const uintptr_t mask = alignment - 1;
    return (reinterpret_cast<uintptr_t>(ptr) & mask) == 0;
}

inline void check_aligned_if_required(const void* ptr, std::size_t alignment, const char* name) {
    if (require_aligned_ptrs() && !is_aligned_ptr(ptr, alignment)) {
        TORCH_CHECK(false, "Expected aligned pointer for ", name);
    }
}

inline void check_stride_alignment(int64_t stride_elems, std::size_t elem_size, std::size_t alignment, const char* name) {
    if (require_aligned_ptrs()) {
        const std::size_t stride_bytes = static_cast<std::size_t>(stride_elems) * elem_size;
        if ((stride_bytes & (alignment - 1)) != 0) {
            TORCH_CHECK(false, "Expected aligned stride for ", name);
        }
    }
}

template <typename T, std::size_t Alignment = 64>
inline T* assume_aligned_if(T* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    if (is_aligned_ptr(ptr, Alignment)) {
        return reinterpret_cast<T*>(__builtin_assume_aligned(ptr, Alignment));
    }
#endif
    return ptr;
}

template <typename T, std::size_t Alignment = 64>
inline const T* assume_aligned_if(const T* ptr) {
#if defined(__GNUC__) || defined(__clang__)
    if (is_aligned_ptr(ptr, Alignment)) {
        return reinterpret_cast<const T*>(__builtin_assume_aligned(ptr, Alignment));
    }
#endif
    return ptr;
}

inline int16_t clamp_int16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}

inline int32_t clamp_int32(int32_t v, int32_t lo, int32_t hi) {
    return std::min(std::max(v, lo), hi);
}

inline int32_t clamp_int32_from_i64(int64_t v) {
    if (v > INT32_MAX) return INT32_MAX;
    if (v < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(v);
}

inline int32_t round_shift_right(int32_t v, int32_t shift) {
    if (shift <= 0) {
        return v;
    }
    if (shift >= 31) {
        return v >= 0 ? 0 : -1;
    }
    const int64_t offset = 1LL << (shift - 1);
    const int64_t value = static_cast<int64_t>(v);
    if (value >= 0) {
        return static_cast<int32_t>((value + offset) >> shift);
    }
    return static_cast<int32_t>((value - offset) >> shift);
}

inline int32_t mul_q15(int32_t x, int16_t a_q15) {
    const int64_t prod = static_cast<int64_t>(x) * static_cast<int64_t>(a_q15);
    const int64_t acc = (prod + 16384) >> 15;
    return clamp_int32_from_i64(acc);
}

inline void add_int16_saturate(
    int16_t* __restrict__ dst,
    const int16_t* __restrict__ src,
    int64_t n
) {
#if defined(__AVX2__)
    if (cpu_features().avx2) {
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            __m256i out = _mm256_adds_epi16(a, b);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), out);
        }
        for (; i < n; ++i) {
            int32_t sum = static_cast<int32_t>(dst[i]) + static_cast<int32_t>(src[i]);
            dst[i] = clamp_int16(sum);
        }
        return;
    }
#endif

#if defined(__ARM_NEON)
    if (cpu_features().neon) {
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            int16x8_t a = vld1q_s16(dst + i);
            int16x8_t b = vld1q_s16(src + i);
            int16x8_t out = vqaddq_s16(a, b);
            vst1q_s16(dst + i, out);
        }
        for (; i < n; ++i) {
            int32_t sum = static_cast<int32_t>(dst[i]) + static_cast<int32_t>(src[i]);
            dst[i] = clamp_int16(sum);
        }
        return;
    }
#endif

#if defined(__SSE2__)
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
        __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m128i out = _mm_adds_epi16(a, b);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), out);
    }
    for (; i < n; ++i) {
        int32_t sum = static_cast<int32_t>(dst[i]) + static_cast<int32_t>(src[i]);
        dst[i] = clamp_int16(sum);
    }
    return;
#endif

    for (int64_t i = 0; i < n; ++i) {
        int32_t sum = static_cast<int32_t>(dst[i]) + static_cast<int32_t>(src[i]);
        dst[i] = clamp_int16(sum);
    }
}

inline void copy_i16_to_i16_i32(
    const int16_t* __restrict__ src,
    int16_t* __restrict__ dst16,
    int32_t* __restrict__ dst32,
    int64_t n
) {
#if defined(__AVX2__)
    if (cpu_features().avx2) {
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
            if (dst16) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst16 + i), v);
            }
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            __m256i out0 = _mm256_cvtepi16_epi32(lo);
            __m256i out1 = _mm256_cvtepi16_epi32(hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst32 + i), out0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst32 + i + 8), out1);
        }
        for (; i < n; ++i) {
            int16_t val = src[i];
            if (dst16) {
                dst16[i] = val;
            }
            dst32[i] = static_cast<int32_t>(val);
        }
        return;
    }
#endif

#if defined(__ARM_NEON)
    if (cpu_features().neon) {
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            int16x8_t v = vld1q_s16(src + i);
            if (dst16) {
                vst1q_s16(dst16 + i, v);
            }
            int32x4_t lo = vmovl_s16(vget_low_s16(v));
            int32x4_t hi = vmovl_s16(vget_high_s16(v));
            vst1q_s32(dst32 + i, lo);
            vst1q_s32(dst32 + i + 4, hi);
        }
        for (; i < n; ++i) {
            int16_t val = src[i];
            if (dst16) {
                dst16[i] = val;
            }
            dst32[i] = static_cast<int32_t>(val);
        }
        return;
    }
#endif

#if defined(__SSE4_1__)
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        if (dst16) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst16 + i), v);
        }
        __m128i lo = v;
        __m128i hi = _mm_srli_si128(v, 8);
        __m128i out0 = _mm_cvtepi16_epi32(lo);
        __m128i out1 = _mm_cvtepi16_epi32(hi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst32 + i), out0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst32 + i + 4), out1);
    }
    for (; i < n; ++i) {
        int16_t val = src[i];
        if (dst16) {
            dst16[i] = val;
        }
        dst32[i] = static_cast<int32_t>(val);
    }
    return;
#endif

    for (int64_t i = 0; i < n; ++i) {
        int16_t val = src[i];
        if (dst16) {
            dst16[i] = val;
        }
        dst32[i] = static_cast<int32_t>(val);
    }
}

inline void copy_i16_to_i32(
    const int16_t* __restrict__ src,
    int32_t* __restrict__ dst32,
    int64_t n
) {
    copy_i16_to_i16_i32(src, nullptr, dst32, n);
}

inline void clamp_store_s32_to_s16(
    int32_t* __restrict__ s32,
    int16_t* __restrict__ s16,
    int64_t n
) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const CpuFeatures& features = cpu_features();
    if (features.avx512 && features.avx512bw) {
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(s32 + i));
            __m256i packed = _mm512_cvtsepi32_epi16(v);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(s16 + i), packed);
            __m128i lo = _mm256_castsi256_si128(packed);
            __m128i hi = _mm256_extracti128_si256(packed, 1);
            __m256i out0 = _mm256_cvtepi16_epi32(lo);
            __m256i out1 = _mm256_cvtepi16_epi32(hi);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(s32 + i), out0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(s32 + i + 8), out1);
        }
        for (; i < n; ++i) {
            int16_t v16 = clamp_int16(s32[i]);
            s16[i] = v16;
            s32[i] = static_cast<int32_t>(v16);
        }
        return;
    }
#endif

#if defined(__ARM_NEON)
    if (cpu_features().neon) {
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            int32x4_t s0 = vld1q_s32(s32 + i);
            int32x4_t s1 = vld1q_s32(s32 + i + 4);
            int16x4_t p0 = vqmovn_s32(s0);
            int16x4_t p1 = vqmovn_s32(s1);
            int16x8_t packed = vcombine_s16(p0, p1);
            vst1q_s16(s16 + i, packed);
            int32x4_t out0 = vmovl_s16(vget_low_s16(packed));
            int32x4_t out1 = vmovl_s16(vget_high_s16(packed));
            vst1q_s32(s32 + i, out0);
            vst1q_s32(s32 + i + 4, out1);
        }
        for (; i < n; ++i) {
            int16_t v16 = clamp_int16(s32[i]);
            s16[i] = v16;
            s32[i] = static_cast<int32_t>(v16);
        }
        return;
    }
#endif

#if defined(__SSE4_1__)
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s32 + i));
        __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s32 + i + 4));
        __m128i packed = _mm_packs_epi32(v0, v1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(s16 + i), packed);
        __m128i out0 = _mm_cvtepi16_epi32(packed);
        __m128i out1 = _mm_cvtepi16_epi32(_mm_srli_si128(packed, 8));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(s32 + i), out0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(s32 + i + 4), out1);
    }
    for (; i < n; ++i) {
        int16_t v16 = clamp_int16(s32[i]);
        s16[i] = v16;
        s32[i] = static_cast<int32_t>(v16);
    }
    return;
#endif

    for (int64_t i = 0; i < n; ++i) {
        int16_t v16 = clamp_int16(s32[i]);
        s16[i] = v16;
        s32[i] = static_cast<int32_t>(v16);
    }
}

inline float dot_f32(const float* __restrict__ a, const float* __restrict__ b, int64_t n);

inline int64_t parallel_grain_size(int64_t work, int64_t min_per_task, int64_t tasks_per_thread) {
    if (work <= 0) {
        return work;
    }
    const int64_t threads = at::get_num_threads();
    if (threads <= 1) {
        return work;
    }
    const int64_t target_tasks = std::max<int64_t>(1, threads * tasks_per_thread);
    if (work < target_tasks) {
        return work;
    }
    int64_t grain = work / target_tasks;
    if (grain < min_per_task) {
        grain = min_per_task;
    }
    return grain;
}

inline int64_t parallel_grain_size(int64_t work) {
    return parallel_grain_size(work, 1, 2);
}

inline void normalize_l2(float* __restrict__ vec, int64_t size) {
    float norm = dot_f32(vec, vec, size);
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        float inv = 1.0f / norm;
        for (int64_t i = 0; i < size; ++i) {
            vec[i] *= inv;
        }
    }
}

inline float tanh_fast(float x) {
    const float xc = std::max(-5.0f, std::min(5.0f, x));
    const float x2 = xc * xc;
    const float num = 135135.0f + x2 * (17325.0f + x2 * (378.0f + x2));
    const float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    const float y = xc * (num / den);
    return std::max(-1.0f, std::min(1.0f, y));
}

inline float tanh_eval(float x) {
#ifdef MONOID_FAST_TANH
    return tanh_fast(x);
#else
    return std::tanh(x);
#endif
}

inline float dot_f32_scalar(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(__AVX512F__)
inline float dot_f32_avx512(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    __m512 acc = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
#if defined(__FMA__)
        acc = _mm512_fmadd_ps(va, vb, acc);
#else
        acc = _mm512_add_ps(acc, _mm512_mul_ps(va, vb));
#endif
    }
    alignas(64) float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
        + tmp[4] + tmp[5] + tmp[6] + tmp[7]
        + tmp[8] + tmp[9] + tmp[10] + tmp[11]
        + tmp[12] + tmp[13] + tmp[14] + tmp[15];
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

#if defined(__AVX2__)
inline float dot_f32_avx2(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(va, vb, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

#if defined(__ARM_NEON)
inline float dot_f32_neon(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    alignas(16) float tmp[4];
    vst1q_f32(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

inline float dot_f32(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
    return kernel_ops().dot_f32(a, b, n);
}

inline int32_t dot_i8_i32_scalar(const int8_t* __restrict__ w, const int32_t* __restrict__ u, int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<int64_t>(w[i]) * static_cast<int64_t>(u[i]);
    }
    return clamp_int32_from_i64(sum);
}

inline int32_t dot_i8_i16_scalar(const int8_t* __restrict__ w, const int16_t* __restrict__ u, int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<int64_t>(w[i]) * static_cast<int64_t>(u[i]);
    }
    return clamp_int32_from_i64(sum);
}

#if defined(__AVX2__)
inline int32_t dot_i8_i32_avx2(const int8_t* __restrict__ w, const int32_t* __restrict__ u, int64_t n) {
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i));
        __m256i w32 = _mm256_cvtepi8_epi32(w8);
        __m256i u32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
        __m256i prod = _mm256_mullo_epi32(w32, u32);
        acc = _mm256_add_epi32(acc, prod);
    }
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), acc);
    int64_t sum = static_cast<int64_t>(tmp[0]) + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        sum += static_cast<int64_t>(w[i]) * static_cast<int64_t>(u[i]);
    }
    return clamp_int32_from_i64(sum);
}
#endif

#if defined(__AVX2__)
inline int32_t dot_i8_i16_avx2(const int8_t* __restrict__ w, const int16_t* __restrict__ u, int64_t n) {
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i w8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
        __m256i w16 = _mm256_cvtepi8_epi16(w8);
        __m256i u16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + i));
        __m256i prod = _mm256_madd_epi16(w16, u16);
        acc = _mm256_add_epi32(acc, prod);
    }
    alignas(32) int32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), acc);
    int64_t sum = static_cast<int64_t>(tmp[0]) + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        sum += static_cast<int64_t>(w[i]) * static_cast<int64_t>(u[i]);
    }
    return clamp_int32_from_i64(sum);
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline int32_t dot_i8_i16_avx512(const int8_t* __restrict__ w, const int16_t* __restrict__ u, int64_t n) {
    __m512i acc = _mm512_setzero_si512();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i w8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        __m512i w16 = _mm512_cvtepi8_epi16(w8);
        __m512i u16 = _mm512_loadu_si512(reinterpret_cast<const void*>(u + i));
        __m512i prod = _mm512_madd_epi16(w16, u16);
        acc = _mm512_add_epi32(acc, prod);
    }
    alignas(64) int32_t tmp[16];
    _mm512_storeu_si512(reinterpret_cast<void*>(tmp), acc);
    int64_t sum = 0;
    for (int i = 0; i < 16; ++i) {
        sum += tmp[i];
    }
    for (; i < n; ++i) {
        sum += static_cast<int64_t>(w[i]) * static_cast<int64_t>(u[i]);
    }
    return clamp_int32_from_i64(sum);
}
#endif

#if defined(__ARM_NEON)
inline int32_t dot_i8_i16_neon(const int8_t* __restrict__ w, const int16_t* __restrict__ u, int64_t n) {
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t w8 = vld1_s8(w + i);
        int16x8_t w16 = vmovl_s8(w8);
        int16x8_t u16 = vld1q_s16(u + i);
        int32x4_t prod0 = vmull_s16(vget_low_s16(w16), vget_low_s16(u16));
        int32x4_t prod1 = vmull_s16(vget_high_s16(w16), vget_high_s16(u16));
        acc0 = vaddq_s32(acc0, prod0);
        acc1 = vaddq_s32(acc1, prod1);
    }
    alignas(16) int32_t tmp0[4];
    alignas(16) int32_t tmp1[4];
    vst1q_s32(tmp0, acc0);
    vst1q_s32(tmp1, acc1);
    int64_t sum = static_cast<int64_t>(tmp0[0]) + tmp0[1] + tmp0[2] + tmp0[3]
        + tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
    for (; i < n; ++i) {
        sum += static_cast<int64_t>(w[i]) * static_cast<int64_t>(u[i]);
    }
    return clamp_int32_from_i64(sum);
}
#endif

inline int32_t dot_i8_i32(const int8_t* __restrict__ w, const int32_t* __restrict__ u, int64_t n) {
    return kernel_ops().dot_i8_i32(w, u, n);
}

inline int32_t dot_i8_i16(const int8_t* __restrict__ w, const int16_t* __restrict__ u, int64_t n) {
    const CpuFeatures& features = cpu_features();
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (features.avx512 && features.avx512bw) {
        return dot_i8_i16_avx512(w, u, n);
    }
#endif
#if defined(__AVX2__)
    if (features.avx2) {
        return dot_i8_i16_avx2(w, u, n);
    }
#endif
#if defined(__ARM_NEON)
    if (features.neon) {
        return dot_i8_i16_neon(w, u, n);
    }
#endif
    return dot_i8_i16_scalar(w, u, n);
}

inline void recurrence_f32_scalar(
    float* __restrict__ s,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t n
) {
    for (int64_t i = 0; i < n; ++i) {
        s[i] = a[i] * s[i] + b[i];
    }
}

#if defined(__AVX512F__)
inline void recurrence_f32_avx512(
    float* __restrict__ s,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t n
) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 sv = _mm512_loadu_ps(s + i);
        __m512 av = _mm512_loadu_ps(a + i);
        __m512 bv = _mm512_loadu_ps(b + i);
#if defined(__FMA__)
        __m512 out = _mm512_fmadd_ps(av, sv, bv);
#else
        __m512 out = _mm512_add_ps(_mm512_mul_ps(av, sv), bv);
#endif
        _mm512_storeu_ps(s + i, out);
    }
    for (; i < n; ++i) {
        s[i] = a[i] * s[i] + b[i];
    }
}
#endif

#if defined(__AVX2__)
inline void recurrence_f32_avx2(
    float* __restrict__ s,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t n
) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 sv = _mm256_loadu_ps(s + i);
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        __m256 out = _mm256_fmadd_ps(av, sv, bv);
#else
        __m256 out = _mm256_add_ps(_mm256_mul_ps(av, sv), bv);
#endif
        _mm256_storeu_ps(s + i, out);
    }
    for (; i < n; ++i) {
        s[i] = a[i] * s[i] + b[i];
    }
}
#endif

#if defined(__ARM_NEON)
inline void recurrence_f32_neon(
    float* __restrict__ s,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t n
) {
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t sv = vld1q_f32(s + i);
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        float32x4_t out = vmlaq_f32(bv, av, sv);
        vst1q_f32(s + i, out);
    }
    for (; i < n; ++i) {
        s[i] = a[i] * s[i] + b[i];
    }
}
#endif

inline void recurrence_f32(
    float* __restrict__ s,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t n
) {
    kernel_ops().rec_f32(s, a, b, n);
}

inline void recurrence_q15_scalar(
    int32_t* __restrict__ s,
    const int16_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int64_t n,
    int32_t b_shift
) {
    for (int64_t i = 0; i < n; ++i) {
        int64_t val = static_cast<int64_t>(mul_q15(s[i], a[i]));
        val += static_cast<int64_t>(b[i]) << b_shift;
        s[i] = clamp_int32_from_i64(val);
    }
}

#if defined(__AVX2__)
inline void recurrence_q15_avx2(
    int32_t* __restrict__ s,
    const int16_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int64_t n,
    int32_t b_shift
) {
    const __m256i offset = _mm256_set1_epi32(16384);
    const __m128i shift = _mm_cvtsi32_si128(b_shift);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i s32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
        __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
        __m256i a32 = _mm256_cvtepi16_epi32(a16);
        __m256i prod = _mm256_mullo_epi32(s32, a32);
        __m256i acc = _mm256_add_epi32(prod, offset);
        acc = _mm256_srai_epi32(acc, 15);

        __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        __m256i b32 = _mm256_cvtepi8_epi32(b8);
        if (b_shift > 0) {
            b32 = _mm256_sll_epi32(b32, shift);
        }
        acc = _mm256_add_epi32(acc, b32);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(s + i), acc);
    }
    for (; i < n; ++i) {
        int64_t val = static_cast<int64_t>(mul_q15(s[i], a[i]));
        val += static_cast<int64_t>(b[i]) << b_shift;
        s[i] = clamp_int32_from_i64(val);
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
inline void recurrence_q15_avx512(
    int32_t* __restrict__ s,
    const int16_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int64_t n,
    int32_t b_shift
) {
    const __m512i offset = _mm512_set1_epi32(16384);
    const __m128i shift = _mm_cvtsi32_si128(b_shift);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i s32 = _mm512_loadu_si512(reinterpret_cast<const void*>(s + i));
        __m256i a16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m512i a32 = _mm512_cvtepi16_epi32(a16);
        __m512i prod = _mm512_mullo_epi32(s32, a32);
        __m512i acc = _mm512_add_epi32(prod, offset);
        acc = _mm512_srai_epi32(acc, 15);

        __m128i b8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
        __m512i b32 = _mm512_cvtepi8_epi32(b8);
        if (b_shift > 0) {
            b32 = _mm512_sll_epi32(b32, shift);
        }
        acc = _mm512_add_epi32(acc, b32);
        _mm512_storeu_si512(reinterpret_cast<void*>(s + i), acc);
    }
    for (; i < n; ++i) {
        int64_t val = static_cast<int64_t>(mul_q15(s[i], a[i]));
        val += static_cast<int64_t>(b[i]) << b_shift;
        s[i] = clamp_int32_from_i64(val);
    }
}
#endif

#if defined(__ARM_NEON)
inline void recurrence_q15_neon(
    int32_t* __restrict__ s,
    const int16_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int64_t n,
    int32_t b_shift
) {
    const int32x4_t offset = vdupq_n_s32(16384);
    const int32x4_t shift = vdupq_n_s32(b_shift);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int32x4_t s0 = vld1q_s32(s + i);
        int32x4_t s1 = vld1q_s32(s + i + 4);
        int16x8_t a16 = vld1q_s16(a + i);
        int32x4_t a0 = vmovl_s16(vget_low_s16(a16));
        int32x4_t a1 = vmovl_s16(vget_high_s16(a16));
        int32x4_t acc0 = vshrq_n_s32(vaddq_s32(vmulq_s32(s0, a0), offset), 15);
        int32x4_t acc1 = vshrq_n_s32(vaddq_s32(vmulq_s32(s1, a1), offset), 15);

        int8x8_t b8 = vld1_s8(b + i);
        int16x8_t b16 = vmovl_s8(b8);
        int32x4_t b0 = vmovl_s16(vget_low_s16(b16));
        int32x4_t b1 = vmovl_s16(vget_high_s16(b16));
        if (b_shift > 0) {
            b0 = vshlq_s32(b0, shift);
            b1 = vshlq_s32(b1, shift);
        }
        acc0 = vaddq_s32(acc0, b0);
        acc1 = vaddq_s32(acc1, b1);
        vst1q_s32(s + i, acc0);
        vst1q_s32(s + i + 4, acc1);
    }
    for (; i < n; ++i) {
        int64_t val = static_cast<int64_t>(mul_q15(s[i], a[i]));
        val += static_cast<int64_t>(b[i]) << b_shift;
        s[i] = clamp_int32_from_i64(val);
    }
}
#endif

inline void recurrence_q15(
    int32_t* __restrict__ s,
    const int16_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int64_t n,
    int32_t b_shift
) {
    kernel_ops().rec_q15(s, a, b, n, b_shift);
}

inline float reduce_sum_f32_scalar(const float* __restrict__ a, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

#if defined(__AVX512F__)
inline float reduce_sum_f32_avx512(const float* __restrict__ a, int64_t n) {
    __m512 acc = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_add_ps(acc, _mm512_loadu_ps(a + i));
    }
    alignas(64) float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
        + tmp[4] + tmp[5] + tmp[6] + tmp[7]
        + tmp[8] + tmp[9] + tmp[10] + tmp[11]
        + tmp[12] + tmp[13] + tmp[14] + tmp[15];
    for (; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}
#endif

#if defined(__AVX2__)
inline float reduce_sum_f32_avx2(const float* __restrict__ a, int64_t n) {
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}
#endif

#if defined(__ARM_NEON)
inline float reduce_sum_f32_neon(const float* __restrict__ a, int64_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(a + i));
    }
    alignas(16) float tmp[4];
    vst1q_f32(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}
#endif

inline float reduce_sum_f32(const float* __restrict__ a, int64_t n) {
    return kernel_ops().reduce_sum_f32(a, n);
}

inline float reduce_var_f32_scalar(const float* __restrict__ a, int64_t n, float mean) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float d = a[i] - mean;
        sum += d * d;
    }
    return sum;
}

#if defined(__AVX512F__)
inline float reduce_var_f32_avx512(const float* __restrict__ a, int64_t n, float mean) {
    __m512 acc = _mm512_setzero_ps();
    const __m512 mean_v = _mm512_set1_ps(mean);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(a + i);
        __m512 d = _mm512_sub_ps(v, mean_v);
        acc = _mm512_add_ps(acc, _mm512_mul_ps(d, d));
    }
    alignas(64) float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3]
        + tmp[4] + tmp[5] + tmp[6] + tmp[7]
        + tmp[8] + tmp[9] + tmp[10] + tmp[11]
        + tmp[12] + tmp[13] + tmp[14] + tmp[15];
    for (; i < n; ++i) {
        float d = a[i] - mean;
        sum += d * d;
    }
    return sum;
}
#endif

#if defined(__AVX2__)
inline float reduce_var_f32_avx2(const float* __restrict__ a, int64_t n, float mean) {
    __m256 acc = _mm256_setzero_ps();
    const __m256 mean_v = _mm256_set1_ps(mean);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        __m256 d = _mm256_sub_ps(v, mean_v);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
    }
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i) {
        float d = a[i] - mean;
        sum += d * d;
    }
    return sum;
}
#endif

#if defined(__ARM_NEON)
inline float reduce_var_f32_neon(const float* __restrict__ a, int64_t n, float mean) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    const float32x4_t mean_v = vdupq_n_f32(mean);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(a + i);
        float32x4_t d = vsubq_f32(v, mean_v);
        acc = vmlaq_f32(acc, d, d);
    }
    alignas(16) float tmp[4];
    vst1q_f32(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) {
        float d = a[i] - mean;
        sum += d * d;
    }
    return sum;
}
#endif

inline float reduce_var_f32(const float* __restrict__ a, int64_t n, float mean) {
    return kernel_ops().reduce_var_f32(a, n, mean);
}

inline KernelOps build_kernel_ops() {
    KernelOps ops;
    ops.dot_f32 = dot_f32_scalar;
    ops.dot_i8_i32 = dot_i8_i32_scalar;
    ops.rec_f32 = recurrence_f32_scalar;
    ops.rec_q15 = recurrence_q15_scalar;
    ops.reduce_sum_f32 = reduce_sum_f32_scalar;
    ops.reduce_var_f32 = reduce_var_f32_scalar;

    const CpuFeatures& features = cpu_features();
    bool used_avx512 = false;

#if defined(__AVX512F__)
    if (features.avx512) {
        ops.dot_f32 = dot_f32_avx512;
        ops.rec_f32 = recurrence_f32_avx512;
        ops.reduce_sum_f32 = reduce_sum_f32_avx512;
        ops.reduce_var_f32 = reduce_var_f32_avx512;
        used_avx512 = true;
    }
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (features.avx512 && features.avx512bw) {
        ops.rec_q15 = recurrence_q15_avx512;
    }
#endif

#if defined(__AVX2__)
    if (features.avx2) {
        if (!used_avx512) {
            ops.dot_f32 = dot_f32_avx2;
            ops.rec_f32 = recurrence_f32_avx2;
            ops.reduce_sum_f32 = reduce_sum_f32_avx2;
            ops.reduce_var_f32 = reduce_var_f32_avx2;
        }
        ops.dot_i8_i32 = dot_i8_i32_avx2;
        if (!used_avx512) {
            ops.rec_q15 = recurrence_q15_avx2;
        }
    }
#endif

#if defined(__ARM_NEON)
    if (features.neon) {
        ops.dot_f32 = dot_f32_neon;
        ops.rec_f32 = recurrence_f32_neon;
        ops.reduce_sum_f32 = reduce_sum_f32_neon;
        ops.reduce_var_f32 = reduce_var_f32_neon;
        ops.rec_q15 = recurrence_q15_neon;
    }
#endif

    return ops;
}

const KernelOps& kernel_ops() {
    static const KernelOps ops = build_kernel_ops();
    return ops;
}


inline void layer_norm_row(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int64_t size
) {
    float mean = reduce_sum_f32(input, size) / static_cast<float>(size);
    float var = reduce_var_f32(input, size, mean) / static_cast<float>(size);
    float inv = 1.0f / std::sqrt(var + 1e-5f);
    if (weight && bias) {
        for (int64_t i = 0; i < size; ++i) {
            output[i] = (input[i] - mean) * inv * weight[i] + bias[i];
        }
    } else if (weight) {
        for (int64_t i = 0; i < size; ++i) {
            output[i] = (input[i] - mean) * inv * weight[i];
        }
    } else if (bias) {
        for (int64_t i = 0; i < size; ++i) {
            output[i] = (input[i] - mean) * inv + bias[i];
        }
    } else {
        for (int64_t i = 0; i < size; ++i) {
            output[i] = (input[i] - mean) * inv;
        }
    }
}


inline int64_t butterfly_mix_stages(int64_t tile_dim) {
    TORCH_CHECK(tile_dim > 0, "tile_dim must be positive.");
    int64_t tmp = tile_dim;
    int64_t stages = 0;
    while (tmp > 1) {
        TORCH_CHECK(tmp % 2 == 0, "tile_dim must be a power of two.");
        tmp >>= 1;
        stages += 1;
    }
    return stages;
}

void butterfly_mix_float_interleaved_stages(
    float* __restrict__ base,
    int64_t tile_dim,
    int64_t n_tiles,
    int64_t stages
) {
    TORCH_CHECK(tile_dim > 0, "tile_dim must be positive.");
    if (n_tiles == 8) {
        const CpuFeatures& features = cpu_features();
#if defined(__AVX512F__)
        if (features.avx512) {
            int64_t step = 1;
            for (int64_t stage = 0; stage < stages; ++stage) {
                const int64_t jump = step * 2;
                for (int64_t start = 0; start < tile_dim; start += jump) {
                    float* p0 = base + start * 8;
                    float* p1 = base + (start + step) * 8;
                    int64_t i = 0;
                    for (; i + 1 < step; i += 2) {
                        __m512 a = _mm512_loadu_ps(p0 + i * 8);
                        __m512 b = _mm512_loadu_ps(p1 + i * 8);
                        _mm512_storeu_ps(p0 + i * 8, _mm512_add_ps(a, b));
                        _mm512_storeu_ps(p1 + i * 8, _mm512_sub_ps(a, b));
                    }
                    for (; i < step; ++i) {
                        float* a_ptr = p0 + i * 8;
                        float* b_ptr = p1 + i * 8;
                        for (int64_t t = 0; t < 8; ++t) {
                            float a = a_ptr[t];
                            float b = b_ptr[t];
                            a_ptr[t] = a + b;
                            b_ptr[t] = a - b;
                        }
                    }
                }
                step *= 2;
            }
            return;
        }
#endif
#if defined(__AVX2__)
        if (features.avx2) {
            int64_t step = 1;
            for (int64_t stage = 0; stage < stages; ++stage) {
                const int64_t jump = step * 2;
                for (int64_t start = 0; start < tile_dim; start += jump) {
                    float* p0 = base + start * 8;
                    float* p1 = base + (start + step) * 8;
                    for (int64_t i = 0; i < step; ++i) {
                        __m256 a = _mm256_loadu_ps(p0 + i * 8);
                        __m256 b = _mm256_loadu_ps(p1 + i * 8);
                        _mm256_storeu_ps(p0 + i * 8, _mm256_add_ps(a, b));
                        _mm256_storeu_ps(p1 + i * 8, _mm256_sub_ps(a, b));
                    }
                }
                step *= 2;
            }
            return;
        }
#endif
#if defined(__ARM_NEON)
        if (features.neon) {
            int64_t step = 1;
            for (int64_t stage = 0; stage < stages; ++stage) {
                const int64_t jump = step * 2;
                for (int64_t start = 0; start < tile_dim; start += jump) {
                    float* p0 = base + start * 8;
                    float* p1 = base + (start + step) * 8;
                    for (int64_t i = 0; i < step; ++i) {
                        float* a_ptr = p0 + i * 8;
                        float* b_ptr = p1 + i * 8;
                        float32x4_t a0 = vld1q_f32(a_ptr);
                        float32x4_t a1 = vld1q_f32(a_ptr + 4);
                        float32x4_t b0 = vld1q_f32(b_ptr);
                        float32x4_t b1 = vld1q_f32(b_ptr + 4);
                        vst1q_f32(a_ptr, vaddq_f32(a0, b0));
                        vst1q_f32(a_ptr + 4, vaddq_f32(a1, b1));
                        vst1q_f32(b_ptr, vsubq_f32(a0, b0));
                        vst1q_f32(b_ptr + 4, vsubq_f32(a1, b1));
                    }
                }
                step *= 2;
            }
            return;
        }
#endif
    }
    int64_t step = 1;
    for (int64_t stage = 0; stage < stages; ++stage) {
        for (int64_t start = 0; start < tile_dim; start += step * 2) {
            for (int64_t i = 0; i < step; ++i) {
                int64_t offset0 = (start + i) * n_tiles;
                int64_t offset1 = (start + step + i) * n_tiles;
                for (int64_t t = 0; t < n_tiles; ++t) {
                    float a = base[offset0 + t];
                    float b = base[offset1 + t];
                    base[offset0 + t] = a + b;
                    base[offset1 + t] = a - b;
                }
            }
        }
        step *= 2;
    }
}

void butterfly_mix_float_interleaved(float* __restrict__ base, int64_t tile_dim, int64_t n_tiles) {
    const int64_t stages = butterfly_mix_stages(tile_dim);
    butterfly_mix_float_interleaved_stages(base, tile_dim, n_tiles, stages);
}

void butterfly_mix_int16_interleaved(int16_t* __restrict__ base, int64_t tile_dim, int64_t n_tiles) {
    TORCH_CHECK(tile_dim > 0, "tile_dim must be positive.");
    int64_t tmp = tile_dim;
    int64_t stages = 0;
    while (tmp > 1) {
        TORCH_CHECK(tmp % 2 == 0, "tile_dim must be a power of two.");
        tmp >>= 1;
        stages += 1;
    }
    if (n_tiles == 8) {
#if defined(__ARM_NEON)
        int64_t step = 1;
        for (int64_t stage = 0; stage < stages; ++stage) {
            const int64_t jump = step * 2;
            for (int64_t start = 0; start < tile_dim; start += jump) {
                int16_t* p0 = base + start * 8;
                int16_t* p1 = base + (start + step) * 8;
                for (int64_t i = 0; i < step; ++i) {
                    int16x8_t a = vld1q_s16(p0 + i * 8);
                    int16x8_t b = vld1q_s16(p1 + i * 8);
                    vst1q_s16(p0 + i * 8, vqaddq_s16(a, b));
                    vst1q_s16(p1 + i * 8, vqsubq_s16(a, b));
                }
            }
            step *= 2;
        }
        return;
#elif defined(__SSE2__)
        int64_t step = 1;
        for (int64_t stage = 0; stage < stages; ++stage) {
            const int64_t jump = step * 2;
            for (int64_t start = 0; start < tile_dim; start += jump) {
                int16_t* p0 = base + start * 8;
                int16_t* p1 = base + (start + step) * 8;
                for (int64_t i = 0; i < step; ++i) {
                    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + i * 8));
                    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + i * 8));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(p0 + i * 8), _mm_adds_epi16(a, b));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(p1 + i * 8), _mm_subs_epi16(a, b));
                }
            }
            step *= 2;
        }
        return;
#endif
    }
    int64_t step = 1;
    for (int64_t stage = 0; stage < stages; ++stage) {
        for (int64_t start = 0; start < tile_dim; start += step * 2) {
            for (int64_t i = 0; i < step; ++i) {
                int64_t offset0 = (start + i) * n_tiles;
                int64_t offset1 = (start + step + i) * n_tiles;
                for (int64_t t = 0; t < n_tiles; ++t) {
                    int32_t a = base[offset0 + t];
                    int32_t b = base[offset1 + t];
                    int32_t s = a + b;
                    int32_t d = a - b;
                    base[offset0 + t] = clamp_int16(s);
                    base[offset1 + t] = clamp_int16(d);
                }
            }
        }
        step *= 2;
    }
}

struct ExchangeIndexTable {
    int64_t phases = 0;
    int64_t groups = 0;
    int64_t per_group = 0;
    std::vector<int32_t> indices;

    const int32_t* ptr(int64_t phase, int64_t group, int64_t j) const {
        const int64_t offset = ((phase * groups + group) * per_group + j) * 4;
        return indices.data() + offset;
    }
};

ExchangeIndexTable build_exchange_index_table(int64_t tile_dim, int64_t n_tiles, int64_t exchange_dim) {
    ExchangeIndexTable table;
    if (exchange_dim <= 0) {
        return table;
    }
    table.phases = 4;
    table.groups = n_tiles / 4;
    table.per_group = exchange_dim / table.groups;
    table.indices.resize(table.phases * table.groups * table.per_group * 4);

    for (int64_t phase = 0; phase < table.phases; ++phase) {
        for (int64_t g = 0; g < table.groups; ++g) {
            const int64_t base_tile = g * 4;
            for (int64_t j = 0; j < table.per_group; ++j) {
                const int64_t start_idx = (4 * j + phase) % (4 * tile_dim);
                for (int k = 0; k < 4; ++k) {
                    const int64_t local_idx = (start_idx + k) % (4 * tile_dim);
                    const int64_t tile_off = local_idx / tile_dim;
                    const int64_t g_idx = local_idx - tile_off * tile_dim;
                    const int64_t idx = g_idx * n_tiles + (base_tile + tile_off);
                    const int64_t offset = ((phase * table.groups + g) * table.per_group + j) * 4 + k;
                    table.indices[offset] = static_cast<int32_t>(idx);
                }
            }
        }
    }
    return table;
}


} // namespace

template <bool Profiled>
torch::Tensor monoid_forward_float_impl(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output,
    ProfileTiming* timing
) {
    TORCH_CHECK(byte_tokens.device().is_cpu(), "byte_tokens must be on CPU");
    TORCH_CHECK(lengths.device().is_cpu(), "lengths must be on CPU");
    TORCH_CHECK(byte_tokens.scalar_type() == torch::kLong, "byte_tokens must be int64");
    TORCH_CHECK(lengths.scalar_type() == torch::kLong, "lengths must be int64");
    TORCH_CHECK(a_f32.scalar_type() == torch::kFloat32, "a_f32 must be float32");
    TORCH_CHECK(b_f32.scalar_type() == torch::kFloat32, "b_f32 must be float32");

    byte_tokens = byte_tokens.contiguous();
    lengths = lengths.contiguous();
    a_f32 = a_f32.contiguous();
    b_f32 = b_f32.contiguous();
    exchange_weight = exchange_weight.contiguous();
    if constexpr (Profiled) {
        TORCH_CHECK(timing != nullptr, "timing must be provided for profile runs");
    }
    Clock::time_point setup_start{};
    if constexpr (Profiled) {
        setup_start = Clock::now();
    }

    const auto batch = byte_tokens.size(0);
    const auto seq_len = byte_tokens.size(1);
    const auto d_state = a_f32.size(1);

    TORCH_CHECK(d_state == n_tiles * tile_dim, "d_state must equal n_tiles * tile_dim");

    auto output = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int64_t> block_lengths(batch, 1);
    const int64_t* __restrict__ lengths_ptr = lengths.data_ptr<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
        int64_t bl = (lengths_ptr[b] + microblock_size - 1) / microblock_size;
        block_lengths[b] = std::max<int64_t>(1, bl);
    }
    const int64_t total_blocks = (seq_len + microblock_size - 1) / microblock_size;

    auto state = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    float* __restrict__ state_ptr = assume_aligned_if<float, 64>(state.data_ptr<float>());
    const int64_t* __restrict__ tokens_ptr = byte_tokens.data_ptr<int64_t>();
    const float* __restrict__ a_ptr = assume_aligned_if<float, 64>(a_f32.data_ptr<float>());
    const float* __restrict__ b_ptr = assume_aligned_if<float, 64>(b_f32.data_ptr<float>());
    const float* __restrict__ exchange_ptr = exchange_weight.numel() > 0
        ? assume_aligned_if<float, 64>(exchange_weight.data_ptr<float>())
        : nullptr;
    float* __restrict__ output_ptr = assume_aligned_if<float, 64>(output.data_ptr<float>());

    const int64_t vocab_stride = d_state;
    const int64_t state_stride = d_state;
    check_aligned_if_required(state_ptr, kCacheLineBytes, "state");
    check_aligned_if_required(output_ptr, kCacheLineBytes, "output");
    check_aligned_if_required(a_ptr, kCacheLineBytes, "a_f32");
    check_aligned_if_required(b_ptr, kCacheLineBytes, "b_f32");
    if (exchange_ptr) {
        check_aligned_if_required(exchange_ptr, kCacheLineBytes, "exchange_weight_f32");
    }
    check_stride_alignment(state_stride, sizeof(float), kCacheLineBytes, "state_stride");
    check_stride_alignment(vocab_stride, sizeof(float), kCacheLineBytes, "vocab_stride");
    const float activation_T_f = static_cast<float>(activation_T);
    const bool activation_enabled = activation_T_f > 0.0f;
    const bool activation_is_one = std::abs(activation_T_f - 1.0f) < 1e-6f;
    const float inv_activation_T = activation_enabled ? (1.0f / activation_T_f) : 0.0f;
    const float inj_scale = 1.0f / static_cast<float>(1 << inj_shift);
    const bool exchange_enabled = use_exchange && exchange_ptr && exchange_every > 0;
    const int64_t exchange_dim = exchange_enabled ? exchange_weight.size(0) : 0;
    const int64_t groups = exchange_enabled ? (n_tiles / 4) : 0;
    const int64_t per_group = exchange_enabled ? (exchange_dim / groups) : 0;
    const int64_t exchange_dim_padded = exchange_enabled ? cacheline_pad_elems<float>(exchange_dim) : 0;
    if (exchange_enabled) {
        check_stride_alignment(exchange_dim, sizeof(float), kCacheLineBytes, "exchange_stride");
    }
    const bool do_normalize = normalize_output != 0;
    const bool in_parallel = at::in_parallel_region();
    const bool use_parallel = !in_parallel && batch >= 2 && at::get_num_threads() > 1;
    const int64_t mix_stages = butterfly_mix_stages(tile_dim);
    const bool second_activation_enabled = use_second_activation && activation_enabled;
    if (exchange_enabled) {
        TORCH_CHECK(exchange_weight.dim() == 2, "exchange_weight must be (exchange_dim, exchange_dim)");
        TORCH_CHECK(exchange_weight.size(0) == exchange_weight.size(1), "exchange_weight must be square.");
        TORCH_CHECK(n_tiles % 4 == 0, "n_tiles must be divisible by 4 for exchange.");
        TORCH_CHECK(exchange_dim % groups == 0, "exchange_dim must be divisible by n_tiles/4.");
    }
    const ExchangeIndexTable exchange_indices = exchange_enabled
        ? build_exchange_index_table(tile_dim, n_tiles, exchange_dim)
        : ExchangeIndexTable{};

    const bool exchange_parallel = (!use_parallel) && exchange_enabled && (at::get_num_threads() > 1) && !in_parallel;
    const int64_t exchange_grain = exchange_parallel ? parallel_grain_size(exchange_dim, 16, 1) : 0;
    const int64_t batch_grain = use_parallel ? parallel_grain_size(batch) : 0;
    AlignedVector<float> exchange_u_shared;
    AlignedVector<float> exchange_v_shared;
    if (exchange_enabled && exchange_parallel) {
        exchange_u_shared.assign(exchange_dim_padded, 0.0f);
        exchange_v_shared.assign(exchange_dim_padded, 0.0f);
    }
    if constexpr (Profiled) {
        timing->setup_ms += elapsed_ms(setup_start);
    }
    auto process_batch = [&](int64_t b0, int64_t b1) {
        AlignedVector<float>* u_ptr = nullptr;
        AlignedVector<float>* v_ptr = nullptr;
        if (exchange_enabled) {
            if (exchange_parallel) {
                u_ptr = &exchange_u_shared;
                v_ptr = &exchange_v_shared;
            } else {
                thread_local AlignedVector<float> u;
                thread_local AlignedVector<float> v;
                if (u.size() != static_cast<size_t>(exchange_dim_padded)) {
                    u.assign(exchange_dim_padded, 0.0f);
                    v.assign(exchange_dim_padded, 0.0f);
                }
                u_ptr = &u;
                v_ptr = &v;
            }
        }
        for (int64_t b = b0; b < b1; ++b) {
            float* __restrict__ s_row = assume_aligned_if<float, 64>(state_ptr + b * state_stride);
            float* __restrict__ out_row = assume_aligned_if<float, 64>(output_ptr + b * d_state);
            const int64_t* __restrict__ tokens_row = tokens_ptr + b * seq_len;
            const int64_t block_count = total_blocks;
            for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
                const int64_t start = block_idx * microblock_size;
                const int64_t end = std::min<int64_t>(start + microblock_size, seq_len);
                const int64_t pos_end = std::min<int64_t>(end, lengths_ptr[b]);
                profile_block<Profiled>(timing, &ProfileTiming::recurrence_ms, [&]() {
                    for (int64_t pos = start; pos < pos_end; ++pos) {
                        int64_t code = tokens_row[pos];
                        const float* __restrict__ a_row = assume_aligned_if<float, 64>(a_ptr + code * vocab_stride);
                        const float* __restrict__ b_row = assume_aligned_if<float, 64>(b_ptr + code * vocab_stride);
                        recurrence_f32(s_row, a_row, b_row, d_state);
                    }
                });

                profile_block<Profiled>(timing, &ProfileTiming::butterfly_ms, [&]() {
                    butterfly_mix_float_interleaved_stages(s_row, tile_dim, n_tiles, mix_stages);
                });

                if (activation_enabled) {
                    profile_block<Profiled>(timing, &ProfileTiming::activation_ms, [&]() {
                        if (activation_is_one) {
#pragma omp simd
                            for (int64_t d = 0; d < d_state; ++d) {
                                s_row[d] = tanh_eval(s_row[d]);
                            }
                        } else {
#pragma omp simd
                            for (int64_t d = 0; d < d_state; ++d) {
                                s_row[d] = activation_T_f * tanh_eval(s_row[d] * inv_activation_T);
                            }
                        }
                    });
                }

                if (exchange_enabled && ((block_idx + 1) % exchange_every == 0)) {
                    profile_block<Profiled>(timing, &ProfileTiming::exchange_ms, [&]() {
                        AlignedVector<float>& u = *u_ptr;
                        AlignedVector<float>& v = *v_ptr;
                        int64_t phase = block_idx % 4;
                        std::fill(u.begin(), u.end(), 0.0f);
                        std::fill(v.begin(), v.end(), 0.0f);
                        for (int64_t g = 0; g < groups; ++g) {
                            for (int64_t j = 0; j < per_group; ++j) {
                                const int32_t* idx = exchange_indices.ptr(phase, g, j);
                                float sum_val = s_row[idx[0]];
                                sum_val += s_row[idx[1]];
                                sum_val += s_row[idx[2]];
                                sum_val += s_row[idx[3]];
                                u[g * per_group + j] = sum_val;
                            }
                        }
                        const float* __restrict__ u_data = assume_aligned_if<float, 64>(u.data());
                        float* __restrict__ v_data = assume_aligned_if<float, 64>(v.data());
                        if (exchange_parallel) {
                            at::parallel_for(0, exchange_dim, exchange_grain, [&](int64_t r0, int64_t r1) {
                                for (int64_t r = r0; r < r1; ++r) {
                                    const float* w_row = exchange_ptr + r * exchange_dim;
                                    v_data[r] = dot_f32(w_row, u_data, exchange_dim);
                                }
                            });
                        } else {
                            for (int64_t r = 0; r < exchange_dim; ++r) {
                                const float* w_row = exchange_ptr + r * exchange_dim;
                                v_data[r] = dot_f32(w_row, u_data, exchange_dim);
                            }
                        }
                        for (int64_t g = 0; g < groups; ++g) {
                            for (int64_t j = 0; j < per_group; ++j) {
                                const float inj = v_data[g * per_group + j] * inj_scale;
                                const int32_t* idx = exchange_indices.ptr(phase, g, j);
                                s_row[idx[0]] += inj;
                                s_row[idx[1]] += inj;
                                s_row[idx[2]] -= inj;
                                s_row[idx[3]] -= inj;
                            }
                        }
                    });
                    if (second_activation_enabled) {
                        profile_block<Profiled>(timing, &ProfileTiming::activation_ms, [&]() {
                            if (activation_is_one) {
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    s_row[d] = tanh_eval(s_row[d]);
                                }
                            } else {
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    s_row[d] = activation_T_f * tanh_eval(s_row[d] * inv_activation_T);
                                }
                            }
                        });
                    }
                }

                const bool include_block = (pool_strategy == 1) || (block_idx < block_lengths[b]);
                if (include_block) {
                    profile_block<Profiled>(timing, &ProfileTiming::pooling_ms, [&]() {
                        if (pool_strategy == 1) {
                            if (do_normalize) {
                                float norm = dot_f32(s_row, s_row, d_state);
                                float inv = norm > 0.0f ? (1.0f / std::sqrt(norm)) : 1.0f;
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    out_row[d] = s_row[d] * inv;
                                }
                            } else {
                                std::memcpy(out_row, s_row, sizeof(float) * d_state);
                            }
                        } else {
                            if (do_normalize) {
                                float norm = dot_f32(s_row, s_row, d_state);
                                float inv = norm > 0.0f ? (1.0f / std::sqrt(norm)) : 1.0f;
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    out_row[d] += s_row[d] * inv;
                                }
                            } else {
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    out_row[d] += s_row[d];
                                }
                            }
                        }
                    });
                }
            }
            if (pool_strategy == 0) {
                float inv = 1.0f / static_cast<float>(block_lengths[b]);
                profile_block<Profiled>(timing, &ProfileTiming::pooling_ms, [&]() {
#pragma omp simd
                    for (int64_t d = 0; d < d_state; ++d) {
                        out_row[d] *= inv;
                    }
                });
            }
        }
    };
    if (use_parallel) {
        at::parallel_for(0, batch, batch_grain, process_batch);
    } else {
        process_batch(0, batch);
    }

    return output;
}

torch::Tensor monoid_forward_float(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_float_impl<false>(
        byte_tokens,
        lengths,
        a_f32,
        b_f32,
        exchange_weight,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_T,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output,
        nullptr
    );
}

std::tuple<torch::Tensor, std::map<std::string, double>> monoid_forward_float_profile(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    ProfileTiming timing;
    auto output = monoid_forward_float_impl<true>(
        byte_tokens,
        lengths,
        a_f32,
        b_f32,
        exchange_weight,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_T,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output,
        &timing
    );
    return {output, profile_to_map(timing)};
}

template <bool Profiled>
torch::Tensor monoid_forward_float_stacked_impl(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor proj_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output,
    ProfileTiming* timing
) {
    TORCH_CHECK(byte_tokens.device().is_cpu(), "byte_tokens must be on CPU");
    TORCH_CHECK(lengths.device().is_cpu(), "lengths must be on CPU");
    TORCH_CHECK(byte_tokens.scalar_type() == torch::kLong, "byte_tokens must be int64");
    TORCH_CHECK(lengths.scalar_type() == torch::kLong, "lengths must be int64");
    TORCH_CHECK(a_f32.scalar_type() == torch::kFloat32, "a_f32 must be float32");
    TORCH_CHECK(b_f32.scalar_type() == torch::kFloat32, "b_f32 must be float32");
    TORCH_CHECK(a_f32.dim() == 3, "a_f32 must be (n_layers, vocab, d_state)");
    TORCH_CHECK(b_f32.dim() == 3, "b_f32 must be (n_layers, vocab, d_state)");

    byte_tokens = byte_tokens.contiguous();
    lengths = lengths.contiguous();
    a_f32 = a_f32.contiguous();
    b_f32 = b_f32.contiguous();
    exchange_weight = exchange_weight.contiguous();
    ln_weight = ln_weight.contiguous();
    ln_bias = ln_bias.contiguous();
    proj_weight = proj_weight.contiguous();
    if constexpr (Profiled) {
        TORCH_CHECK(timing != nullptr, "timing must be provided for profile runs");
    }
    Clock::time_point setup_start{};
    if constexpr (Profiled) {
        setup_start = Clock::now();
    }

    const auto batch = byte_tokens.size(0);
    const auto seq_len = byte_tokens.size(1);
    const auto n_layers = a_f32.size(0);
    const auto vocab = a_f32.size(1);
    const auto d_state = a_f32.size(2);

    TORCH_CHECK(b_f32.size(0) == n_layers, "b_f32 must match a_f32 layers");
    TORCH_CHECK(b_f32.size(1) == vocab && b_f32.size(2) == d_state, "b_f32 must match a_f32 shape");
    TORCH_CHECK(d_state == n_tiles * tile_dim, "d_state must equal n_tiles * tile_dim");
    if (exchange_weight.numel() > 0) {
        TORCH_CHECK(exchange_weight.dim() == 3, "exchange_weight must be (n_layers, exchange_dim, exchange_dim)");
        TORCH_CHECK(exchange_weight.size(0) == n_layers, "exchange_weight must match n_layers");
        TORCH_CHECK(exchange_weight.size(1) == exchange_weight.size(2), "exchange_weight must be square");
    }
    if (ln_weight.numel() > 0) {
        TORCH_CHECK(ln_weight.dim() == 2, "ln_weight must be (n_layers, d_state)");
        TORCH_CHECK(ln_weight.size(0) == n_layers && ln_weight.size(1) == d_state, "ln_weight shape mismatch");
    }
    if (ln_bias.numel() > 0) {
        TORCH_CHECK(ln_bias.dim() == 2, "ln_bias must be (n_layers, d_state)");
        TORCH_CHECK(ln_bias.size(0) == n_layers && ln_bias.size(1) == d_state, "ln_bias shape mismatch");
    }
    if (proj_weight.numel() > 0) {
        TORCH_CHECK(proj_weight.dim() == 2, "proj_weight must be (out_dim, d_state)");
        TORCH_CHECK(proj_weight.size(1) == d_state, "proj_weight input dim mismatch");
    }


    auto output = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int64_t> block_lengths(batch, 1);
    const int64_t* __restrict__ lengths_ptr = lengths.data_ptr<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
        int64_t bl = (lengths_ptr[b] + microblock_size - 1) / microblock_size;
        block_lengths[b] = std::max<int64_t>(1, bl);
    }
    const int64_t total_blocks = (seq_len + microblock_size - 1) / microblock_size;

    auto state = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    auto state_block = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    float* __restrict__ state_ptr = assume_aligned_if<float, 64>(state.data_ptr<float>());
    float* __restrict__ state_block_ptr = assume_aligned_if<float, 64>(state_block.data_ptr<float>());
    const int64_t* __restrict__ tokens_ptr = byte_tokens.data_ptr<int64_t>();
    const float* __restrict__ a_ptr = assume_aligned_if<float, 64>(a_f32.data_ptr<float>());
    const float* __restrict__ b_ptr = assume_aligned_if<float, 64>(b_f32.data_ptr<float>());
    const float* __restrict__ exchange_ptr = exchange_weight.numel() > 0
        ? assume_aligned_if<float, 64>(exchange_weight.data_ptr<float>())
        : nullptr;
    const float* __restrict__ ln_weight_ptr = ln_weight.numel() > 0
        ? assume_aligned_if<float, 64>(ln_weight.data_ptr<float>())
        : nullptr;
    const float* __restrict__ ln_bias_ptr = ln_bias.numel() > 0
        ? assume_aligned_if<float, 64>(ln_bias.data_ptr<float>())
        : nullptr;
    const float* __restrict__ proj_ptr = proj_weight.numel() > 0
        ? assume_aligned_if<float, 64>(proj_weight.data_ptr<float>())
        : nullptr;
    float* __restrict__ output_ptr = assume_aligned_if<float, 64>(output.data_ptr<float>());

    const int64_t vocab_stride = d_state;
    const int64_t state_stride = d_state;
    const int64_t layer_stride = vocab * d_state;
    const int64_t exchange_stride = exchange_weight.numel() > 0
        ? exchange_weight.size(1) * exchange_weight.size(2)
        : 0;
    const int64_t exchange_dim = exchange_weight.numel() > 0 ? exchange_weight.size(1) : 0;
    const bool exchange_configured = use_exchange && exchange_ptr;
    const bool exchange_active = exchange_configured && exchange_every > 0;
    const int64_t groups = exchange_configured ? n_tiles / 4 : 0;
    const int64_t per_group = (exchange_configured && groups > 0) ? exchange_dim / groups : 0;
    const int64_t exchange_dim_padded = exchange_configured ? cacheline_pad_elems<float>(exchange_dim) : 0;
    check_aligned_if_required(state_ptr, kCacheLineBytes, "state");
    check_aligned_if_required(state_block_ptr, kCacheLineBytes, "state_block");
    check_aligned_if_required(output_ptr, kCacheLineBytes, "output");
    check_aligned_if_required(a_ptr, kCacheLineBytes, "a_f32");
    check_aligned_if_required(b_ptr, kCacheLineBytes, "b_f32");
    if (exchange_ptr) {
        check_aligned_if_required(exchange_ptr, kCacheLineBytes, "exchange_weight_f32");
    }
    if (ln_weight_ptr) {
        check_aligned_if_required(ln_weight_ptr, kCacheLineBytes, "ln_weight");
    }
    if (ln_bias_ptr) {
        check_aligned_if_required(ln_bias_ptr, kCacheLineBytes, "ln_bias");
    }
    if (proj_ptr) {
        check_aligned_if_required(proj_ptr, kCacheLineBytes, "proj_weight");
    }
    check_stride_alignment(state_stride, sizeof(float), kCacheLineBytes, "state_stride");
    check_stride_alignment(vocab_stride, sizeof(float), kCacheLineBytes, "vocab_stride");
    check_stride_alignment(layer_stride, sizeof(float), kCacheLineBytes, "layer_stride");
    if (exchange_configured) {
        check_stride_alignment(exchange_stride, sizeof(float), kCacheLineBytes, "exchange_stride");
        check_stride_alignment(exchange_dim, sizeof(float), kCacheLineBytes, "exchange_row_stride");
    }
    if (exchange_configured) {
        TORCH_CHECK(n_tiles % 4 == 0, "n_tiles must be divisible by 4 for exchange.");
        TORCH_CHECK(exchange_dim % groups == 0, "exchange_dim must be divisible by n_tiles/4.");
    }

    const float activation_T_f = static_cast<float>(activation_T);
    const bool activation_enabled = activation_T_f > 0.0f;
    const bool activation_is_one = std::abs(activation_T_f - 1.0f) < 1e-6f;
    const float inv_activation_T = activation_enabled ? (1.0f / activation_T_f) : 0.0f;
    const float inj_scale = 1.0f / static_cast<float>(1 << inj_shift);
    const bool use_ln = n_layers > 1;
    const bool do_normalize = normalize_output != 0;
    const bool in_parallel = at::in_parallel_region();
    const bool use_parallel = !in_parallel && batch >= 2 && at::get_num_threads() > 1;
    const int64_t mix_stages = butterfly_mix_stages(tile_dim);
    const bool second_activation_enabled = use_second_activation && activation_enabled;
    const ExchangeIndexTable exchange_indices = exchange_active
        ? build_exchange_index_table(tile_dim, n_tiles, exchange_dim)
        : ExchangeIndexTable{};
    const bool exchange_parallel = (!use_parallel) && exchange_active && (at::get_num_threads() > 1) && !in_parallel;
    const int64_t exchange_grain = exchange_parallel ? parallel_grain_size(exchange_dim, 16, 1) : 0;
    const int64_t batch_grain = use_parallel ? parallel_grain_size(batch) : 0;
    const bool proj_parallel = (batch == 1) && (at::get_num_threads() > 1) && !exchange_parallel && !in_parallel;
    AlignedVector<float> exchange_u_shared;
    AlignedVector<float> exchange_v_shared;
    if (exchange_active && exchange_parallel) {
        exchange_u_shared.assign(exchange_dim_padded, 0.0f);
        exchange_v_shared.assign(exchange_dim_padded, 0.0f);
    }
    if constexpr (Profiled) {
        timing->setup_ms += elapsed_ms(setup_start);
    }

    auto process_batch = [&](int64_t b0, int64_t b1) {
        AlignedVector<float>* u_ptr = nullptr;
        AlignedVector<float>* v_ptr = nullptr;
        if (exchange_active) {
            if (exchange_parallel) {
                u_ptr = &exchange_u_shared;
                v_ptr = &exchange_v_shared;
            } else {
                thread_local AlignedVector<float> u;
                thread_local AlignedVector<float> v;
                if (u.size() != static_cast<size_t>(exchange_dim_padded)) {
                    u.assign(exchange_dim_padded, 0.0f);
                    v.assign(exchange_dim_padded, 0.0f);
                }
                u_ptr = &u;
                v_ptr = &v;
            }
        }
        for (int64_t b = b0; b < b1; ++b) {
            float* __restrict__ state_row = assume_aligned_if<float, 64>(state_ptr + b * state_stride);
            float* __restrict__ block_row = assume_aligned_if<float, 64>(state_block_ptr + b * state_stride);
            float* __restrict__ out_row = assume_aligned_if<float, 64>(output_ptr + b * d_state);
            const int64_t* __restrict__ tokens_row = tokens_ptr + b * seq_len;
            const int64_t block_count = total_blocks;

            for (int64_t layer = 0; layer < n_layers; ++layer) {
                const float* __restrict__ a_layer = assume_aligned_if<float, 64>(a_ptr + layer * layer_stride);
                const float* __restrict__ b_layer = assume_aligned_if<float, 64>(b_ptr + layer * layer_stride);
                const float* __restrict__ exchange_layer = exchange_ptr
                    ? assume_aligned_if<float, 64>(exchange_ptr + layer * exchange_stride)
                    : nullptr;
                const float* __restrict__ ln_w = ln_weight_ptr
                    ? assume_aligned_if<float, 64>(ln_weight_ptr + layer * d_state)
                    : nullptr;
                const float* __restrict__ ln_b = ln_bias_ptr
                    ? assume_aligned_if<float, 64>(ln_bias_ptr + layer * d_state)
                    : nullptr;
                const bool exchange_enabled = exchange_active && exchange_layer;

                if (use_ln) {
                    profile_block<Profiled>(timing, &ProfileTiming::layer_norm_ms, [&]() {
                        layer_norm_row(state_row, block_row, ln_w, ln_b, d_state);
                    });
                } else {
                    std::memcpy(block_row, state_row, sizeof(float) * d_state);
                }

                for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
                    const int64_t start = block_idx * microblock_size;
                    const int64_t end = std::min<int64_t>(start + microblock_size, seq_len);
                    const int64_t pos_end = std::min<int64_t>(end, lengths_ptr[b]);
                    profile_block<Profiled>(timing, &ProfileTiming::recurrence_ms, [&]() {
                        for (int64_t pos = start; pos < pos_end; ++pos) {
                            int64_t code = tokens_row[pos];
                            const float* __restrict__ a_row = assume_aligned_if<float, 64>(a_layer + code * vocab_stride);
                            const float* __restrict__ b_row = assume_aligned_if<float, 64>(b_layer + code * vocab_stride);
                            recurrence_f32(block_row, a_row, b_row, d_state);
                        }
                    });

                    profile_block<Profiled>(timing, &ProfileTiming::butterfly_ms, [&]() {
                        butterfly_mix_float_interleaved_stages(block_row, tile_dim, n_tiles, mix_stages);
                    });

                    if (activation_enabled) {
                        profile_block<Profiled>(timing, &ProfileTiming::activation_ms, [&]() {
                            if (activation_is_one) {
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    block_row[d] = tanh_eval(block_row[d]);
                                }
                            } else {
#pragma omp simd
                                for (int64_t d = 0; d < d_state; ++d) {
                                    block_row[d] = activation_T_f * tanh_eval(block_row[d] * inv_activation_T);
                                }
                            }
                        });
                    }

                    if (exchange_enabled && ((block_idx + 1) % exchange_every == 0)) {
                        profile_block<Profiled>(timing, &ProfileTiming::exchange_ms, [&]() {
                            AlignedVector<float>& u = *u_ptr;
                            AlignedVector<float>& v = *v_ptr;
                            int64_t phase = block_idx % 4;
                            std::fill(u.begin(), u.end(), 0.0f);
                            std::fill(v.begin(), v.end(), 0.0f);
                            for (int64_t g = 0; g < groups; ++g) {
                                for (int64_t j = 0; j < per_group; ++j) {
                                    const int32_t* idx = exchange_indices.ptr(phase, g, j);
                                    float sum_val = block_row[idx[0]];
                                    sum_val += block_row[idx[1]];
                                    sum_val += block_row[idx[2]];
                                    sum_val += block_row[idx[3]];
                                    u[g * per_group + j] = sum_val;
                                }
                            }
                            const float* __restrict__ u_data = assume_aligned_if<float, 64>(u.data());
                            float* __restrict__ v_data = assume_aligned_if<float, 64>(v.data());
                            if (exchange_parallel) {
                                at::parallel_for(0, exchange_dim, exchange_grain, [&](int64_t r0, int64_t r1) {
                                    for (int64_t r = r0; r < r1; ++r) {
                                        const float* w_row = exchange_layer + r * exchange_dim;
                                        v_data[r] = dot_f32(w_row, u_data, exchange_dim);
                                    }
                                });
                            } else {
                                for (int64_t r = 0; r < exchange_dim; ++r) {
                                    const float* w_row = exchange_layer + r * exchange_dim;
                                    v_data[r] = dot_f32(w_row, u_data, exchange_dim);
                                }
                            }
                            for (int64_t g = 0; g < groups; ++g) {
                                for (int64_t j = 0; j < per_group; ++j) {
                                    float inj = v_data[g * per_group + j] * inj_scale;
                                    const int32_t* idx = exchange_indices.ptr(phase, g, j);
                                    block_row[idx[0]] += inj;
                                    block_row[idx[1]] += inj;
                                    block_row[idx[2]] -= inj;
                                    block_row[idx[3]] -= inj;
                                }
                            }
                        });
                        if (second_activation_enabled) {
                            profile_block<Profiled>(timing, &ProfileTiming::activation_ms, [&]() {
                                if (activation_is_one) {
#pragma omp simd
                                    for (int64_t d = 0; d < d_state; ++d) {
                                        block_row[d] = tanh_eval(block_row[d]);
                                    }
                                } else {
#pragma omp simd
                                    for (int64_t d = 0; d < d_state; ++d) {
                                        block_row[d] = activation_T_f * tanh_eval(block_row[d] * inv_activation_T);
                                    }
                                }
                            });
                        }
                    }

                    if (layer == (n_layers - 1)) {
                        const bool include_block = (pool_strategy == 1) || (block_idx < block_lengths[b]);
                        if (include_block) {
                            profile_block<Profiled>(timing, &ProfileTiming::pooling_ms, [&]() {
                                if (pool_strategy == 1) {
                                    if (do_normalize) {
                                        float norm = dot_f32(block_row, block_row, d_state);
                                        float inv = norm > 0.0f ? (1.0f / std::sqrt(norm)) : 1.0f;
#pragma omp simd
                                        for (int64_t d = 0; d < d_state; ++d) {
                                            out_row[d] = block_row[d] * inv;
                                        }
                                    } else {
                                        std::memcpy(out_row, block_row, sizeof(float) * d_state);
                                    }
                                } else {
                                    if (do_normalize) {
                                        float norm = dot_f32(block_row, block_row, d_state);
                                        float inv = norm > 0.0f ? (1.0f / std::sqrt(norm)) : 1.0f;
#pragma omp simd
                                        for (int64_t d = 0; d < d_state; ++d) {
                                            out_row[d] += block_row[d] * inv;
                                        }
                                    } else {
#pragma omp simd
                                        for (int64_t d = 0; d < d_state; ++d) {
                                            out_row[d] += block_row[d];
                                        }
                                    }
                                }
                            });
                        }
                    }
                }

                profile_block<Profiled>(timing, &ProfileTiming::pooling_ms, [&]() {
#pragma omp simd
                    for (int64_t d = 0; d < d_state; ++d) {
                        state_row[d] += block_row[d];
                    }
                });
            }

            if (pool_strategy == 0) {
                float inv = 1.0f / static_cast<float>(block_lengths[b]);
                profile_block<Profiled>(timing, &ProfileTiming::pooling_ms, [&]() {
#pragma omp simd
                    for (int64_t d = 0; d < d_state; ++d) {
                        out_row[d] *= inv;
                    }
                });
            }
        }
    };
    if (use_parallel) {
        at::parallel_for(0, batch, batch_grain, process_batch);
    } else {
        process_batch(0, batch);
    }

    if (proj_ptr) {
        const int64_t out_dim = proj_weight.size(0);
        auto proj_out = torch::zeros({batch, out_dim}, torch::TensorOptions().dtype(torch::kFloat32));
        float* __restrict__ proj_out_ptr = assume_aligned_if<float, 64>(proj_out.data_ptr<float>());
        const bool proj_intrinsics = force_proj_intrinsics();
        profile_block<Profiled>(timing, &ProfileTiming::proj_ms, [&]() {
            if (proj_parallel) {
                const float* __restrict__ in_row = output_ptr;
                float* __restrict__ out_row = proj_out_ptr;
                const int64_t grain = parallel_grain_size(out_dim, 16, 1);
                at::parallel_for(0, out_dim, grain, [&](int64_t o0, int64_t o1) {
                    if (proj_intrinsics) {
                        for (int64_t o = o0; o < o1; ++o) {
                            const float* __restrict__ w_row = proj_ptr + o * d_state;
                            out_row[o] = dot_f32(in_row, w_row, d_state);
                        }
                    } else {
                        for (int64_t o = o0; o < o1; ++o) {
                            const float* __restrict__ w_row = proj_ptr + o * d_state;
                            float acc = 0.0f;
#pragma omp simd reduction(+:acc)
                            for (int64_t d = 0; d < d_state; ++d) {
                                acc += in_row[d] * w_row[d];
                            }
                            out_row[o] = acc;
                        }
                    }
                });
                if (do_normalize) {
                    normalize_l2(out_row, out_dim);
                }
            } else {
                auto project_batch = [&](int64_t b0, int64_t b1) {
                    for (int64_t b = b0; b < b1; ++b) {
                        const float* __restrict__ in_row = output_ptr + b * d_state;
                        float* __restrict__ out_row = proj_out_ptr + b * out_dim;
                        if (proj_intrinsics) {
                            for (int64_t o = 0; o < out_dim; ++o) {
                                const float* __restrict__ w_row = proj_ptr + o * d_state;
                                out_row[o] = dot_f32(in_row, w_row, d_state);
                            }
                        } else {
                            for (int64_t o = 0; o < out_dim; ++o) {
                                const float* __restrict__ w_row = proj_ptr + o * d_state;
                                float acc = 0.0f;
#pragma omp simd reduction(+:acc)
                                for (int64_t d = 0; d < d_state; ++d) {
                                    acc += in_row[d] * w_row[d];
                                }
                                out_row[o] = acc;
                            }
                        }
                        if (do_normalize) {
                            normalize_l2(out_row, out_dim);
                        }
                    }
                };
                if (use_parallel) {
                    at::parallel_for(0, batch, 1, project_batch);
                } else {
                    project_batch(0, batch);
                }
            }
        });
        return proj_out;
    }

    return output;
}

torch::Tensor monoid_forward_float_stacked(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor proj_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_float_stacked_impl<false>(
        byte_tokens,
        lengths,
        a_f32,
        b_f32,
        exchange_weight,
        ln_weight,
        ln_bias,
        proj_weight,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_T,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output,
        nullptr
    );
}

std::tuple<torch::Tensor, std::map<std::string, double>> monoid_forward_float_stacked_profile(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor proj_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    ProfileTiming timing;
    auto output = monoid_forward_float_stacked_impl<true>(
        byte_tokens,
        lengths,
        a_f32,
        b_f32,
        exchange_weight,
        ln_weight,
        ln_bias,
        proj_weight,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_T,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output,
        &timing
    );
    return {output, profile_to_map(timing)};
}

struct QuantLayerParams {
    int64_t batch = 0;
    int64_t seq_len = 0;
    int64_t d_state = 0;
    int64_t microblock_size = 0;
    int64_t n_tiles = 0;
    int64_t tile_dim = 0;
    int64_t activation_shift = 0;
    int64_t activation_T_q15 = 0;
    int64_t b_shift = 0;
    int64_t exchange_every = 0;
    int64_t pool_strategy = 0;
    int64_t normalize_output = 0;
    bool do_pool = true;
    int64_t total_blocks = 0;
    int64_t vocab_stride = 0;
    int64_t state_stride = 0;
    int64_t exchange_dim = 0;
    int64_t groups = 0;
    int64_t per_group = 0;
    int64_t exchange_dim_padded = 0;
    int64_t exchange_grain = 0;
    int32_t inj_shift_i32 = 0;
    float inv_q15 = 0.0f;
    bool exchange_active = false;
    bool exchange_parallel = false;
    bool use_second_activation = false;
    const int64_t* lengths_ptr = nullptr;
    const int64_t* tokens_ptr = nullptr;
    const int16_t* a_ptr = nullptr;
    const int8_t* b_ptr = nullptr;
    const int16_t* lut_ptr = nullptr;
    const int8_t* exchange_ptr = nullptr;
    const int8_t* exchange_shift_ptr = nullptr;
    int32_t* state32_ptr = nullptr;
    int16_t* state16_ptr = nullptr;
    float* output_ptr = nullptr;
    const ExchangeIndexTable* exchange_indices = nullptr;
    const int64_t* block_lengths = nullptr;
};

static void run_quant_layer_range(
    const QuantLayerParams& params,
    int64_t b0,
    int64_t b1,
    AlignedVector<int32_t>* exchange_u_shared,
    AlignedVector<int32_t>* exchange_v_shared,
    AlignedVector<int16_t>* exchange_u16_shared
) {
    thread_local AlignedVector<float> temp;
    AlignedVector<int32_t>* u_ptr = nullptr;
    AlignedVector<int32_t>* v_ptr = nullptr;
    AlignedVector<int16_t>* u16_ptr = nullptr;
    if (params.do_pool) {
        const int64_t temp_padded = cacheline_pad_elems<float>(params.d_state);
        if (temp.size() != static_cast<size_t>(temp_padded)) {
            temp.assign(temp_padded, 0.0f);
        }
    }
    if (params.exchange_active) {
        if (params.exchange_parallel) {
            u_ptr = exchange_u_shared;
            v_ptr = exchange_v_shared;
            u16_ptr = exchange_u16_shared;
        } else {
            thread_local AlignedVector<int32_t> u;
            thread_local AlignedVector<int32_t> v;
            thread_local AlignedVector<int16_t> u16;
            if (u.size() != static_cast<size_t>(params.exchange_dim_padded)) {
                u.assign(params.exchange_dim_padded, 0);
                v.assign(params.exchange_dim_padded, 0);
            }
            u_ptr = &u;
            v_ptr = &v;
            if (u16.size() != static_cast<size_t>(params.exchange_dim_padded)) {
                u16.assign(params.exchange_dim_padded, 0);
            }
            u16_ptr = &u16;
        }
    }
    for (int64_t b = b0; b < b1; ++b) {
        int32_t* __restrict__ s_row32 = assume_aligned_if<int32_t, 64>(
            params.state32_ptr + b * params.state_stride
        );
        int16_t* __restrict__ s_row16 = assume_aligned_if<int16_t, 64>(
            params.state16_ptr + b * params.state_stride
        );
        float* __restrict__ out_row = assume_aligned_if<float, 64>(params.output_ptr + b * params.d_state);
        const int64_t* __restrict__ tokens_row = params.tokens_ptr + b * params.seq_len;
        const int64_t block_count = params.total_blocks;

        for (int64_t block_idx = 0; block_idx < block_count; ++block_idx) {
            const int64_t start = block_idx * params.microblock_size;
            const int64_t end = std::min<int64_t>(start + params.microblock_size, params.seq_len);
            const int64_t pos_end = std::min<int64_t>(end, params.lengths_ptr[b]);
            for (int64_t pos = start; pos < pos_end; ++pos) {
                int64_t code = tokens_row[pos];
                const int16_t* __restrict__ a_row = assume_aligned_if<int16_t, 64>(
                    params.a_ptr + code * params.vocab_stride
                );
                const int8_t* __restrict__ b_row = params.b_ptr + code * params.vocab_stride;
                recurrence_q15(s_row32, a_row, b_row, params.d_state, static_cast<int32_t>(params.b_shift));
            }

            clamp_store_s32_to_s16(s_row32, s_row16, params.d_state);

            butterfly_mix_int16_interleaved(s_row16, params.tile_dim, params.n_tiles);

            for (int64_t d = 0; d < params.d_state; ++d) {
                int32_t x = static_cast<int32_t>(s_row16[d]) >> params.activation_shift;
                x = clamp_int32(x, -128, 127) + 128;
                int16_t tval = params.lut_ptr[x];
                int32_t out = (static_cast<int32_t>(tval) * params.activation_T_q15) >> 15;
                s_row16[d] = clamp_int16(out);
            }

            if (params.exchange_active && ((block_idx + 1) % params.exchange_every == 0)) {
                AlignedVector<int32_t>& u = *u_ptr;
                AlignedVector<int32_t>& v = *v_ptr;
                int64_t phase = block_idx % 4;
                std::fill(u.begin(), u.end(), 0);
                std::fill(v.begin(), v.end(), 0);
                for (int64_t g = 0; g < params.groups; ++g) {
                    for (int64_t j = 0; j < params.per_group; ++j) {
                        const int32_t* idx = params.exchange_indices->ptr(phase, g, j);
                        const int32_t sum = static_cast<int32_t>(s_row16[idx[0]])
                            + static_cast<int32_t>(s_row16[idx[1]])
                            + static_cast<int32_t>(s_row16[idx[2]])
                            + static_cast<int32_t>(s_row16[idx[3]]);
                        u[g * params.per_group + j] = sum;
                    }
                }
                const int32_t* __restrict__ u_data = assume_aligned_if<int32_t, 64>(u.data());
                int32_t* __restrict__ v_data = assume_aligned_if<int32_t, 64>(v.data());
                const int16_t* u16_data = nullptr;
                bool use_u16 = false;
                if (u16_ptr) {
                    int32_t max_abs = 0;
                    for (int64_t i = 0; i < params.exchange_dim; ++i) {
                        const int32_t vabs = std::abs(u_data[i]);
                        if (vabs > max_abs) {
                            max_abs = vabs;
                        }
                    }
                    if (max_abs <= 32767) {
                        int16_t* u16_out = u16_ptr->data();
                        for (int64_t i = 0; i < params.exchange_dim; ++i) {
                            u16_out[i] = static_cast<int16_t>(u_data[i]);
                        }
                        u16_data = u16_out;
                        use_u16 = true;
                    }
                }
                if (params.exchange_parallel) {
                    at::parallel_for(0, params.exchange_dim, params.exchange_grain, [&](int64_t r0, int64_t r1) {
                        for (int64_t r = r0; r < r1; ++r) {
                            const int8_t* w_row = params.exchange_ptr + r * params.exchange_dim;
                            int32_t acc = use_u16
                                ? dot_i8_i16(w_row, u16_data, params.exchange_dim)
                                : dot_i8_i32(w_row, u_data, params.exchange_dim);
                            int32_t shift = params.exchange_shift_ptr
                                ? static_cast<int32_t>(params.exchange_shift_ptr[r])
                                : 0;
                            v_data[r] = round_shift_right(acc, shift);
                        }
                    });
                } else {
                    for (int64_t r = 0; r < params.exchange_dim; ++r) {
                        const int8_t* w_row = params.exchange_ptr + r * params.exchange_dim;
                        int32_t acc = use_u16
                            ? dot_i8_i16(w_row, u16_data, params.exchange_dim)
                            : dot_i8_i32(w_row, u_data, params.exchange_dim);
                        int32_t shift = params.exchange_shift_ptr
                            ? static_cast<int32_t>(params.exchange_shift_ptr[r])
                            : 0;
                        v_data[r] = round_shift_right(acc, shift);
                    }
                }
                for (int64_t g = 0; g < params.groups; ++g) {
                    for (int64_t j = 0; j < params.per_group; ++j) {
                        int32_t inj = round_shift_right(v_data[g * params.per_group + j], params.inj_shift_i32);
                        const int32_t* idx = params.exchange_indices->ptr(phase, g, j);
                        s_row16[idx[0]] = clamp_int16(static_cast<int32_t>(s_row16[idx[0]]) + inj);
                        s_row16[idx[1]] = clamp_int16(static_cast<int32_t>(s_row16[idx[1]]) + inj);
                        s_row16[idx[2]] = clamp_int16(static_cast<int32_t>(s_row16[idx[2]]) - inj);
                        s_row16[idx[3]] = clamp_int16(static_cast<int32_t>(s_row16[idx[3]]) - inj);
                    }
                }
                if (params.use_second_activation) {
                    for (int64_t d = 0; d < params.d_state; ++d) {
                        int32_t x = static_cast<int32_t>(s_row16[d]) >> params.activation_shift;
                        x = clamp_int32(x, -128, 127) + 128;
                        int16_t tval = params.lut_ptr[x];
                        int32_t out = (static_cast<int32_t>(tval) * params.activation_T_q15) >> 15;
                        s_row16[d] = clamp_int16(out);
                    }
                }
            }

            if (params.do_pool) {
                float* __restrict__ temp_data = assume_aligned_if<float, 64>(temp.data());
                for (int64_t d = 0; d < params.d_state; ++d) {
                    temp_data[d] = static_cast<float>(s_row16[d]) * params.inv_q15;
                }
                if (params.normalize_output) {
                    float norm = dot_f32(temp_data, temp_data, params.d_state);
                    norm = std::sqrt(norm);
                    if (norm > 0.0f) {
                        float inv = 1.0f / norm;
                        for (int64_t d = 0; d < params.d_state; ++d) {
                            temp_data[d] *= inv;
                        }
                    }
                }
                const bool include_block = (params.pool_strategy == 1)
                    || (block_idx < params.block_lengths[b]);
                if (include_block) {
                    if (params.pool_strategy == 1) {
                        std::copy(temp_data, temp_data + params.d_state, out_row);
                    } else {
                        for (int64_t d = 0; d < params.d_state; ++d) {
                            out_row[d] += temp_data[d];
                        }
                    }
                }
            }

            copy_i16_to_i32(s_row16, s_row32, params.d_state);
        }

        if (params.do_pool && params.pool_strategy == 0) {
            float inv = 1.0f / static_cast<float>(params.block_lengths[b]);
            for (int64_t d = 0; d < params.d_state; ++d) {
                out_row[d] *= inv;
            }
        }
    }
}

static void run_quant_layer(
    const QuantLayerParams& params,
    bool use_parallel,
    int64_t batch_grain,
    AlignedVector<int32_t>* exchange_u_shared,
    AlignedVector<int32_t>* exchange_v_shared,
    AlignedVector<int16_t>* exchange_u16_shared
) {
    if (use_parallel) {
        at::parallel_for(0, params.batch, batch_grain, [&](int64_t b0, int64_t b1) {
            run_quant_layer_range(
                params,
                b0,
                b1,
                exchange_u_shared,
                exchange_v_shared,
                exchange_u16_shared
            );
        });
    } else {
        run_quant_layer_range(
            params,
            0,
            params.batch,
            exchange_u_shared,
            exchange_v_shared,
            exchange_u16_shared
        );
    }
}

torch::Tensor monoid_forward_quantized_impl(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    TORCH_CHECK(byte_tokens.device().is_cpu(), "byte_tokens must be on CPU");
    TORCH_CHECK(lengths.device().is_cpu(), "lengths must be on CPU");
    TORCH_CHECK(byte_tokens.scalar_type() == torch::kLong, "byte_tokens must be int64");
    TORCH_CHECK(lengths.scalar_type() == torch::kLong, "lengths must be int64");
    TORCH_CHECK(a_q15.scalar_type() == torch::kInt16, "a_q15 must be int16");
    TORCH_CHECK(b_int8.scalar_type() == torch::kInt8, "b_int8 must be int8");
    TORCH_CHECK(tanh_lut.scalar_type() == torch::kInt16, "tanh_lut must be int16");

    byte_tokens = byte_tokens.contiguous();
    lengths = lengths.contiguous();
    a_q15 = a_q15.contiguous();
    b_int8 = b_int8.contiguous();
    tanh_lut = tanh_lut.contiguous();
    exchange_weight = exchange_weight.contiguous();
    exchange_shift = exchange_shift.contiguous();

    const auto batch = byte_tokens.size(0);
    const auto seq_len = byte_tokens.size(1);
    const auto d_state = a_q15.size(1);

    TORCH_CHECK(d_state == n_tiles * tile_dim, "d_state must equal n_tiles * tile_dim");


    auto output = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int64_t> block_lengths(batch, 1);
    const int64_t* __restrict__ lengths_ptr = lengths.data_ptr<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
        int64_t bl = (lengths_ptr[b] + microblock_size - 1) / microblock_size;
        block_lengths[b] = std::max<int64_t>(1, bl);
    }
    const int64_t total_blocks = (seq_len + microblock_size - 1) / microblock_size;

    auto state32 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt32));
    auto state16 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt16));
    int32_t* __restrict__ state32_ptr = assume_aligned_if<int32_t, 64>(state32.data_ptr<int32_t>());
    int16_t* __restrict__ state16_ptr = assume_aligned_if<int16_t, 64>(state16.data_ptr<int16_t>());
    const int64_t* __restrict__ tokens_ptr = byte_tokens.data_ptr<int64_t>();
    const int16_t* __restrict__ a_ptr = assume_aligned_if<int16_t, 64>(a_q15.data_ptr<int16_t>());
    const int8_t* __restrict__ b_ptr = b_int8.data_ptr<int8_t>();
    const int16_t* __restrict__ lut_ptr = assume_aligned_if<int16_t, 64>(tanh_lut.data_ptr<int16_t>());
    const int8_t* __restrict__ exchange_ptr = exchange_weight.numel() > 0
        ? exchange_weight.data_ptr<int8_t>()
        : nullptr;
    const int8_t* __restrict__ exchange_shift_ptr = exchange_shift.numel() > 0
        ? exchange_shift.data_ptr<int8_t>()
        : nullptr;
    float* __restrict__ output_ptr = assume_aligned_if<float, 64>(output.data_ptr<float>());

    const int64_t vocab_stride = d_state;
    const int64_t state_stride = d_state;
    check_aligned_if_required(state32_ptr, kCacheLineBytes, "state32");
    check_aligned_if_required(state16_ptr, kCacheLineBytes, "state16");
    check_aligned_if_required(a_ptr, kCacheLineBytes, "a_q15");
    check_aligned_if_required(lut_ptr, kCacheLineBytes, "tanh_lut");
    check_aligned_if_required(output_ptr, kCacheLineBytes, "output");
    check_stride_alignment(state_stride, sizeof(int16_t), kCacheLineBytes, "state_stride");
    check_stride_alignment(vocab_stride, sizeof(int16_t), kCacheLineBytes, "vocab_stride");
    const int64_t exchange_dim = exchange_weight.numel() > 0 ? exchange_weight.size(0) : 0;
    if (exchange_weight.numel() > 0) {
        TORCH_CHECK(exchange_weight.dim() == 2, "exchange_weight must be (exchange_dim, exchange_dim)");
        TORCH_CHECK(exchange_weight.size(0) == exchange_weight.size(1), "exchange_weight must be square");
    }
    const bool exchange_configured = use_exchange && exchange_ptr;
    const bool exchange_active = exchange_configured && exchange_every > 0;
    const int64_t groups = exchange_configured ? n_tiles / 4 : 0;
    const int64_t per_group = (exchange_configured && groups > 0) ? exchange_dim / groups : 0;
    const int64_t exchange_dim_padded = exchange_configured ? cacheline_pad_elems<int32_t>(exchange_dim) : 0;
    if (exchange_configured) {
        TORCH_CHECK(n_tiles % 4 == 0, "n_tiles must be divisible by 4 for exchange.");
        TORCH_CHECK(exchange_dim % groups == 0, "exchange_dim must be divisible by n_tiles/4.");
    }

    const float inv_q15 = 1.0f / 32768.0f;
    const bool in_parallel = at::in_parallel_region();
    const bool use_parallel = !in_parallel && batch >= 2 && at::get_num_threads() > 1;
    const ExchangeIndexTable exchange_indices = exchange_active
        ? build_exchange_index_table(tile_dim, n_tiles, exchange_dim)
        : ExchangeIndexTable{};
    const bool exchange_parallel = (!use_parallel) && exchange_active && (at::get_num_threads() > 1) && !in_parallel;
    const int64_t exchange_grain = exchange_parallel ? parallel_grain_size(exchange_dim, 16, 1) : 0;
    const int32_t inj_shift_i32 = static_cast<int32_t>(inj_shift);
    const int64_t batch_grain = use_parallel ? parallel_grain_size(batch) : 0;
    AlignedVector<int32_t> exchange_u_shared;
    AlignedVector<int32_t> exchange_v_shared;
    if (exchange_active && exchange_parallel) {
        exchange_u_shared.assign(exchange_dim_padded, 0);
        exchange_v_shared.assign(exchange_dim_padded, 0);
    }
    AlignedVector<int16_t> exchange_u16_shared;
    if (exchange_active && exchange_parallel) {
        exchange_u16_shared.assign(exchange_dim_padded, 0);
    }

    QuantLayerParams params;
    params.batch = batch;
    params.seq_len = seq_len;
    params.d_state = d_state;
    params.microblock_size = microblock_size;
    params.n_tiles = n_tiles;
    params.tile_dim = tile_dim;
    params.activation_shift = activation_shift;
    params.activation_T_q15 = activation_T_q15;
    params.b_shift = b_shift;
    params.exchange_every = exchange_every;
    params.pool_strategy = pool_strategy;
    params.normalize_output = normalize_output;
    params.do_pool = true;
    params.total_blocks = total_blocks;
    params.vocab_stride = vocab_stride;
    params.state_stride = state_stride;
    params.exchange_dim = exchange_dim;
    params.groups = groups;
    params.per_group = per_group;
    params.exchange_dim_padded = exchange_dim_padded;
    params.exchange_grain = exchange_grain;
    params.inj_shift_i32 = inj_shift_i32;
    params.inv_q15 = inv_q15;
    params.exchange_active = exchange_active;
    params.exchange_parallel = exchange_parallel;
    params.use_second_activation = use_second_activation != 0;
    params.lengths_ptr = lengths_ptr;
    params.tokens_ptr = tokens_ptr;
    params.a_ptr = a_ptr;
    params.b_ptr = b_ptr;
    params.lut_ptr = lut_ptr;
    params.exchange_ptr = exchange_ptr;
    params.exchange_shift_ptr = exchange_shift_ptr;
    params.state32_ptr = state32_ptr;
    params.state16_ptr = state16_ptr;
    params.output_ptr = output_ptr;
    params.exchange_indices = &exchange_indices;
    params.block_lengths = block_lengths.data();

    run_quant_layer(
        params,
        use_parallel,
        batch_grain,
        (exchange_active && exchange_parallel) ? &exchange_u_shared : nullptr,
        (exchange_active && exchange_parallel) ? &exchange_v_shared : nullptr,
        (exchange_active && exchange_parallel) ? &exchange_u16_shared : nullptr
    );

    return output;
}

torch::Tensor monoid_forward_quantized_stacked_impl(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor proj_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    TORCH_CHECK(byte_tokens.device().is_cpu(), "byte_tokens must be on CPU");
    TORCH_CHECK(lengths.device().is_cpu(), "lengths must be on CPU");
    TORCH_CHECK(byte_tokens.scalar_type() == torch::kLong, "byte_tokens must be int64");
    TORCH_CHECK(lengths.scalar_type() == torch::kLong, "lengths must be int64");
    TORCH_CHECK(a_q15.scalar_type() == torch::kInt16, "a_q15 must be int16");
    TORCH_CHECK(b_int8.scalar_type() == torch::kInt8, "b_int8 must be int8");
    TORCH_CHECK(tanh_lut.scalar_type() == torch::kInt16, "tanh_lut must be int16");
    if (ln_weight.numel() > 0) {
        TORCH_CHECK(ln_weight.scalar_type() == torch::kFloat32, "ln_weight must be float32");
    }
    if (ln_bias.numel() > 0) {
        TORCH_CHECK(ln_bias.scalar_type() == torch::kFloat32, "ln_bias must be float32");
    }
    if (proj_weight.numel() > 0) {
        TORCH_CHECK(proj_weight.scalar_type() == torch::kFloat32, "proj_weight must be float32");
    }
    TORCH_CHECK(a_q15.dim() == 2 || a_q15.dim() == 3, "a_q15 must be 2D or 3D");
    TORCH_CHECK(b_int8.dim() == 2 || b_int8.dim() == 3, "b_int8 must be 2D or 3D");

    byte_tokens = byte_tokens.contiguous();
    lengths = lengths.contiguous();
    a_q15 = a_q15.contiguous();
    b_int8 = b_int8.contiguous();
    tanh_lut = tanh_lut.contiguous();
    exchange_weight = exchange_weight.contiguous();
    ln_weight = ln_weight.contiguous();
    ln_bias = ln_bias.contiguous();
    proj_weight = proj_weight.contiguous();
    exchange_shift = exchange_shift.contiguous();

    const auto batch = byte_tokens.size(0);
    const auto seq_len = byte_tokens.size(1);
    const int64_t n_layers = (a_q15.dim() == 3) ? a_q15.size(0) : 1;
    const int64_t vocab = (a_q15.dim() == 3) ? a_q15.size(1) : a_q15.size(0);
    const int64_t d_state = (a_q15.dim() == 3) ? a_q15.size(2) : a_q15.size(1);

    if (a_q15.dim() == 3) {
        TORCH_CHECK(b_int8.dim() == 3, "b_int8 must be 3D when a_q15 is 3D");
        TORCH_CHECK(b_int8.size(0) == n_layers, "b_int8 must match a_q15 layers");
        TORCH_CHECK(b_int8.size(1) == vocab && b_int8.size(2) == d_state, "b_int8 must match a_q15 shape");
    } else {
        TORCH_CHECK(b_int8.dim() == 2, "b_int8 must be 2D when a_q15 is 2D");
        TORCH_CHECK(b_int8.size(0) == vocab && b_int8.size(1) == d_state, "b_int8 must match a_q15 shape");
    }

    TORCH_CHECK(d_state == n_tiles * tile_dim, "d_state must equal n_tiles * tile_dim");

    const int64_t exchange_dim = exchange_weight.numel() > 0
        ? (exchange_weight.dim() == 3 ? exchange_weight.size(1) : exchange_weight.size(0))
        : 0;
    if (exchange_weight.numel() > 0) {
        TORCH_CHECK(exchange_weight.dim() == 2 || exchange_weight.dim() == 3,
            "exchange_weight must be 2D or 3D");
        if (exchange_weight.dim() == 3) {
            TORCH_CHECK(exchange_weight.size(0) == n_layers, "exchange_weight must match n_layers");
            TORCH_CHECK(exchange_weight.size(1) == exchange_weight.size(2), "exchange_weight must be square");
        } else {
            TORCH_CHECK(exchange_weight.size(0) == exchange_weight.size(1), "exchange_weight must be square");
        }
    }
    if (exchange_shift.numel() > 0) {
        TORCH_CHECK(exchange_shift.dim() == 1 || exchange_shift.dim() == 2,
            "exchange_shift must be 1D or 2D");
        if (exchange_shift.dim() == 2) {
            TORCH_CHECK(exchange_shift.size(0) == n_layers, "exchange_shift must match n_layers");
            TORCH_CHECK(exchange_shift.size(1) == exchange_dim, "exchange_shift dim mismatch");
        } else {
            TORCH_CHECK(exchange_shift.size(0) == exchange_dim, "exchange_shift dim mismatch");
        }
    }
    if (ln_weight.numel() > 0) {
        TORCH_CHECK(ln_weight.dim() == 2, "ln_weight must be (n_layers, d_state)");
        TORCH_CHECK(ln_weight.size(0) == n_layers && ln_weight.size(1) == d_state,
            "ln_weight shape mismatch");
    }
    if (ln_bias.numel() > 0) {
        TORCH_CHECK(ln_bias.dim() == 2, "ln_bias must be (n_layers, d_state)");
        TORCH_CHECK(ln_bias.size(0) == n_layers && ln_bias.size(1) == d_state,
            "ln_bias shape mismatch");
    }
    if (proj_weight.numel() > 0) {
        TORCH_CHECK(proj_weight.dim() == 2, "proj_weight must be (out_dim, d_state)");
        TORCH_CHECK(proj_weight.size(1) == d_state, "proj_weight input dim mismatch");
    }

    auto output = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kFloat32));
    std::vector<int64_t> block_lengths(batch, 1);
    const int64_t* __restrict__ lengths_ptr = lengths.data_ptr<int64_t>();
    for (int64_t b = 0; b < batch; ++b) {
        int64_t bl = (lengths_ptr[b] + microblock_size - 1) / microblock_size;
        block_lengths[b] = std::max<int64_t>(1, bl);
    }
    const int64_t total_blocks = (seq_len + microblock_size - 1) / microblock_size;

    auto state_res16 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt16));
    auto state32 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt32));
    auto state16 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt16));
    int16_t* __restrict__ res16_ptr = assume_aligned_if<int16_t, 64>(state_res16.data_ptr<int16_t>());
    int32_t* __restrict__ state32_ptr = assume_aligned_if<int32_t, 64>(state32.data_ptr<int32_t>());
    int16_t* __restrict__ state16_ptr = assume_aligned_if<int16_t, 64>(state16.data_ptr<int16_t>());
    const int64_t* __restrict__ tokens_ptr = byte_tokens.data_ptr<int64_t>();
    const int16_t* __restrict__ a_ptr = assume_aligned_if<int16_t, 64>(a_q15.data_ptr<int16_t>());
    const int8_t* __restrict__ b_ptr = b_int8.data_ptr<int8_t>();
    const int16_t* __restrict__ lut_ptr = assume_aligned_if<int16_t, 64>(tanh_lut.data_ptr<int16_t>());
    const int8_t* __restrict__ exchange_ptr = exchange_weight.numel() > 0
        ? exchange_weight.data_ptr<int8_t>()
        : nullptr;
    const float* __restrict__ ln_weight_ptr = ln_weight.numel() > 0
        ? assume_aligned_if<float, 64>(ln_weight.data_ptr<float>())
        : nullptr;
    const float* __restrict__ ln_bias_ptr = ln_bias.numel() > 0
        ? assume_aligned_if<float, 64>(ln_bias.data_ptr<float>())
        : nullptr;
    const int8_t* __restrict__ exchange_shift_ptr = exchange_shift.numel() > 0
        ? exchange_shift.data_ptr<int8_t>()
        : nullptr;
    const float* __restrict__ proj_ptr = proj_weight.numel() > 0
        ? assume_aligned_if<float, 64>(proj_weight.data_ptr<float>())
        : nullptr;
    float* __restrict__ output_ptr = assume_aligned_if<float, 64>(output.data_ptr<float>());

    const int64_t vocab_stride = d_state;
    const int64_t state_stride = d_state;
    const int64_t layer_stride = vocab * d_state;
    const int64_t exchange_stride = exchange_weight.numel() > 0
        ? (exchange_weight.dim() == 3 ? exchange_weight.size(1) * exchange_weight.size(2)
                                      : exchange_dim * exchange_dim)
        : 0;
    const int64_t exchange_shift_stride = exchange_shift.numel() > 0
        ? (exchange_shift.dim() == 2 ? exchange_shift.size(1) : 0)
        : 0;
    check_aligned_if_required(res16_ptr, kCacheLineBytes, "state_res16");
    check_aligned_if_required(state32_ptr, kCacheLineBytes, "state32");
    check_aligned_if_required(state16_ptr, kCacheLineBytes, "state16");
    check_aligned_if_required(a_ptr, kCacheLineBytes, "a_q15");
    check_aligned_if_required(lut_ptr, kCacheLineBytes, "tanh_lut");
    check_aligned_if_required(output_ptr, kCacheLineBytes, "output");
    if (ln_weight_ptr) {
        check_aligned_if_required(ln_weight_ptr, kCacheLineBytes, "ln_weight");
    }
    if (ln_bias_ptr) {
        check_aligned_if_required(ln_bias_ptr, kCacheLineBytes, "ln_bias");
    }
    if (proj_ptr) {
        check_aligned_if_required(proj_ptr, kCacheLineBytes, "proj_weight");
    }
    check_stride_alignment(state_stride, sizeof(int16_t), kCacheLineBytes, "state_stride");
    check_stride_alignment(vocab_stride, sizeof(int16_t), kCacheLineBytes, "vocab_stride");
    check_stride_alignment(layer_stride, sizeof(int16_t), kCacheLineBytes, "layer_stride");
    const bool exchange_configured = use_exchange && exchange_ptr;
    const bool exchange_active = exchange_configured && exchange_every > 0;
    const int64_t groups = exchange_configured ? n_tiles / 4 : 0;
    const int64_t per_group = (exchange_configured && groups > 0) ? exchange_dim / groups : 0;
    const int64_t exchange_dim_padded = exchange_configured ? cacheline_pad_elems<int32_t>(exchange_dim) : 0;
    if (exchange_configured) {
        TORCH_CHECK(n_tiles % 4 == 0, "n_tiles must be divisible by 4 for exchange.");
        TORCH_CHECK(exchange_dim % groups == 0, "exchange_dim must be divisible by n_tiles/4.");
    }

    const float inv_q15 = 1.0f / 32768.0f;
    const bool in_parallel = at::in_parallel_region();
    const bool use_parallel = !in_parallel && batch >= 2 && at::get_num_threads() > 1;
    const bool use_ln = n_layers > 1;
    const ExchangeIndexTable exchange_indices = exchange_active
        ? build_exchange_index_table(tile_dim, n_tiles, exchange_dim)
        : ExchangeIndexTable{};
    const bool exchange_parallel = (!use_parallel) && exchange_active && (at::get_num_threads() > 1) && !in_parallel;
    const int64_t exchange_grain = exchange_parallel ? parallel_grain_size(exchange_dim, 16, 1) : 0;
    const int32_t inj_shift_i32 = static_cast<int32_t>(inj_shift);
    const int64_t batch_grain = use_parallel ? parallel_grain_size(batch) : 0;
    AlignedVector<int32_t> exchange_u_shared;
    AlignedVector<int32_t> exchange_v_shared;
    if (exchange_active && exchange_parallel) {
        exchange_u_shared.assign(exchange_dim_padded, 0);
        exchange_v_shared.assign(exchange_dim_padded, 0);
    }
    AlignedVector<int16_t> exchange_u16_shared;
    if (exchange_active && exchange_parallel) {
        exchange_u16_shared.assign(exchange_dim_padded, 0);
    }

    auto accumulate_residual = [&](int64_t b0, int64_t b1) {
        for (int64_t b = b0; b < b1; ++b) {
            int16_t* __restrict__ res_row = assume_aligned_if<int16_t, 64>(res16_ptr + b * state_stride);
            const int16_t* __restrict__ layer_row = assume_aligned_if<int16_t, 64>(state16_ptr + b * state_stride);
            add_int16_saturate(res_row, layer_row, d_state);
        }
    };

    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const float* __restrict__ ln_w = ln_weight_ptr ? ln_weight_ptr + layer * d_state : nullptr;
        const float* __restrict__ ln_b = ln_bias_ptr ? ln_bias_ptr + layer * d_state : nullptr;

        auto init_layer_state = [&](int64_t b0, int64_t b1) {
            thread_local AlignedVector<float> ln_in;
            thread_local AlignedVector<float> ln_out;
            float* __restrict__ ln_in_ptr = nullptr;
            float* __restrict__ ln_out_ptr = nullptr;
            if (use_ln) {
                const int64_t ln_padded = cacheline_pad_elems<float>(d_state);
                if (ln_in.size() != static_cast<size_t>(ln_padded)) {
                    ln_in.assign(ln_padded, 0.0f);
                    ln_out.assign(ln_padded, 0.0f);
                }
                ln_in_ptr = ln_in.data();
                ln_out_ptr = ln_out.data();
            }
            for (int64_t b = b0; b < b1; ++b) {
                const int16_t* __restrict__ res_row = assume_aligned_if<int16_t, 64>(
                    res16_ptr + b * state_stride
                );
                int16_t* __restrict__ s_row16 = assume_aligned_if<int16_t, 64>(
                    state16_ptr + b * state_stride
                );
                int32_t* __restrict__ s_row32 = assume_aligned_if<int32_t, 64>(
                    state32_ptr + b * state_stride
                );
                if (use_ln) {
                    for (int64_t d = 0; d < d_state; ++d) {
                        ln_in_ptr[d] = static_cast<float>(res_row[d]) * inv_q15;
                    }
                    layer_norm_row(ln_in_ptr, ln_out_ptr, ln_w, ln_b, d_state);
                    for (int64_t d = 0; d < d_state; ++d) {
                        int32_t q = static_cast<int32_t>(std::lrintf(ln_out_ptr[d] * 32768.0f));
                        int16_t q16 = clamp_int16(q);
                        s_row16[d] = q16;
                        s_row32[d] = static_cast<int32_t>(q16);
                    }
                } else {
                    copy_i16_to_i16_i32(res_row, s_row16, s_row32, d_state);
                }
            }
        };

        const int16_t* __restrict__ a_layer = assume_aligned_if<int16_t, 64>(a_ptr + layer * layer_stride);
        const int8_t* __restrict__ b_layer = b_ptr + layer * layer_stride;
        const int8_t* __restrict__ exchange_layer = exchange_ptr
            ? exchange_ptr + (exchange_weight.dim() == 3 ? layer * exchange_stride : 0)
            : nullptr;
        const int8_t* __restrict__ exchange_shift_layer = exchange_shift_ptr
            ? exchange_shift_ptr + (exchange_shift.dim() == 2 ? layer * exchange_shift_stride : 0)
            : nullptr;

        QuantLayerParams params;
        params.batch = batch;
        params.seq_len = seq_len;
        params.d_state = d_state;
        params.microblock_size = microblock_size;
        params.n_tiles = n_tiles;
        params.tile_dim = tile_dim;
        params.activation_shift = activation_shift;
        params.activation_T_q15 = activation_T_q15;
        params.b_shift = b_shift;
        params.exchange_every = exchange_every;
        params.pool_strategy = pool_strategy;
        params.normalize_output = normalize_output;
        params.do_pool = (layer == (n_layers - 1));
        params.total_blocks = total_blocks;
        params.vocab_stride = vocab_stride;
        params.state_stride = state_stride;
        params.exchange_dim = exchange_dim;
        params.groups = groups;
        params.per_group = per_group;
        params.exchange_dim_padded = exchange_dim_padded;
        params.exchange_grain = exchange_grain;
        params.inj_shift_i32 = inj_shift_i32;
        params.inv_q15 = inv_q15;
        params.exchange_active = exchange_active && exchange_layer;
        params.exchange_parallel = exchange_parallel && exchange_layer;
        params.use_second_activation = use_second_activation != 0;
        params.lengths_ptr = lengths_ptr;
        params.tokens_ptr = tokens_ptr;
        params.a_ptr = a_layer;
        params.b_ptr = b_layer;
        params.lut_ptr = lut_ptr;
        params.exchange_ptr = exchange_layer;
        params.exchange_shift_ptr = exchange_shift_layer;
        params.state32_ptr = state32_ptr;
        params.state16_ptr = state16_ptr;
        params.output_ptr = output_ptr;
        params.exchange_indices = &exchange_indices;
        params.block_lengths = block_lengths.data();
        auto process_layer = [&](int64_t b0, int64_t b1) {
            init_layer_state(b0, b1);
            run_quant_layer_range(
                params,
                b0,
                b1,
                (params.exchange_active && params.exchange_parallel) ? &exchange_u_shared : nullptr,
                (params.exchange_active && params.exchange_parallel) ? &exchange_v_shared : nullptr,
                (params.exchange_active && params.exchange_parallel) ? &exchange_u16_shared : nullptr
            );
            accumulate_residual(b0, b1);
        };

        if (use_parallel) {
            at::parallel_for(0, batch, batch_grain, process_layer);
        } else {
            process_layer(0, batch);
        }
    }

    if (proj_ptr) {
        const int64_t out_dim = proj_weight.size(0);
        auto proj_out = torch::zeros({batch, out_dim}, torch::TensorOptions().dtype(torch::kFloat32));
        float* __restrict__ proj_out_ptr = assume_aligned_if<float, 64>(proj_out.data_ptr<float>());
        for (int64_t b = 0; b < batch; ++b) {
            const float* __restrict__ in_row = output_ptr + b * d_state;
            float* __restrict__ out_row = proj_out_ptr + b * out_dim;
            for (int64_t o = 0; o < out_dim; ++o) {
                const float* __restrict__ w_row = proj_ptr + o * d_state;
                out_row[o] = dot_f32(w_row, in_row, d_state);
            }
            if (normalize_output) {
                normalize_l2(out_row, out_dim);
            }
        }
        return proj_out;
    }

    return output;
}

torch::Tensor monoid_forward_quantized(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_quantized_impl(
        byte_tokens,
        lengths,
        a_q15,
        b_int8,
        tanh_lut,
        exchange_weight,
        exchange_shift,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_shift,
        activation_T_q15,
        b_shift,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );
}

torch::Tensor monoid_forward_quantized_stacked(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor proj_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_quantized_stacked_impl(
        byte_tokens,
        lengths,
        a_q15,
        b_int8,
        tanh_lut,
        exchange_weight,
        ln_weight,
        ln_bias,
        proj_weight,
        exchange_shift,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_shift,
        activation_T_q15,
        b_shift,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );
}

std::vector<torch::Tensor> monoid_forward_quantized_int8(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    auto output = monoid_forward_quantized_impl(
        byte_tokens,
        lengths,
        a_q15,
        b_int8,
        tanh_lut,
        exchange_weight,
        exchange_shift,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_shift,
        activation_T_q15,
        b_shift,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );

    const auto batch = output.size(0);
    const auto d_state = output.size(1);
    auto out_int8 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt8));
    auto scale_q15 = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt16));
    auto out_ptr = output.data_ptr<float>();
    auto int8_ptr = out_int8.data_ptr<int8_t>();
    auto scale_ptr = scale_q15.data_ptr<int16_t>();

    for (int64_t b = 0; b < batch; ++b) {
        const float* row = out_ptr + b * d_state;
        float norm = dot_f32(row, row, d_state);
        norm = std::sqrt(std::max(norm, 1e-12f));
        float scale = 127.0f / norm;
        if (scale > 1.0f) {
            scale = 1.0f;
        }
        int32_t q15 = static_cast<int32_t>(std::lrint(scale * 32768.0f));
        if (q15 > 32767) q15 = 32767;
        if (q15 < 0) q15 = 0;
        scale_ptr[b] = static_cast<int16_t>(q15);
        int8_t* out_row = int8_ptr + b * d_state;
        for (int64_t d = 0; d < d_state; ++d) {
            int32_t val = static_cast<int32_t>(std::lrint(row[d] * scale));
            if (val > 127) val = 127;
            if (val < -127) val = -127;
            out_row[d] = static_cast<int8_t>(val);
        }
    }

    return {output, out_int8, scale_q15};
}

std::vector<torch::Tensor> monoid_forward_quantized_stacked_int8(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor ln_weight,
    torch::Tensor ln_bias,
    torch::Tensor proj_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    auto output = monoid_forward_quantized_stacked_impl(
        byte_tokens,
        lengths,
        a_q15,
        b_int8,
        tanh_lut,
        exchange_weight,
        ln_weight,
        ln_bias,
        proj_weight,
        exchange_shift,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_shift,
        activation_T_q15,
        b_shift,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );

    const auto batch = output.size(0);
    const auto d_state = output.size(1);
    auto out_int8 = torch::zeros({batch, d_state}, torch::TensorOptions().dtype(torch::kInt8));
    auto scale_q15 = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt16));
    auto out_ptr = output.data_ptr<float>();
    auto int8_ptr = out_int8.data_ptr<int8_t>();
    auto scale_ptr = scale_q15.data_ptr<int16_t>();

    for (int64_t b = 0; b < batch; ++b) {
        const float* row = out_ptr + b * d_state;
        float norm = dot_f32(row, row, d_state);
        norm = std::sqrt(std::max(norm, 1e-12f));
        float scale = 127.0f / norm;
        if (scale > 1.0f) {
            scale = 1.0f;
        }
        int32_t q15 = static_cast<int32_t>(std::lrint(scale * 32768.0f));
        if (q15 > 32767) q15 = 32767;
        if (q15 < 0) q15 = 0;
        scale_ptr[b] = static_cast<int16_t>(q15);
        int8_t* out_row = int8_ptr + b * d_state;
        for (int64_t d = 0; d < d_state; ++d) {
            int32_t val = static_cast<int32_t>(std::lrint(row[d] * scale));
            if (val > 127) val = 127;
            if (val < -127) val = -127;
            out_row[d] = static_cast<int8_t>(val);
        }
    }

    return {output, out_int8, scale_q15};
}

torch::Tensor monoid_forward_float_ref(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_float(
        byte_tokens,
        lengths,
        a_f32,
        b_f32,
        exchange_weight,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_T,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );
}

torch::Tensor monoid_forward_float_stacked_ref(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_f32,
    torch::Tensor b_f32,
    torch::Tensor exchange_weight,
    torch::Tensor norm_weight,
    torch::Tensor norm_bias,
    torch::Tensor proj_weight,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    double activation_T,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_float_stacked(
        byte_tokens,
        lengths,
        a_f32,
        b_f32,
        exchange_weight,
        norm_weight,
        norm_bias,
        proj_weight,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_T,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );
}

torch::Tensor monoid_forward_quantized_ref(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_quantized(
        byte_tokens,
        lengths,
        a_q15,
        b_int8,
        tanh_lut,
        exchange_weight,
        exchange_shift,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_shift,
        activation_T_q15,
        b_shift,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );
}

std::vector<torch::Tensor> monoid_forward_quantized_int8_ref(
    torch::Tensor byte_tokens,
    torch::Tensor lengths,
    torch::Tensor a_q15,
    torch::Tensor b_int8,
    torch::Tensor tanh_lut,
    torch::Tensor exchange_weight,
    torch::Tensor exchange_shift,
    int64_t microblock_size,
    int64_t n_tiles,
    int64_t tile_dim,
    int64_t activation_shift,
    int64_t activation_T_q15,
    int64_t b_shift,
    int64_t exchange_every,
    int64_t inj_shift,
    int64_t use_exchange,
    int64_t use_second_activation,
    int64_t pool_strategy,
    int64_t normalize_output
) {
    return monoid_forward_quantized_int8(
        byte_tokens,
        lengths,
        a_q15,
        b_int8,
        tanh_lut,
        exchange_weight,
        exchange_shift,
        microblock_size,
        n_tiles,
        tile_dim,
        activation_shift,
        activation_T_q15,
        b_shift,
        exchange_every,
        inj_shift,
        use_exchange,
        use_second_activation,
        pool_strategy,
        normalize_output
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("monoid_forward_float", &monoid_forward_float, "Monoid full precision forward");
    m.def("monoid_forward_float_stacked", &monoid_forward_float_stacked, "Monoid full precision stacked forward");
    m.def("monoid_forward_float_profile", &monoid_forward_float_profile, "Monoid full precision forward (profile)");
    m.def("monoid_forward_float_stacked_profile", &monoid_forward_float_stacked_profile, "Monoid full precision stacked forward (profile)");
    m.def("monoid_forward_quantized", &monoid_forward_quantized, "Monoid quantized forward");
    m.def("monoid_forward_quantized_stacked", &monoid_forward_quantized_stacked, "Monoid quantized stacked forward");
    m.def("monoid_forward_quantized_int8", &monoid_forward_quantized_int8, "Monoid quantized forward with int8 output");
    m.def("monoid_forward_quantized_stacked_int8", &monoid_forward_quantized_stacked_int8, "Monoid quantized stacked forward with int8 output");
    m.def("monoid_forward_float_ref", &monoid_forward_float_ref, "Monoid full precision forward (ref)");
    m.def("monoid_forward_float_stacked_ref", &monoid_forward_float_stacked_ref, "Monoid full precision stacked forward (ref)");
    m.def("monoid_forward_quantized_ref", &monoid_forward_quantized_ref, "Monoid quantized forward (ref)");
    m.def("monoid_forward_quantized_int8_ref", &monoid_forward_quantized_int8_ref, "Monoid quantized forward int8 (ref)");
}
