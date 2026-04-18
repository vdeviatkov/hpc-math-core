#pragma once

/**
 * @file matrix.hpp
 * @brief Row-major, 64-byte aligned Matrix<T> class template.
 *
 * Design rationale
 * ================
 * Modern CPUs fetch memory in cache lines of exactly 64 bytes. A 64-byte
 * aligned base address guarantees that the first element of every row lands at
 * a cache-line boundary, eliminating "split loads" where a single scalar value
 * spans two cache lines and halves effective memory bandwidth.
 *
 * AVX-512 SIMD registers are 512 bits = 64 bytes wide. Using vmovaps
 * (aligned store) vs vmovups (unaligned store) on Skylake-SP / Ice Lake costs
 * an identical number of cycles *when the data is actually aligned*, but
 * aligned loads enable the compiler and micro-architecture to make stronger
 * assumptions that unlock additional optimisations (e.g. loop vectorisation
 * without a peel prologue, better prefetch distance calculations).
 *
 * Memory layout
 * =============
 * Elements are stored in row-major (C) order:
 *
 *   element (i, j)  →  data_[i * cols_ + j]
 *
 * This means consecutive elements within a row are adjacent in memory. The
 * innermost loop should therefore iterate over columns (j) to walk through
 * memory sequentially and maximise spatial locality.
 */

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace hpc {

/// Alignment boundary in bytes (matches a single AVX-512 register and one
/// cache line).
inline constexpr std::size_t kCacheLineBytes = 64;

// ---------------------------------------------------------------------------
// Custom deleter for aligned memory
// ---------------------------------------------------------------------------

/**
 * @brief Deleter for memory allocated with std::aligned_alloc.
 *
 * std::unique_ptr requires a deleter that calls std::free (not delete[]) when
 * the storage was obtained from aligned_alloc / posix_memalign.
 */
struct AlignedDeleter {
    void operator()(void* ptr) const noexcept {
#if defined(_MSC_VER)
        ::_aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};

// ---------------------------------------------------------------------------
// Matrix<T>
// ---------------------------------------------------------------------------

/**
 * @brief A heap-allocated, row-major, cache-line-aligned 2-D matrix.
 *
 * @tparam T  Element type. Must be a trivially-copyable scalar (float, double,
 *            int32_t, …) so that we can use std::memcpy for copies and avoid
 *            constructor/destructor overhead in tight loops.
 *
 * Thread safety
 * -------------
 * No synchronisation is provided. Multiple concurrent readers are safe;
 * concurrent writes or mixed read/write access require external locking.
 */
template <typename T>
class Matrix {
    static_assert(std::is_trivially_copyable_v<T>,
                  "Matrix<T> requires a trivially-copyable element type "
                  "(e.g. float, double, int32_t).");

  public:
    // ------------------------------------------------------------------
    // Constructors / destructor
    // ------------------------------------------------------------------

    /// Construct a zero-initialised rows×cols matrix.
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(allocate(rows * cols)) {
        std::memset(data_.get(), 0, rows_ * cols_ * sizeof(T));
    }

    /// Copy constructor — performs a deep element-wise copy.
    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), data_(allocate(other.rows_ * other.cols_)) {
        std::memcpy(data_.get(), other.data_.get(), rows_ * cols_ * sizeof(T));
    }

    /// Copy-assignment operator.
    Matrix& operator=(const Matrix& other) {
        if (this == &other)
            return *this;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = allocate(rows_ * cols_);
        std::memcpy(data_.get(), other.data_.get(), rows_ * cols_ * sizeof(T));
        return *this;
    }

    /// Move constructor — transfers ownership without allocation.
    Matrix(Matrix&&) noexcept = default;

    /// Move-assignment operator.
    Matrix& operator=(Matrix&&) noexcept = default;

    ~Matrix() = default;

    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    /**
     * @brief Mutable element access using row-major index arithmetic.
     *
     * element (i, j) lives at data_[i * cols_ + j].
     * No bounds checking in release builds; use the checked variant for
     * debugging.
     */
    [[nodiscard]] T& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }

    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    /// Bounds-checked element access (throws std::out_of_range).
    [[nodiscard]] T& at(std::size_t i, std::size_t j) {
        check_bounds(i, j);
        return data_[i * cols_ + j];
    }

    [[nodiscard]] const T& at(std::size_t i, std::size_t j) const {
        check_bounds(i, j);
        return data_[i * cols_ + j];
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return rows_ * cols_; }

    /**
     * @brief Raw pointer to the first element.
     *
     * Guaranteed to be aligned to kCacheLineBytes (64) bytes. Safe to pass
     * directly to SIMD intrinsics that require aligned loads (e.g.
     * _mm512_load_pd).
     */
    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }

    // ------------------------------------------------------------------
    // Utility
    // ------------------------------------------------------------------

    /// Fill every element with a constant value.
    void fill(T value) noexcept {
        T* p                = data_.get();
        const std::size_t n = rows_ * cols_;
        for (std::size_t i = 0; i < n; ++i)
            p[i] = value;
    }

    /// Zero all elements (equivalent to fill(T{})).
    void zero() noexcept { std::memset(data_.get(), 0, rows_ * cols_ * sizeof(T)); }

  private:
    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Allocate an aligned buffer for `n` elements of type T.
    static std::unique_ptr<T[], AlignedDeleter> allocate(std::size_t n) {
        if (n == 0)
            return {nullptr, AlignedDeleter{}};

        // Total bytes must be a multiple of the alignment requirement.
        std::size_t bytes = n * sizeof(T);
        // Round up to the nearest multiple of kCacheLineBytes.
        bytes = (bytes + kCacheLineBytes - 1) & ~(kCacheLineBytes - 1);

#if defined(_MSC_VER)
        void* raw = ::_aligned_malloc(bytes, kCacheLineBytes);
#else
        void* raw = std::aligned_alloc(kCacheLineBytes, bytes);
#endif
        if (!raw)
            throw std::bad_alloc{};

        return {static_cast<T*>(raw), AlignedDeleter{}};
    }

    void check_bounds(std::size_t i, std::size_t j) const {
        if (i >= rows_ || j >= cols_)
            throw std::out_of_range("Matrix index out of range");
    }

    // ------------------------------------------------------------------
    // Data members
    // ------------------------------------------------------------------

    std::size_t rows_{0};
    std::size_t cols_{0};

    /// Contiguous, 64-byte-aligned storage in row-major order.
    std::unique_ptr<T[], AlignedDeleter> data_;
};

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

using MatrixF = Matrix<float>;   ///< Single-precision matrix
using MatrixD = Matrix<double>;  ///< Double-precision matrix

}  // namespace hpc
