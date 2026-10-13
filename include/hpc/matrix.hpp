#pragma once

/**
 * @file matrix.hpp
 * @brief Matrix<T>: a dense, row-major matrix on a 64-byte aligned buffer.
 *
 * Element (i, j) is at data()[i * cols() + j], so a loop over j walks memory
 * with stride 1. The buffer starts on a 64-byte boundary (one x86 cache line,
 * one AVX-512 register); row i is aligned too only when cols() * sizeof(T)
 * is a multiple of 64.
 */

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

namespace hpc {

/// Alignment of every Matrix buffer, in bytes.
inline constexpr std::size_t kMatrixAlignment = 64;

template <typename T>
class Matrix {
    static_assert(std::is_trivially_copyable_v<T>, "Matrix<T> copies elements with memcpy");

  public:
    using value_type = T;

    /// A rows×cols matrix of zeros.
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(allocate(rows * cols)) {
        zero();
    }

    Matrix(const Matrix& other) : Matrix(other.rows_, other.cols_) {
        std::memcpy(data(), other.data(), size() * sizeof(T));
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other)
            *this = Matrix(other);
        return *this;
    }

    Matrix(Matrix&&) noexcept            = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix()                            = default;

    /// Unchecked element access.
    [[nodiscard]] T& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    [[nodiscard]] const T& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return rows_ * cols_; }

    [[nodiscard]] T* data() noexcept { return data_.get(); }
    [[nodiscard]] const T* data() const noexcept { return data_.get(); }

    void fill(T value) noexcept {
        for (std::size_t i = 0; i < size(); ++i)
            data_[i] = value;
    }

    void zero() noexcept {
        if (size() != 0)
            std::memset(data(), 0, size() * sizeof(T));
    }

  private:
    struct Free {
        void operator()(T* p) const noexcept {
#if defined(_MSC_VER)
            ::_aligned_free(p);
#else
            std::free(p);
#endif
        }
    };

    static std::unique_ptr<T[], Free> allocate(std::size_t n) {
        if (n == 0)
            return nullptr;
        // aligned_alloc requires the size to be a multiple of the alignment.
        const std::size_t bytes =
            (n * sizeof(T) + kMatrixAlignment - 1) / kMatrixAlignment * kMatrixAlignment;
#if defined(_MSC_VER)
        void* p = ::_aligned_malloc(bytes, kMatrixAlignment);
#else
        void* p = std::aligned_alloc(kMatrixAlignment, bytes);
#endif
        if (p == nullptr)
            throw std::bad_alloc{};
        return std::unique_ptr<T[], Free>(static_cast<T*>(p));
    }

    std::size_t rows_;
    std::size_t cols_;
    std::unique_ptr<T[], Free> data_;
};

}  // namespace hpc
