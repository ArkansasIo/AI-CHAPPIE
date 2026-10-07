// ---------------------------------------------------------------------------
// core/math.hpp -- linear algebra primitives used by every subsystem.
//
// Everything is header-declared here and implemented in src/core/math.cpp.
// The design favours clarity over raw speed: these are the same operations the
// Python library exposes (docs/02_algebra.md), ported to C++17.
// ---------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <functional>
#include <ostream>
#include <random>
#include <string>
#include <vector>

namespace chappie
{

    // ---------------------------------------------------------------------------
    // Vec -- a dense 1-D vector of doubles.
    // ---------------------------------------------------------------------------
    class Vec
    {
    public:
        Vec() = default;
        explicit Vec(std::size_t n, double fill = 0.0);
        Vec(std::initializer_list<double> values);
        explicit Vec(const std::vector<double> &values);

        std::size_t size() const { return data_.size(); }
        bool empty() const { return data_.empty(); }

        double &operator[](std::size_t i) { return data_[i]; }
        double operator[](std::size_t i) const { return data_[i]; }

        // element-wise arithmetic
        Vec operator+(const Vec &o) const;
        Vec operator-(const Vec &o) const;
        Vec operator*(double s) const;
        Vec operator/(double s) const;
        Vec &operator+=(const Vec &o);
        Vec &operator-=(const Vec &o);
        Vec &operator*=(double s);

        // reductions
        double dot(const Vec &o) const;
        double sum() const;
        double mean() const;
        double max() const;
        double min() const;
        double norm() const;    // L2
        double norm_l1() const; // L1

        // element-wise helpers
        Vec hadamard(const Vec &o) const; // element-wise product
        Vec abs() const;
        Vec clip(double lo, double hi) const;

        // statistics
        double variance() const;

        const std::vector<double> &raw() const { return data_; }
        std::vector<double> &raw() { return data_; }

        static Vec zeros(std::size_t n);
        static Vec ones(std::size_t n);
        static Vec random(std::size_t n, double scale = 1.0,
                          std::mt19937 *rng = nullptr);

        std::string to_string(int precision = 4) const;

    private:
        std::vector<double> data_;
    };

    // ---------------------------------------------------------------------------
    // Mat -- a dense 2-D matrix, row-major.
    // ---------------------------------------------------------------------------
    class Mat
    {
    public:
        Mat() = default;
        Mat(std::size_t rows, std::size_t cols, double fill = 0.0);

        std::size_t rows() const { return rows_; }
        std::size_t cols() const { return cols_; }

        double &at(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
        double at(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

        // row / column access as vectors
        Vec row(std::size_t r) const;
        Vec col(std::size_t c) const;
        void set_row(std::size_t r, const Vec &v);
        void set_col(std::size_t c, const Vec &v);

        // core operations
        Mat transpose() const;
        Vec operator*(const Vec &v) const; // matrix-vector
        Mat operator*(const Mat &o) const; // matrix-matrix
        Mat operator+(const Mat &o) const;
        Mat operator-(const Mat &o) const;
        Mat operator*(double s) const;

        // outer product: u v^T
        static Mat outer(const Vec &u, const Vec &v);

        // applying a per-column bias to every row
        Mat add_row_vector(const Vec &bias) const;

        double sum() const;
        double frobenius() const;

        static Mat random(std::size_t rows, std::size_t cols,
                          double scale = 0.1, std::mt19937 *rng = nullptr);
        static Mat identity(std::size_t n);

        std::string to_string(int precision = 4) const;

    private:
        std::size_t rows_ = 0;
        std::size_t cols_ = 0;
        std::vector<double> data_;
    };

    // ---------------------------------------------------------------------------
    // Activation functions (value + derivative), matching nn/activations.py.
    // ---------------------------------------------------------------------------
    namespace activation
    {
        double sigmoid(double z);
        double sigmoid_prime_from_output(double a);
        double tanh_f(double z);
        double tanh_prime_from_output(double a);
        double relu(double z);
        double relu_prime(double z);
        double leaky_relu(double z, double alpha = 0.01);
        double leaky_relu_prime(double z, double alpha = 0.01);
        double linear(double z);
        double linear_prime(double z);
    } // namespace activation

    // Softmax over a vector (numerically stable).
    Vec softmax(const Vec &z);

    // Cross-entropy loss and its gradient w.r.t. logits: (p - y).
    double cross_entropy(const Vec &logits, const Vec &target);
    Vec cross_entropy_grad(const Vec &logits, const Vec &target);

    // Mean squared error between two vectors.
    double mse(const Vec &a, const Vec &b);
    Vec mse_grad(const Vec &a, const Vec &b);

    inline double clamp(double v, double lo, double hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    inline double lerp(double a, double b, double t) { return a + (b - a) * t; }

    std::ostream &operator<<(std::ostream &os, const Vec &v);
    std::ostream &operator<<(std::ostream &os, const Mat &m);

} // namespace chappie
