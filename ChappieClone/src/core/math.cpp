// ---------------------------------------------------------------------------
// core/math.cpp -- implementation of Vec, Mat, activations, losses.
// ---------------------------------------------------------------------------
#include "core/math.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace chappie
{

    // ===========================================================================
    // Vec
    // ===========================================================================
    Vec::Vec(std::size_t n, double fill) : data_(n, fill) {}

    Vec::Vec(std::initializer_list<double> values) : data_(values) {}

    Vec::Vec(const std::vector<double> &values) : data_(values) {}

    Vec Vec::operator+(const Vec &o) const
    {
        if (size() != o.size())
            throw std::invalid_argument("Vec::+ size mismatch");
        Vec r(size());
        for (std::size_t i = 0; i < size(); ++i)
            r[i] = data_[i] + o[i];
        return r;
    }

    Vec Vec::operator-(const Vec &o) const
    {
        if (size() != o.size())
            throw std::invalid_argument("Vec::- size mismatch");
        Vec r(size());
        for (std::size_t i = 0; i < size(); ++i)
            r[i] = data_[i] - o[i];
        return r;
    }

    Vec Vec::operator*(double s) const
    {
        Vec r(size());
        for (std::size_t i = 0; i < size(); ++i)
            r[i] = data_[i] * s;
        return r;
    }

    Vec Vec::operator/(double s) const
    {
        if (s == 0.0)
            throw std::invalid_argument("Vec::/ division by zero");
        return *this * (1.0 / s);
    }

    Vec &Vec::operator+=(const Vec &o)
    {
        if (size() != o.size())
            throw std::invalid_argument("Vec::+= size mismatch");
        for (std::size_t i = 0; i < size(); ++i)
            data_[i] += o[i];
        return *this;
    }

    Vec &Vec::operator-=(const Vec &o)
    {
        if (size() != o.size())
            throw std::invalid_argument("Vec::-= size mismatch");
        for (std::size_t i = 0; i < size(); ++i)
            data_[i] -= o[i];
        return *this;
    }

    Vec &Vec::operator*=(double s)
    {
        for (double &v : data_)
            v *= s;
        return *this;
    }

    double Vec::dot(const Vec &o) const
    {
        if (size() != o.size())
            throw std::invalid_argument("Vec::dot size mismatch");
        double s = 0.0;
        for (std::size_t i = 0; i < size(); ++i)
            s += data_[i] * o[i];
        return s;
    }

    double Vec::sum() const
    {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    double Vec::mean() const
    {
        return data_.empty() ? 0.0 : sum() / static_cast<double>(size());
    }

    double Vec::max() const
    {
        if (data_.empty())
            return 0.0;
        return *std::max_element(data_.begin(), data_.end());
    }

    double Vec::min() const
    {
        if (data_.empty())
            return 0.0;
        return *std::min_element(data_.begin(), data_.end());
    }

    double Vec::norm() const { return std::sqrt(dot(*this)); }

    double Vec::norm_l1() const
    {
        double s = 0.0;
        for (double v : data_)
            s += std::fabs(v);
        return s;
    }

    Vec Vec::hadamard(const Vec &o) const
    {
        if (size() != o.size())
            throw std::invalid_argument("Vec::hadamard size mismatch");
        Vec r(size());
        for (std::size_t i = 0; i < size(); ++i)
            r[i] = data_[i] * o[i];
        return r;
    }

    Vec Vec::abs() const
    {
        Vec r(size());
        for (std::size_t i = 0; i < size(); ++i)
            r[i] = std::fabs(data_[i]);
        return r;
    }

    Vec Vec::clip(double lo, double hi) const
    {
        Vec r(size());
        for (std::size_t i = 0; i < size(); ++i)
            r[i] = clamp(data_[i], lo, hi);
        return r;
    }

    double Vec::variance() const
    {
        if (size() < 2)
            return 0.0;
        const double m = mean();
        double s = 0.0;
        for (double v : data_)
            s += (v - m) * (v - m);
        return s / static_cast<double>(size() - 1);
    }

    Vec Vec::zeros(std::size_t n) { return Vec(n, 0.0); }

    Vec Vec::ones(std::size_t n) { return Vec(n, 1.0); }

    Vec Vec::random(std::size_t n, double scale, std::mt19937 *rng)
    {
        static std::mt19937 fallback(1337u);
        std::mt19937 &g = rng ? *rng : fallback;
        std::normal_distribution<double> dist(0.0, scale);
        Vec r(n);
        for (std::size_t i = 0; i < n; ++i)
            r[i] = dist(g);
        return r;
    }

    std::string Vec::to_string(int precision) const
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(precision) << "[";
        for (std::size_t i = 0; i < size(); ++i)
        {
            os << data_[i];
            if (i + 1 < size())
                os << ", ";
        }
        os << "]";
        return os.str();
    }

    // ===========================================================================
    // Mat
    // ===========================================================================
    Mat::Mat(std::size_t rows, std::size_t cols, double fill)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    Vec Mat::row(std::size_t r) const
    {
        if (r >= rows_)
            throw std::out_of_range("Mat::row out of range");
        Vec v(cols_);
        for (std::size_t c = 0; c < cols_; ++c)
            v[c] = at(r, c);
        return v;
    }

    Vec Mat::col(std::size_t c) const
    {
        if (c >= cols_)
            throw std::out_of_range("Mat::col out of range");
        Vec v(rows_);
        for (std::size_t r = 0; r < rows_; ++r)
            v[r] = at(r, c);
        return v;
    }

    void Mat::set_row(std::size_t r, const Vec &v)
    {
        if (v.size() != cols_)
            throw std::invalid_argument("set_row size mismatch");
        for (std::size_t c = 0; c < cols_; ++c)
            at(r, c) = v[c];
    }

    void Mat::set_col(std::size_t c, const Vec &v)
    {
        if (v.size() != rows_)
            throw std::invalid_argument("set_col size mismatch");
        for (std::size_t r = 0; r < rows_; ++r)
            at(r, c) = v[r];
    }

    Mat Mat::transpose() const
    {
        Mat t(cols_, rows_);
        for (std::size_t r = 0; r < rows_; ++r)
            for (std::size_t c = 0; c < cols_; ++c)
                t.at(c, r) = at(r, c);
        return t;
    }

    Vec Mat::operator*(const Vec &v) const
    {
        if (v.size() != cols_)
            throw std::invalid_argument("Mat::* vector size mismatch");
        Vec r(rows_);
        for (std::size_t i = 0; i < rows_; ++i)
        {
            double s = 0.0;
            for (std::size_t j = 0; j < cols_; ++j)
                s += at(i, j) * v[j];
            r[i] = s;
        }
        return r;
    }

    Mat Mat::operator*(const Mat &o) const
    {
        if (cols_ != o.rows_)
            throw std::invalid_argument("Mat::* matrix size mismatch");
        Mat r(rows_, o.cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k)
            {
                const double a = at(i, k);
                if (a == 0.0)
                    continue;
                for (std::size_t j = 0; j < o.cols_; ++j)
                    r.at(i, j) += a * o.at(k, j);
            }
        return r;
    }

    Mat Mat::operator+(const Mat &o) const
    {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument("Mat::+ size mismatch");
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            r.data_[i] = data_[i] + o.data_[i];
        return r;
    }

    Mat Mat::operator-(const Mat &o) const
    {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument("Mat::- size mismatch");
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            r.data_[i] = data_[i] - o.data_[i];
        return r;
    }

    Mat Mat::operator*(double s) const
    {
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i)
            r.data_[i] = data_[i] * s;
        return r;
    }

    Mat Mat::outer(const Vec &u, const Vec &v)
    {
        Mat r(u.size(), v.size());
        for (std::size_t i = 0; i < u.size(); ++i)
            for (std::size_t j = 0; j < v.size(); ++j)
                r.at(i, j) = u[i] * v[j];
        return r;
    }

    Mat Mat::add_row_vector(const Vec &bias) const
    {
        if (bias.size() != cols_)
            throw std::invalid_argument("add_row_vector size mismatch");
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j)
                r.at(i, j) = at(i, j) + bias[j];
        return r;
    }

    double Mat::sum() const
    {
        return std::accumulate(data_.begin(), data_.end(), 0.0);
    }

    double Mat::frobenius() const
    {
        double s = 0.0;
        for (double v : data_)
            s += v * v;
        return std::sqrt(s);
    }

    Mat Mat::random(std::size_t rows, std::size_t cols, double scale,
                    std::mt19937 *rng)
    {
        static std::mt19937 fallback(4242u);
        std::mt19937 &g = rng ? *rng : fallback;
        std::normal_distribution<double> dist(0.0, scale);
        Mat m(rows, cols);
        for (std::size_t i = 0; i < m.data_.size(); ++i)
            m.data_[i] = dist(g);
        return m;
    }

    Mat Mat::identity(std::size_t n)
    {
        Mat m(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            m.at(i, i) = 1.0;
        return m;
    }

    std::string Mat::to_string(int precision) const
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(precision);
        os << "Mat(" << rows_ << "x" << cols_ << ")\n";
        for (std::size_t r = 0; r < rows_; ++r)
        {
            os << "  [";
            for (std::size_t c = 0; c < cols_; ++c)
            {
                os << at(r, c);
                if (c + 1 < cols_)
                    os << ", ";
            }
            os << "]\n";
        }
        return os.str();
    }

    // ===========================================================================
    // Activations
    // ===========================================================================
    namespace activation
    {

        double sigmoid(double z)
        {
            // Stable for large |z| (mirrors nn/activations.py).
            if (z >= 0.0)
            {
                const double e = std::exp(-z);
                return 1.0 / (1.0 + e);
            }
            const double e = std::exp(z);
            return e / (1.0 + e);
        }

        double sigmoid_prime_from_output(double a) { return a * (1.0 - a); }

        double tanh_f(double z) { return std::tanh(z); }

        double tanh_prime_from_output(double a) { return 1.0 - a * a; }

        double relu(double z) { return z > 0.0 ? z : 0.0; }

        double relu_prime(double z) { return z > 0.0 ? 1.0 : 0.0; }

        double leaky_relu(double z, double alpha) { return z > 0.0 ? z : alpha * z; }

        double leaky_relu_prime(double z, double alpha) { return z > 0.0 ? 1.0 : alpha; }

        double linear(double z) { return z; }

        double linear_prime(double) { return 1.0; }

    } // namespace activation

    Vec softmax(const Vec &z)
    {
        const double m = z.max();
        Vec e(z.size());
        double total = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i)
        {
            e[i] = std::exp(z[i] - m);
            total += e[i];
        }
        if (total <= 0.0)
            total = 1e-12;
        for (std::size_t i = 0; i < e.size(); ++i)
            e[i] /= total;
        return e;
    }

    double cross_entropy(const Vec &logits, const Vec &target)
    {
        const Vec p = softmax(logits);
        double loss = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i)
        {
            const double pi = clamp(p[i], 1e-12, 1.0);
            loss -= target[i] * std::log(pi);
        }
        return loss;
    }

    Vec cross_entropy_grad(const Vec &logits, const Vec &target)
    {
        return softmax(logits) - target;
    }

    double mse(const Vec &a, const Vec &b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("mse size mismatch");
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const double d = a[i] - b[i];
            s += d * d;
        }
        return s / static_cast<double>(a.size());
    }

    Vec mse_grad(const Vec &a, const Vec &b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("mse_grad size mismatch");
        Vec g(a.size());
        const double n = static_cast<double>(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            g[i] = 2.0 * (a[i] - b[i]) / n;
        return g;
    }

    std::ostream &operator<<(std::ostream &os, const Vec &v)
    {
        os << v.to_string();
        return os;
    }

    std::ostream &operator<<(std::ostream &os, const Mat &m)
    {
        os << m.to_string();
        return os;
    }

} // namespace chappie
