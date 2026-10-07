// ---------------------------------------------------------------------------
// ai/neural_network.cpp -- DenseLayer and NeuralNetwork implementation.
// ---------------------------------------------------------------------------
#include "ai/neural_network.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "core/logger.hpp"

namespace chappie
{

    // ===========================================================================
    // Activation helpers
    // ===========================================================================
    ActivationKind parse_activation(const std::string &name)
    {
        if (name == "sigmoid")
            return ActivationKind::Sigmoid;
        if (name == "tanh")
            return ActivationKind::Tanh;
        if (name == "relu")
            return ActivationKind::Relu;
        if (name == "leaky_relu")
            return ActivationKind::LeakyRelu;
        if (name == "linear" || name == "identity")
            return ActivationKind::Linear;
        return ActivationKind::Relu;
    }

    const char *activation_name(ActivationKind kind)
    {
        switch (kind)
        {
        case ActivationKind::Sigmoid:
            return "sigmoid";
        case ActivationKind::Tanh:
            return "tanh";
        case ActivationKind::Relu:
            return "relu";
        case ActivationKind::LeakyRelu:
            return "leaky_relu";
        case ActivationKind::Linear:
            return "linear";
        }
        return "?";
    }

    namespace
    {

        Vec apply_activation(ActivationKind kind, const Vec &z)
        {
            Vec a(z.size());
            for (std::size_t i = 0; i < z.size(); ++i)
            {
                switch (kind)
                {
                case ActivationKind::Sigmoid:
                    a[i] = activation::sigmoid(z[i]);
                    break;
                case ActivationKind::Tanh:
                    a[i] = activation::tanh_f(z[i]);
                    break;
                case ActivationKind::Relu:
                    a[i] = activation::relu(z[i]);
                    break;
                case ActivationKind::LeakyRelu:
                    a[i] = activation::leaky_relu(z[i]);
                    break;
                case ActivationKind::Linear:
                    a[i] = activation::linear(z[i]);
                    break;
                }
            }
            return a;
        }

        // d(activation)/dz evaluated element-wise, given z and the output a.
        Vec activation_grad(ActivationKind kind, const Vec &z, const Vec &a)
        {
            Vec g(z.size());
            for (std::size_t i = 0; i < z.size(); ++i)
            {
                switch (kind)
                {
                case ActivationKind::Sigmoid:
                    g[i] = activation::sigmoid_prime_from_output(a[i]);
                    break;
                case ActivationKind::Tanh:
                    g[i] = activation::tanh_prime_from_output(a[i]);
                    break;
                case ActivationKind::Relu:
                    g[i] = activation::relu_prime(z[i]);
                    break;
                case ActivationKind::LeakyRelu:
                    g[i] = activation::leaky_relu_prime(z[i]);
                    break;
                case ActivationKind::Linear:
                    g[i] = activation::linear_prime(z[i]);
                    break;
                }
            }
            return g;
        }

    } // namespace

    // ===========================================================================
    // DenseLayer
    // ===========================================================================
    DenseLayer::DenseLayer(std::size_t in_features, std::size_t out_features,
                           ActivationKind act, double init_scale, std::mt19937 *rng,
                           bool use_bias)
        : w_(Mat::random(out_features, in_features, init_scale, rng)),
          b_(Vec::zeros(out_features)),
          dw_(out_features, in_features, 0.0),
          db_(out_features, 0.0),
          act_(act),
          use_bias_(use_bias) {}

    Vec DenseLayer::forward(const Vec &x)
    {
        if (x.size() != w_.cols())
            throw std::invalid_argument("DenseLayer::forward input size mismatch");
        x_ = x;
        z_ = w_ * x;
        if (use_bias_)
            z_ += b_;
        a_ = apply_activation(act_, z_);
        return a_;
    }

    Vec DenseLayer::backward(const Vec &dout)
    {
        if (dout.size() != a_.size())
            throw std::invalid_argument("DenseLayer::backward size mismatch");

        // dL/dz = dL/da * activation'(z)
        const Vec dz = dout.hadamard(activation_grad(act_, z_, a_));

        // dW = dz x^T     (accumulate)
        const Mat outer = Mat::outer(dz, x_);
        dw_ = dw_ + outer;

        // db = dz         (accumulate)
        if (use_bias_)
            db_ += dz;

        // dx = W^T dz
        return w_.transpose() * dz;
    }

    void DenseLayer::set_weights(const Mat &w, const Vec &b)
    {
        if (w.rows() != w_.rows() || w.cols() != w_.cols())
            throw std::invalid_argument("set_weights shape mismatch");
        w_ = w;
        if (use_bias_ && b.size() == b_.size())
            b_ = b;
    }

    void DenseLayer::apply_gradients(double lr, double weight_decay)
    {
        for (std::size_t r = 0; r < w_.rows(); ++r)
            for (std::size_t c = 0; c < w_.cols(); ++c)
            {
                double g = dw_.at(r, c);
                if (weight_decay > 0.0)
                    g += weight_decay * w_.at(r, c);
                w_.at(r, c) -= lr * g;
            }
        if (use_bias_)
            for (std::size_t i = 0; i < b_.size(); ++i)
                b_[i] -= lr * db_[i];
    }

    void DenseLayer::zero_grad()
    {
        dw_ = Mat(dw_.rows(), dw_.cols(), 0.0);
        db_ = Vec::zeros(db_.size());
    }

    Json DenseLayer::to_json() const
    {
        Json j = Json::object();
        j.set("in", Json(static_cast<int>(w_.cols())));
        j.set("out", Json(static_cast<int>(w_.rows())));
        j.set("activation", Json(activation_name(act_)));
        j.set("use_bias", Json(use_bias_));

        Json w = Json::array();
        for (std::size_t r = 0; r < w_.rows(); ++r)
        {
            Json row = Json::array();
            for (std::size_t c = 0; c < w_.cols(); ++c)
                row.push_back(Json(w_.at(r, c)));
            w.push_back(row);
        }
        j.set("W", w);

        Json b = Json::array();
        for (double v : b_.raw())
            b.push_back(Json(v));
        j.set("b", b);
        return j;
    }

    DenseLayer DenseLayer::from_json(const Json &j)
    {
        const std::size_t in = static_cast<std::size_t>(j.get_int("in", 1));
        const std::size_t out = static_cast<std::size_t>(j.get_int("out", 1));
        DenseLayer layer(in, out, parse_activation(j.get_string("activation", "relu")),
                         0.1, nullptr, j.get_bool("use_bias", true));
        const Json &w = j["W"];
        Mat m(out, in, 0.0);
        for (std::size_t r = 0; r < out && r < w.size(); ++r)
            for (std::size_t c = 0; c < in && c < w[r].size(); ++c)
                m.at(r, c) = w[r][c].as_number();
        Vec b(out, 0.0);
        const Json &bj = j["b"];
        for (std::size_t i = 0; i < out && i < bj.size(); ++i)
            b[i] = bj[i].as_number();
        layer.set_weights(m, b);
        return layer;
    }

    // ===========================================================================
    // NeuralNetwork
    // ===========================================================================
    NeuralNetwork::NeuralNetwork(const std::vector<std::size_t> &architecture,
                                 ActivationKind hidden_act,
                                 ActivationKind output_act, std::uint32_t seed)
    {
        if (architecture.size() < 2)
            throw std::invalid_argument("architecture needs at least 2 sizes");
        std::mt19937 rng(seed);
        for (std::size_t i = 0; i + 1 < architecture.size(); ++i)
        {
            const bool is_output = (i + 2 == architecture.size());
            const ActivationKind act = is_output ? output_act : hidden_act;
            // He-style scaling for relu, Xavier otherwise.
            const double scale =
                (hidden_act == ActivationKind::Relu)
                    ? std::sqrt(2.0 / static_cast<double>(architecture[i]))
                    : std::sqrt(1.0 / static_cast<double>(architecture[i]));
            layers_.emplace_back(architecture[i], architecture[i + 1], act, scale, &rng);
        }
    }

    Vec NeuralNetwork::forward(const Vec &x)
    {
        Vec h = x;
        for (auto &layer : layers_)
            h = layer.forward(h);
        last_a_ = h;
        return h;
    }

    Vec NeuralNetwork::predict(const Vec &x)
    {
        Vec out = forward(x);
        if (softmax_output_)
            out = softmax(out);
        return out;
    }

    double NeuralNetwork::backward_from_target(const Vec &target)
    {
        // MSE at the output: dL/da = 2(a - y)/n
        const double loss = mse(last_a_, target);
        Vec dout = mse_grad(last_a_, target);
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            dout = it->backward(dout);
        return loss;
    }

    double NeuralNetwork::backward_from_logits(const Vec &target)
    {
        // Softmax + cross-entropy fused: dL/dz = (p - y)
        const double loss = cross_entropy(last_a_, target);
        Vec dout = cross_entropy_grad(last_a_, target);
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
            dout = it->backward(dout);
        return loss;
    }

    double NeuralNetwork::train_step(const Vec &x, const Vec &target, double lr)
    {
        forward(x);
        const double loss = softmax_output_ ? backward_from_logits(target)
                                            : backward_from_target(target);
        apply_gradients(lr);
        zero_grad();
        return loss;
    }

    double NeuralNetwork::train(const std::vector<Vec> &inputs,
                                const std::vector<Vec> &targets, int epochs,
                                double lr, bool shuffle)
    {
        if (inputs.size() != targets.size())
            throw std::invalid_argument("train: inputs/targets size mismatch");
        if (inputs.empty())
            return 0.0;

        std::mt19937 rng(20240607u);
        std::vector<std::size_t> order(inputs.size());
        std::iota(order.begin(), order.end(), 0u);

        double last_loss = 0.0;
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            if (shuffle)
                std::shuffle(order.begin(), order.end(), rng);
            double total = 0.0;
            for (std::size_t idx : order)
                total += train_step(inputs[idx], targets[idx], lr);
            last_loss = total / static_cast<double>(inputs.size());
        }
        return last_loss;
    }

    void NeuralNetwork::zero_grad()
    {
        for (auto &layer : layers_)
            layer.zero_grad();
    }

    void NeuralNetwork::apply_gradients(double lr, double weight_decay)
    {
        for (auto &layer : layers_)
            layer.apply_gradients(lr, weight_decay);
    }

    std::size_t NeuralNetwork::num_parameters() const
    {
        std::size_t n = 0;
        for (const auto &layer : layers_)
            n += layer.weights().rows() * layer.weights().cols() +
                 layer.bias().size();
        return n;
    }

    std::size_t NeuralNetwork::input_size() const
    {
        return layers_.empty() ? 0 : layers_.front().in_features();
    }

    std::size_t NeuralNetwork::output_size() const
    {
        return layers_.empty() ? 0 : layers_.back().out_features();
    }

    std::string NeuralNetwork::summary() const
    {
        std::ostringstream os;
        os << "NeuralNetwork(" << num_parameters() << " params)\n";
        for (std::size_t i = 0; i < layers_.size(); ++i)
        {
            const auto &l = layers_[i];
            const std::size_t n =
                l.weights().rows() * l.weights().cols() + l.bias().size();
            os << "  [" << i << "] Dense(" << l.in_features() << " -> "
               << l.out_features() << ", " << activation_name(l.activation()) << ") "
               << n << " params\n";
        }
        if (softmax_output_)
            os << "  output: softmax + cross-entropy\n";
        return os.str();
    }

    Json NeuralNetwork::to_json() const
    {
        Json j = Json::object();
        j.set("softmax_output", Json(softmax_output_));
        Json arr = Json::array();
        for (const auto &layer : layers_)
            arr.push_back(layer.to_json());
        j.set("layers", arr);
        return j;
    }

    NeuralNetwork NeuralNetwork::from_json(const Json &j)
    {
        NeuralNetwork net;
        net.softmax_output_ = j.get_bool("softmax_output", false);
        const Json &arr = j["layers"];
        for (std::size_t i = 0; i < arr.size(); ++i)
            net.layers_.push_back(DenseLayer::from_json(arr[i]));
        return net;
    }

    bool NeuralNetwork::save(const std::string &path) const
    {
        return to_json().save_file(path);
    }

    bool NeuralNetwork::load(const std::string &path)
    {
        Json j;
        std::string err;
        if (!Json::load_file(path, j, err))
        {
            LOG_ERROR() << "NeuralNetwork::load failed: " << err;
            return false;
        }
        *this = NeuralNetwork::from_json(j);
        return true;
    }

    // ===========================================================================
    // Gradient clipping
    // ===========================================================================
    double clip_gradients(std::vector<DenseLayer> &layers, double max_norm)
    {
        double total = 0.0;
        for (const auto &layer : layers)
        {
            const Mat &g = layer.weight_grad();
            for (std::size_t r = 0; r < g.rows(); ++r)
                for (std::size_t c = 0; c < g.cols(); ++c)
                {
                    const double v = g.at(r, c);
                    total += v * v;
                }
            const Vec &bg = layer.bias_grad();
            for (double v : bg.raw())
                total += v * v;
        }
        const double norm = std::sqrt(total);
        if (norm <= max_norm || norm == 0.0)
            return norm;
        const double scale = max_norm / norm;
        for (auto &layer : layers)
        {
            Mat &g = const_cast<Mat &>(layer.weight_grad());
            for (std::size_t r = 0; r < g.rows(); ++r)
                for (std::size_t c = 0; c < g.cols(); ++c)
                    g.at(r, c) *= scale;
            Vec &bg = const_cast<Vec &>(layer.bias_grad());
            for (double &v : bg.raw())
                v *= scale;
        }
        return norm;
    }

} // namespace chappie
