    // ---------------------------------------------------------------------------
// ai/neural_network.hpp -- a fully-connected feed-forward network with
// backpropagation, trained online. This is the C++ counterpart of the Python
// `nn` package (layers + losses + optimizers), sized for the robot's brain.
// ---------------------------------------------------------------------------
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/math.hpp"
#include "core/json.hpp"

namespace chappie
{

    enum class ActivationKind
    {
        Sigmoid,
        Tanh,
        Relu,
        LeakyRelu,
        Linear
    };

    ActivationKind parse_activation(const std::string &name);
    const char *activation_name(ActivationKind kind);

    // ---------------------------------------------------------------------------
    // One dense layer: y = activation(W x + b)
    //   forward caches (input, pre-activation z, output a)
    //   backward(dout) computes dW, db and returns dx
    //   Mirrors nn/layers.py Dense.
    // ---------------------------------------------------------------------------
    class DenseLayer
    {
    public:
        DenseLayer() = default;
        DenseLayer(std::size_t in_features, std::size_t out_features,
                   ActivationKind act = ActivationKind::Relu, double init_scale = 0.1,
                   std::mt19937 *rng = nullptr, bool use_bias = true);

        Vec forward(const Vec &x);
        Vec backward(const Vec &dout);

        std::size_t in_features() const { return w_.cols(); }
        std::size_t out_features() const { return w_.rows(); }

        const Mat &weights() const { return w_; }
        const Vec &bias() const { return b_; }
        const Mat &weight_grad() const { return dw_; }
        const Vec &bias_grad() const { return db_; }

        void set_weights(const Mat &w, const Vec &b);

        // Apply one SGD-style update using accumulated gradients and `lr`.
        void apply_gradients(double lr, double weight_decay = 0.0);

        void zero_grad();

        ActivationKind activation() const { return act_; }
        void set_activation(ActivationKind a) { act_ = a; }

        // Serialization for checkpoints.
        Json to_json() const;
        static DenseLayer from_json(const Json &j);

    private:
        Mat w_;
        Vec b_;
        Mat dw_;
        Vec db_;

        ActivationKind act_ = ActivationKind::Relu;
        bool use_bias_ = true;

        // caches
        Vec x_;
        Vec z_;
        Vec a_;
    };

    // ---------------------------------------------------------------------------
    // NeuralNetwork -- a stack of DenseLayer, with softmax + cross-entropy output
    // (classification) or linear + MSE output (regression).
    // ---------------------------------------------------------------------------
    class NeuralNetwork
    {
    public:
        NeuralNetwork() = default;

        // architecture: {n_inputs, h1, h2, ..., n_outputs}
        NeuralNetwork(const std::vector<std::size_t> &architecture,
                      ActivationKind hidden_act = ActivationKind::Relu,
                      ActivationKind output_act = ActivationKind::Sigmoid,
                      std::uint32_t seed = 1337);

        // -- forward / backward ----------------------------------------------
        Vec forward(const Vec &x);
        double backward_from_target(const Vec &target); // MSE, returns loss
        double backward_from_logits(const Vec &target); // softmax+CE, returns loss

        Vec predict(const Vec &x);

        // One online training step (forward + backward + update).
        double train_step(const Vec &x, const Vec &target, double lr);

        // Batch training over (input, target) pairs.
        double train(const std::vector<Vec> &inputs,
                     const std::vector<Vec> &targets,
                     int epochs, double lr, bool shuffle = true);

        void zero_grad();
        void apply_gradients(double lr, double weight_decay = 0.0);

        std::size_t num_parameters() const;
        std::size_t input_size() const;
        std::size_t output_size() const;

        bool uses_softmax_output() const { return softmax_output_; }
        void set_softmax_output(bool v) { softmax_output_ = v; }

        // Serialization.
        Json to_json() const;
        static NeuralNetwork from_json(const Json &j);
        bool save(const std::string &path) const;
        bool load(const std::string &path);

        const std::vector<DenseLayer> &layers() const { return layers_; }
        std::vector<DenseLayer> &layers() { return layers_; }

        std::string summary() const;

    private:
        std::vector<DenseLayer> layers_;
        bool softmax_output_ = false;

        // caches for the output head
        Vec last_z_;
        Vec last_a_;
    };

    // ---------------------------------------------------------------------------
    // Gradient clipping -- essential for stable recurrent/robot learning.
    // ---------------------------------------------------------------------------
    double clip_gradients(std::vector<DenseLayer> &layers, double max_norm);

} // namespace chappie
