#include "common.h"
#include "hyp_container.h"
#include "utils.h"

using namespace zh;

class CapsuleConv
{
    public:
        explicit CapsuleConv(HypContainer &hyp):
            dim_capsule(hyp.viget("dim_capsule")),
            dim_conv_rker(hyp.viget("dim_conv_rker")),
            dim_conv_cker(hyp.viget("dim_conv_rker")),
            dim_conv_rstrd(hyp.viget("dim_conv_rstrd")),
            dim_conv_cstrd(hyp.viget("dim_conv_rstrd")),
            num_filter(hyp.viget("num_filter")),
            dim_fc(hyp.viget("dim_fc")),
            batch_size(hyp.iget("batch_size")),
            num_channel(hyp.iget("num_channel")),
            dim_y(hyp.iget("dim_y")),
            num_routing(hyp.iget("num_routing")),
            m_plus(hyp.fget("m_plus")),
            m_minus(hyp.fget("m_minus")),
            lambda(hyp.fget("lambda"))
        {}

        // the Conv1 layer, input shape = (batch_size, 28, 28)
        Symbol conv1_layer(Symbol input, vector<pair<string, Shape>> &arg_shapes)
        {
            const string w_name("w_conv1");
            const string b_name("b_conv1");

            arg_shapes.push_back({w_name, Shape(num_filter[0], num_channel, dim_conv_rker[0], dim_conv_cker[0])});
            arg_shapes.push_back({b_name, Shape(num_filter[0])});

            auto conv = Convolution(input,
                    Symbol(w_name),
                    Symbol(b_name),
                    Shape(dim_conv_rker[0], dim_conv_cker[0]),
                    num_filter[0],
                    Shape(dim_conv_rstrd[0], dim_conv_cstrd[0]));
            auto nolinear = Activation(conv, ActivationActType::kRelu);
            return nolinear;
        }

        // the PrimaryCaps layer, input shape = (batch_size, 256, 20, 20)
        Symbol primary_caps_layer(Symbol input, vector<pair<string, Shape>> &arg_shapes)
        {
            vector<Symbol> capsules;
            for (int i = 0; i < dim_capsule[0]; ++i)
            {
                const string w_name("w_conv1_cap" + to_string(i + 1));
                const string b_name("b_conv1_cap" + to_string(i + 1));
                arg_shapes.push_back({w_name, Shape(num_filter[1], num_filter[0], dim_conv_rker[1], dim_conv_cker[1])});
                arg_shapes.push_back({b_name, Shape(num_filter[1])});

                auto tmp = Convolution(input,
                        Symbol(w_name),
                        Symbol(b_name),
                        Shape(dim_conv_rker[1], dim_conv_cker[1]),
                        num_filter[1],
                        Shape(dim_conv_rstrd[1], dim_conv_cstrd[1])); // Shape(batch_size, 32, 6, 6)
                capsules.push_back(Reshape(tmp, Shape(batch_size, -1, 1))); 
            }
            return squash(Concat(capsules, dim_capsule[0], 2), 2); // shape = (batch_size, 1152, 8), 32 * 6 * 6 == 1152
        }

        // the DigitCaps layer
        tuple<Symbol, Symbol> digit_caps_layer(Symbol input, Symbol b_ij, vector<pair<string, Shape>> &arg_shapes)
        //pair<Symbol, Symbol> digit_caps_layer(Symbol input, Symbol b_ij, vector<pair<string, Shape>> &arg_shapes)
        {
            const string w_ij_name("w_ij");
            Symbol w_ij(w_ij_name); // Shape(8, 16)
            arg_shapes.push_back({w_ij_name, Shape(dim_capsule[0], dim_capsule[1])});

            Symbol v("v"), new_b_ij("new_b_ij");
            // routing algorithm
            for (int i = 0; i < num_routing; ++i)
            {
                // line 4
                auto c_ij = softmax(b_ij); // Shape(1152, 10)
                // line 5
                auto u_hat = dot(input, w_ij); // Shape(batch_size, 1152, 16)
                auto s = dot(SwapAxis(u_hat, 2, 1), c_ij); // Shape(batch_size, 16, 10)
                // line 6
                v = squash(s, 1);
                // line 7
                new_b_ij = b_ij + sum(batch_dot(u_hat, v), Shape(0)); // Shape(1152, 10)
            }

            return make_tuple(v, new_b_ij);
        }

        // input Shape(batch_size, 16, 10)
        Symbol margin_loss(Symbol input, Symbol y)
        {
            auto T_c = one_hot(y, dim_y); // Shape(batch_size, 10)
            auto loss = T_c * square(clip(m_plus - input, 0, m_plus)) +
                lambda * (1 - T_c) * square(clip(input - m_minus, 0, 1.0));

            return loss;
        }

        Symbol reconstruct_loss(Symbol input, Symbol x, Symbol y, vector<pair<string, Shape>> &arg_shapes)
        {
            vector<Symbol> layers;

            auto y_expand = expand_dims(y, 1);
            auto idx_pick = tile(y_expand, Shape(1, dim_capsule.back()));

            layers.push_back(pick(input, idx_pick, dmlc::optional<int>(2))); // activity vector, Shape(batch_size, 16)

            layers.push_back(FullyConnected(layers.back(),
                        Symbol("w_fc1"),
                        Symbol("b_fc1"),
                        dim_fc[0]));
            arg_shapes.push_back({"w_fc1", Shape(dim_fc[0], dim_capsule.back())});
            arg_shapes.push_back({"b_fc1", Shape(dim_fc[0])});
            layers.push_back(Activation(layers.back(), ActivationActType::kRelu));

            layers.push_back(FullyConnected(layers.back(),
                        Symbol("w_fc2"),
                        Symbol("b_fc2"),
                        dim_fc[1]));
            arg_shapes.push_back({"w_fc2", Shape(dim_fc[1], dim_fc[0])});
            arg_shapes.push_back({"b_fc2", Shape(dim_fc[1])});
            layers.push_back(Activation(layers.back(), ActivationActType::kRelu));

            layers.push_back(FullyConnected(layers.back(),
                        Symbol("w_fc3"),
                        Symbol("b_fc3"),
                        dim_fc[2]));
            arg_shapes.push_back({"w_fc3", Shape(dim_fc[2], dim_fc[1])});
            arg_shapes.push_back({"b_fc3", Shape(dim_fc[2])});
            layers.push_back(Activation(layers.back(), ActivationActType::kSigmoid));

            // sum of square loss
            auto loss = square(Reshape(x, Shape(batch_size, -1)) - layers.back());
            return loss; // Shape(batch_size, 784)
        }

    private:
        Symbol squash(Symbol input, const int axis)
        {
            auto input_norm = sqrt(sum(square(input), Shape(axis)));
            auto normalizer = square(input_norm) / (1 + square(input_norm)) / input_norm;
            return broadcast_mul(expand_dims(normalizer, axis), input);
        }

        vector<int> &dim_capsule;
        vector<int> &dim_conv_rker;
        vector<int> &dim_conv_cker;
        vector<int> &dim_conv_rstrd;
        vector<int> &dim_conv_cstrd;
        vector<int> &num_filter;
        vector<int> &dim_fc;
        int &batch_size;
        int &num_channel;
        int &dim_y;
        int &num_routing;
        mx_float &m_plus;
        mx_float &m_minus;
        mx_float &lambda;
};
