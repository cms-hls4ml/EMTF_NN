#include <iostream>

#include "emtfnn.h"
#include "parameters.h"

namespace hls4ml_emtfnn_v1 {

void emtfnn(
    input_t input1[N_INPUT_1_1],
    result_t layer12_out[N_LAYER_11]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    //#pragma HLS INTERFACE ap_vld port=input1,layer12_out 
    //#pragma HLS PIPELINE 
    #pragma HLS INLINE

#ifdef LOAD_WEIGHTS_FROM_TXT
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<batch_normalization_scale_t, 29>(s2, "s2.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 29>(b2, "b2.txt");
        nnet::load_weights_from_txt<dense_weight_t, 580>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 20>(b3, "b3.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 320>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 16>(b7, "b7.txt");
        nnet::load_weights_from_txt<dense_2_weight_t, 32>(w11, "w11.txt");
        nnet::load_weights_from_txt<dense_2_bias_t, 2>(b11, "b11.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_INPUT_1_1];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::normalize<input_t, layer2_t, config2>(input1, layer2_out, s2, b2); // batch_normalization

    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // dense

    layer6_t layer6_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::relu<layer3_t, layer6_t, relu_config6>(layer3_out, layer6_out); // activation

    layer7_t layer7_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::dense<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7); // dense_1

    layer10_t layer10_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::relu<layer7_t, layer10_t, relu_config10>(layer7_out, layer10_out); // activation_1

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer10_t, layer11_t, config11>(layer10_out, layer11_out, w11, b11); // dense_2

    nnet::relu<layer11_t, result_t, relu_config12>(layer11_out, layer12_out); // dense_2_relu

}

} // hls4ml_emtfnn_v1
