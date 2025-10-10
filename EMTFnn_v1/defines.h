#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 29
#define N_INPUT_1_1 29
#define N_LAYER_3 20
#define N_LAYER_3 20
#define N_LAYER_7 16
#define N_LAYER_7 16
#define N_LAYER_11 2
#define N_LAYER_11 2

// hls-fpga-machine-learning insert layer-precision
typedef ap_uint<13> input_t;
typedef ap_fixed<25,9> layer2_t;
typedef ap_fixed<25,9> batch_normalization_scale_t;
typedef ap_fixed<25,9> batch_normalization_bias_t;
typedef ap_fixed<25,10> model_default_t;
typedef ap_fixed<25,9> layer3_t;
typedef ap_fixed<25,9> dense_weight_t;
typedef ap_fixed<25,9> bias3_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<18,8> layer6_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<25,9> layer7_t;
typedef ap_fixed<25,9> dense_1_weight_t;
typedef ap_fixed<25,9> bias7_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<18,8> layer10_t;
typedef ap_fixed<18,8> activation_1_table_t;
typedef ap_fixed<24,9,AP_TRN,AP_SAT> layer11_t;
typedef ap_fixed<25,9> dense_2_weight_t;
typedef ap_fixed<25,9> dense_2_bias_t;
typedef ap_uint<1> layer11_index;
typedef ap_uint<8> result_t;
typedef ap_fixed<18,8> dense_2_relu_table_t;

#endif
