#ifndef EMTFNN_H_
#define EMTFNN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

namespace hls4ml_emtfnn_v1 {

// Prototype of top level function for C-synthesis
void emtfnn(
    hls4ml_emtfnn_v1::input_t input1[N_INPUT_1_1],
    hls4ml_emtfnn_v1::result_t layer12_out[N_LAYER_11]
);

} // hls4ml_emtfnn_v1

#endif
