#ifndef EMTFNN_H_
#define EMTFNN_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void emtfnn(
    input_t input1[N_INPUT_1_1],
    result_t layer12_out[N_LAYER_11]
);

#endif
