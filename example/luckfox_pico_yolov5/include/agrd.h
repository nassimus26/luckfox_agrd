// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _AGRD_H_
#define _AGRD_H_

#include "rknn.h"

int const nbrImgs = 8;
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    rknn_tensor_mem* net_mem;
#if defined(RV1106_1103) 
    rknn_tensor_mem* input_mems[nbrImgs];
    rknn_tensor_mem* output_mems[1];
    rknn_dma_buf img_dma_buf;
#endif
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_agrd_context_t;

int init_agrd_model(const char* model_path, rknn_agrd_context_t* app_ctx);

int release_agrd_model(rknn_agrd_context_t* app_ctx);

int inference_agrd_model(rknn_agrd_context_t* app_ctx);
#endif