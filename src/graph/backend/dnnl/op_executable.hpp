/*******************************************************************************
 * Copyright 2021-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
#ifndef GRAPH_BACKEND_DNNL_OP_EXECUTABLE_HPP
#define GRAPH_BACKEND_DNNL_OP_EXECUTABLE_HPP

// Include all the executables
#include "graph/backend/dnnl/executables/base.hpp"
#include "graph/backend/dnnl/executables/batch_norm.hpp"
#include "graph/backend/dnnl/executables/concat.hpp"
#include "graph/backend/dnnl/executables/const_memory_filler.hpp"
#include "graph/backend/dnnl/executables/conv.hpp"
#include "graph/backend/dnnl/executables/eltwise.hpp"
#include "graph/backend/dnnl/executables/gen_index.hpp"
#include "graph/backend/dnnl/executables/group_norm.hpp"
#include "graph/backend/dnnl/executables/host_scalar.hpp"
#include "graph/backend/dnnl/executables/layer_norm.hpp"
#include "graph/backend/dnnl/executables/matmul.hpp"
#include "graph/backend/dnnl/executables/memory_reparser.hpp"
#include "graph/backend/dnnl/executables/pool.hpp"
#include "graph/backend/dnnl/executables/reduction.hpp"
#include "graph/backend/dnnl/executables/reorder.hpp"
#include "graph/backend/dnnl/executables/resampling.hpp"
#include "graph/backend/dnnl/executables/sdpa.hpp"
#include "graph/backend/dnnl/executables/shuffle.hpp"
#include "graph/backend/dnnl/executables/softmax.hpp"
#include "graph/backend/dnnl/executables/sum.hpp"

#endif
