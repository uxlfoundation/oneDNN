#!/usr/bin/env bash

# *******************************************************************************
# Copyright 2024-2025 Arm Limited and affiliates.
# Copyright 2025 Intel Corporation
# Copyright 2026 Institute of Software, Chinese Academy of Sciences
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

# Test oneDNN for RISC-V.

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

get_filtered_tests() {
    local skipped_tests="$1"
    ctest -N -E "${skipped_tests}" 2>/dev/null \
            | sed -n 's/^  Test *#[0-9][0-9]*: //p'
}

regex_escape() {
    printf '%s' "$1" | sed -e 's/[][(){}.^$*+?|\\-]/\\&/g'
}

run_balanced_ci_tests() {
    local skipped_tests="$1"
    local part=${ONEDNN_TEST_PART:-1}
    local total_parts=${ONEDNN_TEST_STRIDE:-4}
    local default_cost=60
    local best_group best_cost best_count group cost test regex

    if (( part < 1 || part > total_parts )); then
        echo "Invalid test part: ${part}/${total_parts}"
        exit 1
    fi

    mapfile -t filtered_tests < <(get_filtered_tests "${skipped_tests}")

    if (( ${#filtered_tests[@]} == 0 )); then
        echo "No tests matched the balanced RISC-V partitioning input."
        exit 1
    fi

    # Per-test seconds from upstream main Weekly RISC-V, 2026-10-24,
    # https://github.com/uxlfoundation/oneDNN/actions/runs/40065222440
    # QEMU 10.2.3, VLEN=128, OMP, CTEST_PARALLEL_LEVEL=4.
    # Rounded up; unknown tests retain the default cost. These are
    # scheduling estimates, not predictions for a newer QEMU version.
    declare -A test_costs=(
        [api-c]=72
        [cpu-bnorm-u8-via-binary-postops-cpp]=1
        [cpu-cnn-inference-f32-c]=17
        [cpu-cnn-inference-f32-cpp]=791
        [cpu-cnn-inference-int8-cpp]=48
        [cpu-cnn-training-bf16-cpp]=2480
        [cpu-cnn-training-f32-c]=423
        [cpu-cnn-training-f32-cpp]=169
        [cpu-getting-started-cpp]=1
        [cpu-graph-gated-mlp-cpp]=700
        [cpu-graph-gated-mlp-int4-cpp]=3352
        [cpu-graph-gated-mlp-wei-combined-cpp]=734
        [cpu-graph-getting-started-cpp]=27
        [cpu-graph-gqa-cpp]=235
        [cpu-graph-gqa-training-cpp]=1087
        [cpu-graph-inference-int8-cpp]=17
        [cpu-graph-mqa-cpp]=125
        [cpu-graph-sdpa-bottom-right-causal-mask-cpp]=227
        [cpu-graph-sdpa-dynamic-quantized-cpp]=1
        [cpu-graph-sdpa-quantized-cpp]=42
        [cpu-graph-sdpa-stacked-qkv-cpp]=171
        [cpu-graph-single-op-partition-cpp]=3
        [cpu-matmul-f8-quantization-cpp]=2
        [cpu-matmul-perf-cpp]=23
        [cpu-matmul-weights-compression-cpp]=1
        [cpu-matmul-with-host-scalar-scale-cpp]=1
        [cpu-memory-format-propagation-cpp]=3
        [cpu-performance-profiling-cpp]=718
        [cpu-primitives-augru-cpp]=1
        [cpu-primitives-batch-normalization-cpp]=1
        [cpu-primitives-binary-cpp]=1
        [cpu-primitives-concat-cpp]=1
        [cpu-primitives-convolution-cpp]=2
        [cpu-primitives-deconvolution-cpp]=3
        [cpu-primitives-eltwise-cpp]=1
        [cpu-primitives-group-normalization-cpp]=28
        [cpu-primitives-inner-product-cpp]=18
        [cpu-primitives-layer-normalization-cpp]=2
        [cpu-primitives-lbr-gru-cpp]=2
        [cpu-primitives-lrn-cpp]=2
        [cpu-primitives-lstm-cpp]=2
        [cpu-primitives-matmul-cpp]=4
        [cpu-primitives-pooling-cpp]=2
        [cpu-primitives-prelu-cpp]=2
        [cpu-primitives-reduction-cpp]=1
        [cpu-primitives-reorder-cpp]=1
        [cpu-primitives-resampling-cpp]=2
        [cpu-primitives-shuffle-cpp]=7
        [cpu-primitives-softmax-cpp]=1
        [cpu-primitives-sum-cpp]=2
        [cpu-primitives-vanilla-rnn-cpp]=3
        [cpu-rnn-inference-f32-cpp]=64
        [cpu-rnn-inference-int8-cpp]=1
        [cpu-rnn-training-f32-cpp]=189
        [cpu-tutorials-matmul-inference-int8-matmul-cpp]=3
        [cpu-tutorials-matmul-matmul-quantization-cpp]=2
        [cpu-tutorials-matmul-matmul-with-weight-only-quantization-cpp]=9
        [cpu-tutorials-matmul-mxfp-matmul-cpp]=2
        [noexcept-cpp]=1
        [test_api]=4
        [test_batch_normalization]=63
        [test_benchdnn_modeC_augru_ci_cpu]=19
        [test_benchdnn_modeC_binary_ci_cpu]=356
        [test_benchdnn_modeC_binary_different_dt_ci_cpu]=638
        [test_benchdnn_modeC_bnorm_ci_cpu]=1727
        [test_benchdnn_modeC_brgemm_ci_cpu]=2
        [test_benchdnn_modeC_concat_ci_cpu]=495
        [test_benchdnn_modeC_conv_ci_cpu]=3801
        [test_benchdnn_modeC_deconv_ci_cpu]=932
        [test_benchdnn_modeC_eltwise_ci_cpu]=217
        [test_benchdnn_modeC_gnorm_ci_cpu]=345
        [test_benchdnn_modeC_gru_ci_cpu]=573
        [test_benchdnn_modeC_lnorm_ci_cpu]=2829
        [test_benchdnn_modeC_lrn_ci_cpu]=48
        [test_benchdnn_modeC_lstm_ci_cpu]=1544
        [test_benchdnn_modeC_pool_ci_cpu]=767
        [test_benchdnn_modeC_prelu_ci_cpu]=96
        [test_benchdnn_modeC_reduction_ci_cpu]=774
        [test_benchdnn_modeC_reorder_ci_cpu]=1187
        [test_benchdnn_modeC_resampling_ci_cpu]=81
        [test_benchdnn_modeC_rnn_ci_cpu]=746
        [test_benchdnn_modeC_sdpa_ci_cpu]=10
        [test_benchdnn_modeC_self_ci_cpu]=6
        [test_benchdnn_modeC_shuffle_ci_cpu]=12
        [test_benchdnn_modeC_softmax_ci_cpu]=659
        [test_benchdnn_modeC_sum_ci_cpu]=114
        [test_benchdnn_modeC_zeropad_ci_cpu]=8005
        [test_binary]=105
        [test_c_symbols-c]=2
        [test_concat]=16
        [test_concurrency]=2
        [test_convolution_backward_data_f32]=754
        [test_convolution_backward_weights_f32]=1072
        [test_convolution_eltwise_forward_f32]=637
        [test_convolution_eltwise_forward_x8s8f32s32]=385
        [test_convolution_format_any]=1
        [test_convolution_forward_f32]=911
        [test_convolution_forward_u8s8fp]=10
        [test_convolution_forward_u8s8s32]=21
        [test_cross_engine_reorder]=2
        [test_deconvolution]=23
        [test_eltwise]=106
        [test_gemm_bf16bf16bf16]=2
        [test_gemm_bf16bf16f32]=6175
        [test_gemm_f16]=2
        [test_gemm_f16f16f32]=2
        [test_gemm_f32]=2136
        [test_gemm_s8s8s32]=3399
        [test_gemm_s8u8s32]=2
        [test_gemm_u8s8s32]=2750
        [test_gemm_u8u8s32]=2
        [test_global_scratchpad]=1
        [test_graph_c_api_add_op_cpu]=1
        [test_graph_c_api_compile_cpu]=2
        [test_graph_c_api_compile_parametrized_cpu]=2
        [test_graph_c_api_constant_cache_cpu]=1
        [test_graph_c_api_filter_cpu]=3
        [test_graph_c_api_graph_cpu]=2
        [test_graph_c_api_graph_dump_cpu]=2
        [test_graph_c_api_logical_tensor_cpu]=2
        [test_graph_c_api_op_cpu]=1
        [test_graph_cpp_api_compile_cpu]=3
        [test_graph_cpp_api_constant_cache_cpu]=1
        [test_graph_cpp_api_engine_cpu]=2
        [test_graph_cpp_api_graph_cpu]=2
        [test_graph_cpp_api_graph_dump_cpu]=2
        [test_graph_cpp_api_logical_tensor_cpu]=1
        [test_graph_cpp_api_op_cpu]=2
        [test_graph_cpp_api_partition_cpu]=4
        [test_graph_cpp_api_tensor_cpu]=2
        [test_graph_unit_dnnl_batch_norm_cpu]=8
        [test_graph_unit_dnnl_binary_op_cpu]=9
        [test_graph_unit_dnnl_bmm_cpu]=4
        [test_graph_unit_dnnl_common_cpu]=4
        [test_graph_unit_dnnl_compiled_partition_cpu]=3
        [test_graph_unit_dnnl_concat_cpu]=5
        [test_graph_unit_dnnl_constant_cache_cpu]=3
        [test_graph_unit_dnnl_convolution_cpu]=152
        [test_graph_unit_dnnl_convtranspose_cpu]=34
        [test_graph_unit_dnnl_dequantize_cpu]=5
        [test_graph_unit_dnnl_dnnl_infer_shape_cpu]=4
        [test_graph_unit_dnnl_dnnl_utils_cpu]=3
        [test_graph_unit_dnnl_eltwise_cpu]=8
        [test_graph_unit_dnnl_fusion_info_cpu]=3
        [test_graph_unit_dnnl_graph_cpu]=2
        [test_graph_unit_dnnl_group_norm_cpu]=4
        [test_graph_unit_dnnl_insert_ops_cpu]=3
        [test_graph_unit_dnnl_interpolate_cpu]=2
        [test_graph_unit_dnnl_large_partition_cpu]=84
        [test_graph_unit_dnnl_layer_norm_cpu]=4
        [test_graph_unit_dnnl_layout_id_cpu]=3
        [test_graph_unit_dnnl_layout_propagator_cpu]=16
        [test_graph_unit_dnnl_logical_tensor_cpu]=2
        [test_graph_unit_dnnl_matmul_cpu]=101
        [test_graph_unit_dnnl_memory_planning_cpu]=3
        [test_graph_unit_dnnl_mqa_decomp_cpu]=4689
        [test_graph_unit_dnnl_op_executable_cpu]=14
        [test_graph_unit_dnnl_op_schema_cpu]=2
        [test_graph_unit_dnnl_partition_cpu]=2
        [test_graph_unit_dnnl_pass_cpu]=10
        [test_graph_unit_dnnl_pool_cpu]=10
        [test_graph_unit_dnnl_prelu_cpu]=4
        [test_graph_unit_dnnl_quantize_cpu]=5
        [test_graph_unit_dnnl_reduce_cpu]=8
        [test_graph_unit_dnnl_reorder_cpu]=6
        [test_graph_unit_dnnl_scratchpad_cpu]=3
        [test_graph_unit_dnnl_sdp_decomp_cpu]=2663
        [test_graph_unit_dnnl_select_cpu]=4
        [test_graph_unit_dnnl_softmax_cpu]=3
        [test_graph_unit_dnnl_subgraph_pass_cpu]=8
        [test_graph_unit_dnnl_thread_local_cache_cpu]=3
        [test_graph_unit_dnnl_typecast_cpu]=4
        [test_graph_unit_fake_cpu]=4
        [test_graph_unit_interface_allocator_cpu]=4
        [test_graph_unit_interface_backend_cpu]=4
        [test_graph_unit_interface_compiled_partition_cpu]=4
        [test_graph_unit_interface_graph_cpu]=3
        [test_graph_unit_interface_logical_tensor_cpu]=2
        [test_graph_unit_interface_op_cpu]=3
        [test_graph_unit_interface_op_def_constraint_cpu]=2
        [test_graph_unit_interface_op_schema_cpu]=35
        [test_graph_unit_interface_partition_hashing_cpu]=2
        [test_graph_unit_interface_shape_infer_cpu]=5
        [test_graph_unit_interface_tensor_cpu]=3
        [test_graph_unit_interface_value_cpu]=4
        [test_graph_unit_utils_allocator_cpu]=3
        [test_graph_unit_utils_attribute_value_cpu]=4
        [test_graph_unit_utils_debug_cpu]=17
        [test_graph_unit_utils_json_cpu]=3
        [test_graph_unit_utils_pattern_matcher_cpu]=3
        [test_graph_unit_utils_utils_cpu]=2
        [test_group_normalization]=56
        [test_iface_attr]=3
        [test_iface_attr_quantization]=2
        [test_iface_binary_bcast]=2
        [test_iface_handle]=2
        [test_iface_pd]=1
        [test_iface_pd_iter]=1
        [test_iface_primitive_cache]=1
        [test_iface_runtime_dims]=3
        [test_iface_sparse]=2
        [test_iface_weights_format]=2
        [test_iface_wino_convolution]=2
        [test_inner_product_backward_data]=12
        [test_inner_product_backward_weights]=18
        [test_inner_product_forward]=11
        [test_internals]=2
        [test_internals_env_vars_dnnl]=2
        [test_internals_env_vars_onednn]=2
        [test_internals_isa_compat]=2
        [test_internals_sdpa]=2
        [test_layer_normalization]=213
        [test_lrn]=1701
        [test_matmul]=4
        [test_persistent_cache_api]=2
        [test_pooling_backward]=1409
        [test_pooling_forward]=1955
        [test_prelu]=7
        [test_primitive_cache_mt]=1
        [test_reduction]=4
        [test_regression_binary_stride]=2
        [test_reorder]=23
        [test_resampling]=9
        [test_rnn_forward]=6
        [test_shuffle]=9
        [test_softmax]=26
    )

    mapfile -t weighted_tests < <(
        for test in "${filtered_tests[@]}"; do
            cost=${test_costs["${test}"]:-$default_cost}
            printf '%d\t%s\n' "${cost}" "${test}"
        done | sort -t $'\t' -k1,1nr -k2,2
    )

    declare -a group_costs=()
    declare -a group_counts=()
    declare -a group_tests=()
    declare -a selected_tests=()

    for ((group = 1; group <= total_parts; group++)); do
        group_costs[group]=0
        group_counts[group]=0
        group_tests[group]=""
    done

    for weighted_test in "${weighted_tests[@]}"; do
        IFS=$'\t' read -r cost test <<< "${weighted_test}"
        best_group=1
        best_cost=${group_costs[1]}
        best_count=${group_counts[1]}

        for ((group = 2; group <= total_parts; group++)); do
            if (( group_costs[group] < best_cost )) \
                    || (( group_costs[group] == best_cost
                            && group_counts[group] < best_count )); then
                best_group=${group}
                best_cost=${group_costs[group]}
                best_count=${group_counts[group]}
            fi
        done

        group_costs[best_group]=$((group_costs[best_group] + cost))
        group_counts[best_group]=$((group_counts[best_group] + 1))
        group_tests[best_group]+="${test}"$'\n'
    done

    mapfile -t selected_tests < <(printf '%s' "${group_tests[part]}")

    if (( ${#selected_tests[@]} == 0 )); then
        echo "Balanced RISC-V partition ${part}/${total_parts} is empty."
        exit 1
    fi

    regex=""
    for test in "${selected_tests[@]}"; do
        if [[ -n "${regex}" ]]; then
            regex+="|"
        fi
        regex+="$(regex_escape "${test}")"
    done

    echo "Using balanced RISC-V CI partition ${part}/${total_parts}: "\
            "${#selected_tests[@]} tests, estimated load ${group_costs[part]}s"
    set -x
    ctest --no-tests=error --output-on-failure --output-junit riscv-tests.xml -R "^(${regex})$" \
            -E "${skipped_tests}"
    set +x
}

# Cross-compilation mode - need QEMU
echo "Using QEMU for test execution"
export QEMU_LD_PREFIX=/usr/riscv64-linux-gnu

if [[ "$ONEDNN_TEST_SET" == "SMOKE" ]]; then
    set -x
    ctest --no-tests=error --output-on-failure --output-junit riscv-tests.xml -E $("${SCRIPT_DIR}"/skipped-tests.sh)
    set +x

elif [[ "$ONEDNN_TEST_SET" == "CI" ]]; then
    skipped_tests=$("${SCRIPT_DIR}"/skipped-tests.sh)
    start=${ONEDNN_TEST_PART:-1}
    stride=${ONEDNN_TEST_STRIDE:-1}
    partition=${ONEDNN_TEST_PARTITION:-stride}

    if [[ "${partition}" == "balanced" ]]; then
        run_balanced_ci_tests "${skipped_tests}"
    else
        set -x
        ctest --no-tests=error --output-on-failure --output-junit riscv-tests.xml -I ${start},,${stride} \
                -E "${skipped_tests}"
        set +x
    fi

else
    echo "Unknown Test Set: $ONEDNN_TEST_SET"
    exit 1
fi
