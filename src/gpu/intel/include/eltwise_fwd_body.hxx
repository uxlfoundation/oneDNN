/*******************************************************************************
* Copyright 2026 Intel Corporation
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

ELT_T __attribute__((overloadable)) relu_fwd(ELT_T s, float alpha) {
    return s > (ELT_T)0 ? s : s * alpha;
}

ELT_T __attribute__((overloadable)) linear_fwd(
        ELT_T s, float alpha, float beta) {
    return alpha * s + beta;
}

ELT_T __attribute__((overloadable)) soft_relu_fwd(ELT_T s, float alpha) {
    s = alpha * s;
    return (s < (ELT_T)log(CONVERT_POST_OP_DATA_T(DATA_MAX)) ? log1p(exp(s))
                                                             : s)
            / alpha;
}

ELT_T __attribute__((overloadable)) logistic_fwd(ELT_T s) {
    return (ELT_T)POST_OP_LITERAL(1.) / (POST_OP_LITERAL(1.) + exp(-s));
}

ELT_T __attribute__((overloadable)) square_fwd(ELT_T s) {
    return s * s;
}

ELT_T __attribute__((overloadable)) sqrt_fwd(ELT_T s) {
    return sqrt(s);
}

ELT_T __attribute__((overloadable)) abs_fwd(ELT_T s) {
    return s > (ELT_T)0 ? s : -s;
}

ELT_T __attribute__((overloadable)) tanh_fwd(ELT_T s) {
    return tanh(s);
}

ELT_T __attribute__((overloadable)) mish_fwd(ELT_T s) {
    return s * tanh_fwd(soft_relu_fwd(s, POST_OP_LITERAL(1.)));
}

ELT_T __attribute__((overloadable)) elu_fwd(ELT_T s, float alpha) {
    return s > (ELT_T)0 ? s : alpha * expm1(s);
}

ELT_T __attribute__((overloadable)) exp_fwd(ELT_T s) {
    return exp(s);
}

ELT_T __attribute__((overloadable)) gelu_tanh_fwd(ELT_T s) {
    const ELT_T g = POST_OP_LITERAL(1.5957691669464111328125) * s
            * (POST_OP_LITERAL(1.) + POST_OP_LITERAL(0.044715) * s * s);
    return s / (POST_OP_LITERAL(1.) + exp(-g));
}

ELT_T __attribute__((overloadable)) swish_fwd(ELT_T s, float alpha) {
    return s / (POST_OP_LITERAL(1.) + exp(-alpha * s));
}

ELT_T __attribute__((overloadable)) log_fwd(ELT_T s) {
    return log(s);
}

ELT_T __attribute__((overloadable)) clip_fwd(ELT_T s, float alpha, float beta) {
    return fmin(fmax(s, (ELT_T)alpha), (ELT_T)beta);
}

ELT_T __attribute__((overloadable)) clip_v2_fwd(
        ELT_T s, float alpha, float beta) {
    return clip_fwd(s, alpha, beta);
}

ELT_T __attribute__((overloadable)) pow_fwd(ELT_T s, float alpha, float beta) {
    return alpha * pow(s, (ELT_T)beta);
}

ELT_T __attribute__((overloadable)) gelu_erf_fwd(ELT_T s) {
    return POST_OP_LITERAL(0.5) * s
            * (POST_OP_LITERAL(1.)
                    + erf(s * POST_OP_LITERAL(0.707106769084930419921875)));
}

ELT_T __attribute__((overloadable)) round_fwd(ELT_T s) {
    return rint(s);
}

ELT_T __attribute__((overloadable)) hardsigmoid_fwd(
        ELT_T s, float alpha, float beta) {
    const ELT_T v = linear_fwd(s, alpha, beta);
    return isnan(v)
            ? v
            : clamp(v, (ELT_T)POST_OP_LITERAL(0.), (ELT_T)POST_OP_LITERAL(1.));
}

ELT_T __attribute__((overloadable)) hardswish_fwd(
        ELT_T s, float alpha, float beta) {
    return s * hardsigmoid_fwd(s, alpha, beta);
}

ELT_T __attribute__((overloadable)) fwd_eltwise_common(
        int eltwise_alg, ELT_T x, float alpha_, float beta_, float scale_) {
    switch (eltwise_alg) {
        case eltwise_relu: return scale_ * relu_fwd(x, alpha_);
        case eltwise_linear: return scale_ * linear_fwd(x, alpha_, beta_);
        case eltwise_soft_relu: return scale_ * soft_relu_fwd(x, alpha_);
        case eltwise_mish: return scale_ * mish_fwd(x);
        case eltwise_logistic: return scale_ * logistic_fwd(x);
        case eltwise_tanh: return scale_ * tanh_fwd(x);
        case eltwise_elu: return scale_ * elu_fwd(x, alpha_);
        case eltwise_square: return scale_ * square_fwd(x);
        case eltwise_sqrt: return scale_ * sqrt_fwd(x);
        case eltwise_abs: return scale_ * abs_fwd(x);
        case eltwise_exp: return scale_ * exp_fwd(x);
        case eltwise_gelu_tanh: return scale_ * gelu_tanh_fwd(x);
        case eltwise_swish: return scale_ * swish_fwd(x, alpha_);
        case eltwise_log: return scale_ * log_fwd(x);
        case eltwise_clip: return scale_ * clip_fwd(x, alpha_, beta_);
        case eltwise_clip_v2: return scale_ * clip_v2_fwd(x, alpha_, beta_);
        case eltwise_pow: return scale_ * pow_fwd(x, alpha_, beta_);
        case eltwise_gelu_erf: return scale_ * gelu_erf_fwd(x);
        case eltwise_round: return scale_ * round_fwd(x);
        case eltwise_hardswish: return scale_ * hardswish_fwd(x, alpha_, beta_);
        case eltwise_hardsigmoid:
            return scale_ * hardsigmoid_fwd(x, alpha_, beta_);
        case eltwise_relu_dst: return scale_ * relu_fwd(x, alpha_);
        case eltwise_logistic_dst: return scale_ * logistic_fwd(x);
        case eltwise_tanh_dst: return scale_ * tanh_fwd(x);
        case eltwise_elu_dst: return scale_ * elu_fwd(x, alpha_);
        case eltwise_sqrt_dst: return scale_ * sqrt_fwd(x);
        case eltwise_exp_dst: return scale_ * exp_fwd(x);
        case eltwise_clip_v2_dst: return scale_ * clip_v2_fwd(x, alpha_, beta_);
        default: return x;
    }
}
