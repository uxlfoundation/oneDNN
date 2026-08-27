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

ELT_T __attribute__((overloadable)) relu_bwd(ELT_T dd, ELT_T s, float alpha) {
    return s > 0 ? dd : dd * alpha;
}
ELT_T __attribute__((overloadable)) relu_bwd_use_dst(
        ELT_T dd, ELT_T d, float alpha) {
    return d > 0 ? dd : dd * alpha;
}

ELT_T __attribute__((overloadable)) linear_bwd(ELT_T dd, float alpha) {
    return dd * alpha;
}

ELT_T __attribute__((overloadable)) soft_relu_bwd(
        ELT_T dd, ELT_T s, float alpha) {
    s = alpha * s;
    return dd / (POST_OP_LITERAL(1.) + exp(-s));
}

ELT_T __attribute__((overloadable)) logistic_bwd(ELT_T dd, ELT_T s) {
    ELT_T v = logistic_fwd(s);
    return dd * v * (POST_OP_LITERAL(1.) - v);
}
ELT_T __attribute__((overloadable)) logistic_bwd_use_dst(ELT_T dd, ELT_T d) {
    return dd * d * (POST_OP_LITERAL(1.) - d);
}

ELT_T __attribute__((overloadable)) square_bwd(ELT_T dd, ELT_T s) {
    return dd * 2 * s;
}

ELT_T __attribute__((overloadable)) sqrt_bwd(ELT_T dd, ELT_T s) {
    return dd / (POST_OP_LITERAL(2.) * sqrt(s));
}
ELT_T __attribute__((overloadable)) sqrt_bwd_use_dst(ELT_T dd, ELT_T d) {
    return dd / (POST_OP_LITERAL(2.) * d);
}

ELT_T __attribute__((overloadable)) abs_bwd(ELT_T dd, ELT_T s) {
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

ELT_T __attribute__((overloadable)) tanh_bwd(ELT_T dd, ELT_T s) {
    ELT_T e = tanh_fwd(s);
    return dd * (POST_OP_LITERAL(1.) - e) * (POST_OP_LITERAL(1.) + e);
}
ELT_T __attribute__((overloadable)) tanh_bwd_use_dst(ELT_T dd, ELT_T d) {
    return dd * (POST_OP_LITERAL(1.) - d) * (POST_OP_LITERAL(1.) + d);
}

ELT_T __attribute__((overloadable)) mish_bwd(ELT_T dd, ELT_T s) {
    const ELT_T tanh = tanh_fwd(soft_relu_fwd(s, POST_OP_LITERAL(1.)));
    const ELT_T srelu_bwd
            = soft_relu_bwd(POST_OP_LITERAL(1.), s, POST_OP_LITERAL(1.));
    const ELT_T derivative = tanh
            + s * srelu_bwd
                    * (POST_OP_LITERAL(1.) - pow(tanh, POST_OP_LITERAL(2.)));
    return dd * derivative;
}

ELT_T __attribute__((overloadable)) elu_bwd(ELT_T dd, ELT_T s, float alpha) {
    return dd * (s > 0 ? 1 : alpha * exp(s));
}
ELT_T __attribute__((overloadable)) elu_bwd_use_dst(
        ELT_T dd, ELT_T d, float alpha) {
    return dd * (d > 0 ? 1 : d + alpha);
}

ELT_T __attribute__((overloadable)) exp_bwd(ELT_T dd, ELT_T s) {
    return dd * exp_fwd(s);
}
ELT_T __attribute__((overloadable)) exp_bwd_use_dst(ELT_T dd, ELT_T d) {
    return dd * d;
}

ELT_T __attribute__((overloadable)) gelu_tanh_bwd(ELT_T dd, ELT_T s) {
    const ELT_T sqrt_2_over_pi = POST_OP_LITERAL(0.79788458347320556640625);
    const ELT_T fitting_const = POST_OP_LITERAL(0.044715);
    const ELT_T g = sqrt_2_over_pi * s
            * (POST_OP_LITERAL(1.) + fitting_const * s * s);
    const ELT_T dg = sqrt_2_over_pi
            * (POST_OP_LITERAL(1.)
                    + POST_OP_LITERAL(3.) * fitting_const * s * s);
    const ELT_T v = tanh_fwd(g);
    return dd * POST_OP_LITERAL(0.5) * (POST_OP_LITERAL(1.) + v)
            * (POST_OP_LITERAL(1.) + s * (POST_OP_LITERAL(1.) - v) * dg);
}

ELT_T __attribute__((overloadable)) swish_bwd(ELT_T dd, ELT_T s, float alpha) {
    ELT_T v = logistic_fwd(alpha * s);
    return dd * (v + s * alpha * v * (POST_OP_LITERAL(1.) - v));
}

ELT_T __attribute__((overloadable)) log_bwd(ELT_T dd, ELT_T s) {
    return dd / s;
}

ELT_T __attribute__((overloadable)) clip_bwd(
        ELT_T dd, ELT_T s, float alpha, float beta) {
    return dd * (alpha < s && s <= beta ? 1 : 0);
}

ELT_T __attribute__((overloadable)) clip_v2_bwd(
        ELT_T dd, ELT_T s, float alpha, float beta) {
    return dd * (alpha < s && s < beta ? 1 : 0);
}
ELT_T __attribute__((overloadable)) clip_v2_bwd_use_dst(
        ELT_T dd, ELT_T d, float alpha, float beta) {
    return dd * (alpha < d && d < beta ? 1 : 0);
}

ELT_T __attribute__((overloadable)) pow_bwd(
        ELT_T dd, ELT_T s, float alpha, float beta) {
    if (beta == 0) return 0;

    ELT_T v = pow_fwd(s, alpha * beta, beta - 1);
    return dd * v;
}

ELT_T __attribute__((overloadable)) gelu_erf_bwd(ELT_T dd, ELT_T s) {
    const ELT_T two_over_sqrt_pi = POST_OP_LITERAL(1.12837922573089599609375);
    const ELT_T sqrt_2_over_2 = POST_OP_LITERAL(0.707106769084930419921875);
    ELT_T v = s * sqrt_2_over_2;
    return dd * POST_OP_LITERAL(0.5)
            * (POST_OP_LITERAL(1.) + erf(v)
                    + v * two_over_sqrt_pi * exp(-v * v));
}

ELT_T __attribute__((overloadable)) hardsigmoid_bwd(
        ELT_T dd, ELT_T s, float alpha, float beta) {
    ELT_T v = alpha * s + beta;
    return v <= 0.f ? 0.f : v >= 1.f ? 0.f : dd * alpha;
}

ELT_T __attribute__((overloadable)) hardswish_bwd(
        ELT_T dd, ELT_T s, float alpha, float beta) {
    ELT_T v = alpha * s + beta;
    ELT_T w = POST_OP_LITERAL(2.) * alpha * s + beta;
    return (v <= 0.f ? 0.f : v >= 1.f ? dd : dd * w);
}
