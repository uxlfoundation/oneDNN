/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "benchstone_ref.hpp"

#ifndef GEMMSTONE_WITH_MKL
#include "tools/mini_mkl.hpp"
#elif __has_include("mkl/mkl.h")
#include "mkl/mkl.h"
#else
#include "mkl.h"
#endif

namespace ref {

using namespace gemmstone;

inline char transChar(bool trans) { return trans ? 'T' : 'N'; }

void _gemm(superscalar alpha, const Matrix &A, const Matrix &B, superscalar beta, Matrix &C,
           std::array<int, 2> aqGroup, std::array<int, 2> bqGroup,
           const Matrix &ao, const Matrix &bo, const Matrix &aScale, const Matrix &bScale)
{
    if (C.trans) {
        std::swap(aqGroup[0], aqGroup[1]);
        std::swap(bqGroup[0], bqGroup[1]);
        auto Ct = C.tview();
        return gemm(alpha, B.tview(), A.tview(), beta, Ct, bqGroup, aqGroup, bo, ao, bScale, aScale);
    }

    if (A.m != C.m || B.n != C.n || A.n != B.m)
        throw std::runtime_error("Inconsistent dimensions");

    auto ST = supertype(A.T, B.T, C.T);

    bool modA = (ao.valid() || aScale.valid());
    bool modB = (bo.valid() || bScale.valid());

    auto Am = A.view(), Bm = B.view(), Cm = C.view();
    if (A.T != ST || modA) Am = A.copyAs(ST);
    if (B.T != ST || modB) Bm = B.copyAs(ST);
    if (C.T != ST) {
        Cm = Matrix(ST, C);
        if (beta != 0)
            copy(Cm, C);
    }

    if (modA) dequantize(Am, aqGroup, ao, aScale);
    if (modB) dequantize(Bm, bqGroup, bo, bScale);

    auto ta = transChar(A.trans), tb = transChar(B.trans);
    MKL_INT m = C.m, n = C.n, k = A.n;
    MKL_INT lda = A.ld, ldb = B.ld, ldc = C.ld;

    auto alpham = Matrix::scalar(ST, alpha);
    auto betam  = Matrix::scalar(ST, beta);

    switch (ST) {
        case Type::f32:
            sgemm(&ta, &tb, &m, &n, &k, alpham.data.as<float>(), Am.data.as<float>(), &lda, Bm.data.as<float>(), &ldb, betam.data.as<float>(), Cm.data.as<float>(), &ldc);
            break;
        case Type::f64:
            dgemm(&ta, &tb, &m, &n, &k, alpham.data.as<double>(), Am.data.as<double>(), &lda, Bm.data.as<double>(), &ldb, betam.data.as<double>(), Cm.data.as<double>(), &ldc);
            break;
        default: stub();
    }

    if (C.T != ST)
        copy(C, Cm);
}

void gemm(superscalar alpha, const Matrix &A, const Matrix &B, superscalar beta, Matrix &C,
          std::array<int, 2> aqGroup, std::array<int, 2> bqGroup, const Matrix &ao, const Matrix &bo, const Matrix &aScale, const Matrix &bScale)
{
    batchify(_gemm, alpha, A, B, beta, C, aqGroup, bqGroup, ao, bo, aScale, bScale);
}

} /* namespace ref */
