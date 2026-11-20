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

#include "kernel_info.hpp"

#include "benchstone_common.hpp"
#include "tools/kernel_modifiers.hpp"

namespace {
void inputPrecision(std::istream &s, char &p1, char &p2, int &components)
{
    components = 1;
    s >> p1;
    if (p1 == '[') {
        char rbracket;
        s >> p1 >> p2;
        if (s.peek() == 'x') {
            char x;
            s >> x >> components;
        }
        s >> rbracket;
        if (rbracket != ']') throw std::runtime_error("Error parsing precisions, expected ]");
    } else
        p2 = p1;
}

void inputPrecision(std::istream &s, char &p1, char &p2)
{
    int components;
    inputPrecision(s, p1, p2, components);
    if (components != 1) throw std::runtime_error("Error parsing precisions: components not supported here.");
}

void inputLayout(std::istream &s, char precision, char &layout, int &crosspack, int &tileR, int &tileC, int &packSize, int &align)
{
    crosspack = 1;
    tileR = 0;
    tileC = 0;
    align = charToType(precision).paddedSize();
    packSize = 0;
    s >> layout;
    if (std::isdigit(s.peek()))
        s >> crosspack;
    if (s.peek() == '#') {
        char eat;
        s >> eat >> tileR >> eat >> tileC;
    }
    if (s.peek() == '%') {
        char eat;
        s >> eat >> packSize;
    }
    if (s.peek() == '@') {
        char eat;
        s >> eat >> align;
    }
}

double inputRealScalar(std::istream &s)
{
    bool negative = false;

    s >> std::ws;
    if (s.peek() == 'n') {
        std::string nan;
        s >> nan;
        return qnan<double>();
    } else if (s.peek() == '-') {
        char eat;
        s >> eat;
        if (s.peek() == ' ' || s.eof())
            return qnan<double>();
        negative = true;
    }

    double scalar;
    s >> scalar;
    if (negative)
        scalar *= -1;
    return scalar;
}

void processProblemInfo(Level3KernelInfo &info)
{
    // Interpret problem string.
    info.strongReadCheck = false;
    info.aqGroup.fill(0);
    info.bqGroup.fill(0);
    info.systolic = false;

    std::stringstream ss(info.problem);
    std::string word;

    utils::KernelModifiers modifiers;
    for (;;) {
        if (!(ss >> word))
            throw std::runtime_error("Missing kernel type in problem: " + info.problem);
        if (!modifiers.parse(word)) break;
    }
    info.readCheck = modifiers.readCheck;
    info.batch = modifiers.batch == gemmstone::BatchMode::Strided ? 's'
               : modifiers.batch == gemmstone::BatchMode::Nonstrided ? 'n'
               : modifiers.batch == gemmstone::BatchMode::Variable ? 'v' : 0;
    auto offsetKind = [](const utils::QuantizationModifier &offset) -> char {
        return offset.enabled ? "spvg"[offset.ptrDims + 1] : 0;
    };
    auto precision = [](const utils::QuantizationModifier &value) {
        return gemmstone::precisionChar(value.T == Type::invalid ? Type(Type::s32) : value.T);
    };
    info.aOffset = offsetKind(modifiers.aOffset);
    info.bOffset = offsetKind(modifiers.bOffset);
    info.cOffset = modifiers.cOffset.enabled;
    info.cBias = modifiers.bias;
    info.sumA = modifiers.sumA;
    info.sumB = modifiers.sumB;
    info.scaleA = modifiers.aScale.enabled;
    info.scaleB = modifiers.bScale.enabled;
    info.precisionAO = precision(modifiers.aOffset);
    info.precisionBO = precision(modifiers.bOffset);
    info.precisionAScale = precision(modifiers.aScale);
    info.precisionBScale = precision(modifiers.bScale);
    if (modifiers.aOffset.grouped()) info.aqGroup = modifiers.aOffset.groupSize;
    if (modifiers.bOffset.grouped()) info.bqGroup = modifiers.bOffset.groupSize;
    if (modifiers.aScale.grouped()) info.aqGroup = modifiers.aScale.groupSize;
    if (modifiers.bScale.grouped()) info.bqGroup = modifiers.bScale.groupSize;

    inputPrecision(ss, info.precisionA, info.precisionAInt, info.A_components);
    inputPrecision(ss, info.precisionB, info.precisionBInt, info.B_components);
    inputPrecision(ss, info.precisionCInt, info.precisionC);
    if (isalpha(ss.peek()))
        ss >> info.precisionS;
    else
        info.precisionS = info.precisionCInt;

    inputLayout(ss, info.precisionA, info.layoutA, info.crosspackA, info.A_tileR, info.A_tileC, info.packSizeA, info.alignA);
    inputLayout(ss, info.precisionB, info.layoutB, info.crosspackB, info.B_tileR, info.B_tileC, info.packSizeB, info.alignB);
    inputLayout(ss, info.precisionC, info.layoutC, info.crosspackC, info.C_tileR, info.C_tileC, info.packSizeC, info.alignC);

    if (info.packSizeA == 0) info.packSizeA = info.unroll[LoopM];
    if (info.packSizeB == 0) info.packSizeB = info.unroll[LoopN];

    info.alpha = inputRealScalar(ss);
    info.beta  = inputRealScalar(ss);

    std::stringstream ssExtra(info.extra);
    while (!ssExtra.eof()) {
        ssExtra >> std::ws >> word;
        if (word == "sys")
            info.systolic = true;
        else if (word == "track")
            info.tracking = true;
        else if (word == "src")
            info.strongReadCheck = true;
        else if (word == "alocal")
            info.localA = true;
        else if (word == "blocal")
            info.localB = true;
        else if (word == "clocal")
            info.localC = true;
        else if (word == "rx")
            info.relaxedAccumulation = true;
        else if (word == "a2d")
            info.a2D = true;
        else if (word == "b2d")
            info.b2D = true;
        else if (word == "c2d")
            info.c2D = true;
    }

    if (info.needsAGroupSums() && info.aqGroup[1] == 0)
        info.aqGroup = {1, info.bqGroup[0]};
    if (info.needsBGroupSums() && info.bqGroup[0] == 0)
        info.bqGroup = {info.aqGroup[1], 1};
}

} // anonymous namespace

KernelInfo::KernelInfo(const std::string& info_str) {
    std::stringstream s(info_str);
    while (!s.eof()) {
        char comma;
        s >> std::ws;
        if (s.peek() == ',')
            s >> comma >> std::ws;

        auto dinfo = gemmstone::CommonDriverInfo::undump(s);

        l3.resize(l3.size() + 1);
        auto &last = l3[l3.size() - 1];
        static_cast<gemmstone::CommonDriverInfo &>(last) = dinfo;
        processProblemInfo(last);

        s >> std::ws;
    }

}
