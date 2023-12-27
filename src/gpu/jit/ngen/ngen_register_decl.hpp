/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

/*
 * When compiling nGEN in C++11 or C++14 mode, this header file should be
 *  #include'd exactly once in your source code.
 */

#if (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS)

#include "ngen.hpp"

#define NGEN_REGISTER_DECL_MAIN(CG, PREFIX) \
PREFIX constexpr ngen::IndirectRegisterFrame CG::indirect; \
\
PREFIX constexpr ngen::GRF CG::r0; \
PREFIX constexpr ngen::GRF CG::r1; \
PREFIX constexpr ngen::GRF CG::r2; \
PREFIX constexpr ngen::GRF CG::r3; \
PREFIX constexpr ngen::GRF CG::r4; \
PREFIX constexpr ngen::GRF CG::r5; \
PREFIX constexpr ngen::GRF CG::r6; \
PREFIX constexpr ngen::GRF CG::r7; \
PREFIX constexpr ngen::GRF CG::r8; \
PREFIX constexpr ngen::GRF CG::r9; \
PREFIX constexpr ngen::GRF CG::r10; \
PREFIX constexpr ngen::GRF CG::r11; \
PREFIX constexpr ngen::GRF CG::r12; \
PREFIX constexpr ngen::GRF CG::r13; \
PREFIX constexpr ngen::GRF CG::r14; \
PREFIX constexpr ngen::GRF CG::r15; \
PREFIX constexpr ngen::GRF CG::r16; \
PREFIX constexpr ngen::GRF CG::r17; \
PREFIX constexpr ngen::GRF CG::r18; \
PREFIX constexpr ngen::GRF CG::r19; \
PREFIX constexpr ngen::GRF CG::r20; \
PREFIX constexpr ngen::GRF CG::r21; \
PREFIX constexpr ngen::GRF CG::r22; \
PREFIX constexpr ngen::GRF CG::r23; \
PREFIX constexpr ngen::GRF CG::r24; \
PREFIX constexpr ngen::GRF CG::r25; \
PREFIX constexpr ngen::GRF CG::r26; \
PREFIX constexpr ngen::GRF CG::r27; \
PREFIX constexpr ngen::GRF CG::r28; \
PREFIX constexpr ngen::GRF CG::r29; \
PREFIX constexpr ngen::GRF CG::r30; \
PREFIX constexpr ngen::GRF CG::r31; \
PREFIX constexpr ngen::GRF CG::r32; \
PREFIX constexpr ngen::GRF CG::r33; \
PREFIX constexpr ngen::GRF CG::r34; \
PREFIX constexpr ngen::GRF CG::r35; \
PREFIX constexpr ngen::GRF CG::r36; \
PREFIX constexpr ngen::GRF CG::r37; \
PREFIX constexpr ngen::GRF CG::r38; \
PREFIX constexpr ngen::GRF CG::r39; \
PREFIX constexpr ngen::GRF CG::r40; \
PREFIX constexpr ngen::GRF CG::r41; \
PREFIX constexpr ngen::GRF CG::r42; \
PREFIX constexpr ngen::GRF CG::r43; \
PREFIX constexpr ngen::GRF CG::r44; \
PREFIX constexpr ngen::GRF CG::r45; \
PREFIX constexpr ngen::GRF CG::r46; \
PREFIX constexpr ngen::GRF CG::r47; \
PREFIX constexpr ngen::GRF CG::r48; \
PREFIX constexpr ngen::GRF CG::r49; \
PREFIX constexpr ngen::GRF CG::r50; \
PREFIX constexpr ngen::GRF CG::r51; \
PREFIX constexpr ngen::GRF CG::r52; \
PREFIX constexpr ngen::GRF CG::r53; \
PREFIX constexpr ngen::GRF CG::r54; \
PREFIX constexpr ngen::GRF CG::r55; \
PREFIX constexpr ngen::GRF CG::r56; \
PREFIX constexpr ngen::GRF CG::r57; \
PREFIX constexpr ngen::GRF CG::r58; \
PREFIX constexpr ngen::GRF CG::r59; \
PREFIX constexpr ngen::GRF CG::r60; \
PREFIX constexpr ngen::GRF CG::r61; \
PREFIX constexpr ngen::GRF CG::r62; \
PREFIX constexpr ngen::GRF CG::r63; \
PREFIX constexpr ngen::GRF CG::r64; \
PREFIX constexpr ngen::GRF CG::r65; \
PREFIX constexpr ngen::GRF CG::r66; \
PREFIX constexpr ngen::GRF CG::r67; \
PREFIX constexpr ngen::GRF CG::r68; \
PREFIX constexpr ngen::GRF CG::r69; \
PREFIX constexpr ngen::GRF CG::r70; \
PREFIX constexpr ngen::GRF CG::r71; \
PREFIX constexpr ngen::GRF CG::r72; \
PREFIX constexpr ngen::GRF CG::r73; \
PREFIX constexpr ngen::GRF CG::r74; \
PREFIX constexpr ngen::GRF CG::r75; \
PREFIX constexpr ngen::GRF CG::r76; \
PREFIX constexpr ngen::GRF CG::r77; \
PREFIX constexpr ngen::GRF CG::r78; \
PREFIX constexpr ngen::GRF CG::r79; \
PREFIX constexpr ngen::GRF CG::r80; \
PREFIX constexpr ngen::GRF CG::r81; \
PREFIX constexpr ngen::GRF CG::r82; \
PREFIX constexpr ngen::GRF CG::r83; \
PREFIX constexpr ngen::GRF CG::r84; \
PREFIX constexpr ngen::GRF CG::r85; \
PREFIX constexpr ngen::GRF CG::r86; \
PREFIX constexpr ngen::GRF CG::r87; \
PREFIX constexpr ngen::GRF CG::r88; \
PREFIX constexpr ngen::GRF CG::r89; \
PREFIX constexpr ngen::GRF CG::r90; \
PREFIX constexpr ngen::GRF CG::r91; \
PREFIX constexpr ngen::GRF CG::r92; \
PREFIX constexpr ngen::GRF CG::r93; \
PREFIX constexpr ngen::GRF CG::r94; \
PREFIX constexpr ngen::GRF CG::r95; \
PREFIX constexpr ngen::GRF CG::r96; \
PREFIX constexpr ngen::GRF CG::r97; \
PREFIX constexpr ngen::GRF CG::r98; \
PREFIX constexpr ngen::GRF CG::r99; \
PREFIX constexpr ngen::GRF CG::r100; \
PREFIX constexpr ngen::GRF CG::r101; \
PREFIX constexpr ngen::GRF CG::r102; \
PREFIX constexpr ngen::GRF CG::r103; \
PREFIX constexpr ngen::GRF CG::r104; \
PREFIX constexpr ngen::GRF CG::r105; \
PREFIX constexpr ngen::GRF CG::r106; \
PREFIX constexpr ngen::GRF CG::r107; \
PREFIX constexpr ngen::GRF CG::r108; \
PREFIX constexpr ngen::GRF CG::r109; \
PREFIX constexpr ngen::GRF CG::r110; \
PREFIX constexpr ngen::GRF CG::r111; \
PREFIX constexpr ngen::GRF CG::r112; \
PREFIX constexpr ngen::GRF CG::r113; \
PREFIX constexpr ngen::GRF CG::r114; \
PREFIX constexpr ngen::GRF CG::r115; \
PREFIX constexpr ngen::GRF CG::r116; \
PREFIX constexpr ngen::GRF CG::r117; \
PREFIX constexpr ngen::GRF CG::r118; \
PREFIX constexpr ngen::GRF CG::r119; \
PREFIX constexpr ngen::GRF CG::r120; \
PREFIX constexpr ngen::GRF CG::r121; \
PREFIX constexpr ngen::GRF CG::r122; \
PREFIX constexpr ngen::GRF CG::r123; \
PREFIX constexpr ngen::GRF CG::r124; \
PREFIX constexpr ngen::GRF CG::r125; \
PREFIX constexpr ngen::GRF CG::r126; \
PREFIX constexpr ngen::GRF CG::r127; \
PREFIX constexpr ngen::GRF CG::r128; \
PREFIX constexpr ngen::GRF CG::r129; \
PREFIX constexpr ngen::GRF CG::r130; \
PREFIX constexpr ngen::GRF CG::r131; \
PREFIX constexpr ngen::GRF CG::r132; \
PREFIX constexpr ngen::GRF CG::r133; \
PREFIX constexpr ngen::GRF CG::r134; \
PREFIX constexpr ngen::GRF CG::r135; \
PREFIX constexpr ngen::GRF CG::r136; \
PREFIX constexpr ngen::GRF CG::r137; \
PREFIX constexpr ngen::GRF CG::r138; \
PREFIX constexpr ngen::GRF CG::r139; \
PREFIX constexpr ngen::GRF CG::r140; \
PREFIX constexpr ngen::GRF CG::r141; \
PREFIX constexpr ngen::GRF CG::r142; \
PREFIX constexpr ngen::GRF CG::r143; \
PREFIX constexpr ngen::GRF CG::r144; \
PREFIX constexpr ngen::GRF CG::r145; \
PREFIX constexpr ngen::GRF CG::r146; \
PREFIX constexpr ngen::GRF CG::r147; \
PREFIX constexpr ngen::GRF CG::r148; \
PREFIX constexpr ngen::GRF CG::r149; \
PREFIX constexpr ngen::GRF CG::r150; \
PREFIX constexpr ngen::GRF CG::r151; \
PREFIX constexpr ngen::GRF CG::r152; \
PREFIX constexpr ngen::GRF CG::r153; \
PREFIX constexpr ngen::GRF CG::r154; \
PREFIX constexpr ngen::GRF CG::r155; \
PREFIX constexpr ngen::GRF CG::r156; \
PREFIX constexpr ngen::GRF CG::r157; \
PREFIX constexpr ngen::GRF CG::r158; \
PREFIX constexpr ngen::GRF CG::r159; \
PREFIX constexpr ngen::GRF CG::r160; \
PREFIX constexpr ngen::GRF CG::r161; \
PREFIX constexpr ngen::GRF CG::r162; \
PREFIX constexpr ngen::GRF CG::r163; \
PREFIX constexpr ngen::GRF CG::r164; \
PREFIX constexpr ngen::GRF CG::r165; \
PREFIX constexpr ngen::GRF CG::r166; \
PREFIX constexpr ngen::GRF CG::r167; \
PREFIX constexpr ngen::GRF CG::r168; \
PREFIX constexpr ngen::GRF CG::r169; \
PREFIX constexpr ngen::GRF CG::r170; \
PREFIX constexpr ngen::GRF CG::r171; \
PREFIX constexpr ngen::GRF CG::r172; \
PREFIX constexpr ngen::GRF CG::r173; \
PREFIX constexpr ngen::GRF CG::r174; \
PREFIX constexpr ngen::GRF CG::r175; \
PREFIX constexpr ngen::GRF CG::r176; \
PREFIX constexpr ngen::GRF CG::r177; \
PREFIX constexpr ngen::GRF CG::r178; \
PREFIX constexpr ngen::GRF CG::r179; \
PREFIX constexpr ngen::GRF CG::r180; \
PREFIX constexpr ngen::GRF CG::r181; \
PREFIX constexpr ngen::GRF CG::r182; \
PREFIX constexpr ngen::GRF CG::r183; \
PREFIX constexpr ngen::GRF CG::r184; \
PREFIX constexpr ngen::GRF CG::r185; \
PREFIX constexpr ngen::GRF CG::r186; \
PREFIX constexpr ngen::GRF CG::r187; \
PREFIX constexpr ngen::GRF CG::r188; \
PREFIX constexpr ngen::GRF CG::r189; \
PREFIX constexpr ngen::GRF CG::r190; \
PREFIX constexpr ngen::GRF CG::r191; \
PREFIX constexpr ngen::GRF CG::r192; \
PREFIX constexpr ngen::GRF CG::r193; \
PREFIX constexpr ngen::GRF CG::r194; \
PREFIX constexpr ngen::GRF CG::r195; \
PREFIX constexpr ngen::GRF CG::r196; \
PREFIX constexpr ngen::GRF CG::r197; \
PREFIX constexpr ngen::GRF CG::r198; \
PREFIX constexpr ngen::GRF CG::r199; \
PREFIX constexpr ngen::GRF CG::r200; \
PREFIX constexpr ngen::GRF CG::r201; \
PREFIX constexpr ngen::GRF CG::r202; \
PREFIX constexpr ngen::GRF CG::r203; \
PREFIX constexpr ngen::GRF CG::r204; \
PREFIX constexpr ngen::GRF CG::r205; \
PREFIX constexpr ngen::GRF CG::r206; \
PREFIX constexpr ngen::GRF CG::r207; \
PREFIX constexpr ngen::GRF CG::r208; \
PREFIX constexpr ngen::GRF CG::r209; \
PREFIX constexpr ngen::GRF CG::r210; \
PREFIX constexpr ngen::GRF CG::r211; \
PREFIX constexpr ngen::GRF CG::r212; \
PREFIX constexpr ngen::GRF CG::r213; \
PREFIX constexpr ngen::GRF CG::r214; \
PREFIX constexpr ngen::GRF CG::r215; \
PREFIX constexpr ngen::GRF CG::r216; \
PREFIX constexpr ngen::GRF CG::r217; \
PREFIX constexpr ngen::GRF CG::r218; \
PREFIX constexpr ngen::GRF CG::r219; \
PREFIX constexpr ngen::GRF CG::r220; \
PREFIX constexpr ngen::GRF CG::r221; \
PREFIX constexpr ngen::GRF CG::r222; \
PREFIX constexpr ngen::GRF CG::r223; \
PREFIX constexpr ngen::GRF CG::r224; \
PREFIX constexpr ngen::GRF CG::r225; \
PREFIX constexpr ngen::GRF CG::r226; \
PREFIX constexpr ngen::GRF CG::r227; \
PREFIX constexpr ngen::GRF CG::r228; \
PREFIX constexpr ngen::GRF CG::r229; \
PREFIX constexpr ngen::GRF CG::r230; \
PREFIX constexpr ngen::GRF CG::r231; \
PREFIX constexpr ngen::GRF CG::r232; \
PREFIX constexpr ngen::GRF CG::r233; \
PREFIX constexpr ngen::GRF CG::r234; \
PREFIX constexpr ngen::GRF CG::r235; \
PREFIX constexpr ngen::GRF CG::r236; \
PREFIX constexpr ngen::GRF CG::r237; \
PREFIX constexpr ngen::GRF CG::r238; \
PREFIX constexpr ngen::GRF CG::r239; \
PREFIX constexpr ngen::GRF CG::r240; \
PREFIX constexpr ngen::GRF CG::r241; \
PREFIX constexpr ngen::GRF CG::r242; \
PREFIX constexpr ngen::GRF CG::r243; \
PREFIX constexpr ngen::GRF CG::r244; \
PREFIX constexpr ngen::GRF CG::r245; \
PREFIX constexpr ngen::GRF CG::r246; \
PREFIX constexpr ngen::GRF CG::r247; \
PREFIX constexpr ngen::GRF CG::r248; \
PREFIX constexpr ngen::GRF CG::r249; \
PREFIX constexpr ngen::GRF CG::r250; \
PREFIX constexpr ngen::GRF CG::r251; \
PREFIX constexpr ngen::GRF CG::r252; \
PREFIX constexpr ngen::GRF CG::r253; \
PREFIX constexpr ngen::GRF CG::r254; \
PREFIX constexpr ngen::GRF CG::r255; \
\
PREFIX constexpr ngen::NullRegister CG::null; \
PREFIX constexpr ngen::AddressRegister CG::a0; \
PREFIX constexpr ngen::AccumulatorRegister CG::acc0; \
PREFIX constexpr ngen::AccumulatorRegister CG::acc1; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc2; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc3; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc4; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc5; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc6; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc7; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc8; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::acc9; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme0; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme1; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme2; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme3; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme4; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme5; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme6; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::mme7; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::nomme; \
PREFIX constexpr ngen::SpecialAccumulatorRegister CG::noacc; \
PREFIX constexpr ngen::FlagRegister CG::f0; \
PREFIX constexpr ngen::FlagRegister CG::f1; \
PREFIX constexpr ngen::FlagRegister CG::f2; \
PREFIX constexpr ngen::FlagRegister CG::f3; \
PREFIX constexpr ngen::FlagRegister CG::f0_0; \
PREFIX constexpr ngen::FlagRegister CG::f0_1; \
PREFIX constexpr ngen::FlagRegister CG::f1_0; \
PREFIX constexpr ngen::FlagRegister CG::f1_1; \
PREFIX constexpr ngen::ChannelEnableRegister CG::ce0; \
PREFIX constexpr ngen::StackPointerRegister CG::sp; \
PREFIX constexpr ngen::StateRegister CG::sr0; \
PREFIX constexpr ngen::StateRegister CG::sr1; \
PREFIX constexpr ngen::ControlRegister CG::cr0; \
PREFIX constexpr ngen::NotificationRegister CG::n0; \
PREFIX constexpr ngen::InstructionPointerRegister CG::ip; \
PREFIX constexpr ngen::ThreadDependencyRegister CG::tdr0; \
PREFIX constexpr ngen::PerformanceRegister CG::tm0; \
PREFIX constexpr ngen::PerformanceRegister CG::tm1; \
PREFIX constexpr ngen::PerformanceRegister CG::tm2; \
PREFIX constexpr ngen::PerformanceRegister CG::tm3; \
PREFIX constexpr ngen::PerformanceRegister CG::tm4; \
PREFIX constexpr ngen::PerformanceRegister CG::pm0; \
PREFIX constexpr ngen::PerformanceRegister CG::tp0; \
PREFIX constexpr ngen::DebugRegister CG::dbg0; \
PREFIX constexpr ngen::FlowControlRegister CG::fc0; \
PREFIX constexpr ngen::FlowControlRegister CG::fc1; \
PREFIX constexpr ngen::FlowControlRegister CG::fc2; \
PREFIX constexpr ngen::FlowControlRegister CG::fc3; \
\
PREFIX constexpr ngen::InstructionModifier CG::NoDDClr; \
PREFIX constexpr ngen::InstructionModifier CG::NoDDChk; \
PREFIX constexpr ngen::InstructionModifier CG::AccWrEn; \
PREFIX constexpr ngen::InstructionModifier CG::NoSrcDepSet; \
PREFIX constexpr ngen::InstructionModifier CG::Breakpoint; \
PREFIX constexpr ngen::InstructionModifier CG::sat; \
PREFIX constexpr ngen::InstructionModifier CG::NoMask; \
PREFIX constexpr ngen::InstructionModifier CG::ExBSO; \
PREFIX constexpr ngen::InstructionModifier CG::AutoSWSB; \
PREFIX constexpr ngen::InstructionModifier CG::Serialize; \
PREFIX constexpr ngen::InstructionModifier CG::EOT; \
PREFIX constexpr ngen::InstructionModifier CG::Align1; \
PREFIX constexpr ngen::InstructionModifier CG::Align16; \
PREFIX constexpr ngen::InstructionModifier CG::Atomic; \
PREFIX constexpr ngen::InstructionModifier CG::Switch; \
PREFIX constexpr ngen::InstructionModifier CG::NoPreempt; \
\
PREFIX constexpr ngen::PredCtrl CG::anyv; \
PREFIX constexpr ngen::PredCtrl CG::allv; \
PREFIX constexpr ngen::PredCtrl CG::any2h; \
PREFIX constexpr ngen::PredCtrl CG::all2h; \
PREFIX constexpr ngen::PredCtrl CG::any4h; \
PREFIX constexpr ngen::PredCtrl CG::all4h; \
PREFIX constexpr ngen::PredCtrl CG::any8h; \
PREFIX constexpr ngen::PredCtrl CG::all8h; \
PREFIX constexpr ngen::PredCtrl CG::any16h; \
PREFIX constexpr ngen::PredCtrl CG::all16h; \
PREFIX constexpr ngen::PredCtrl CG::any32h; \
PREFIX constexpr ngen::PredCtrl CG::all32h; \
PREFIX constexpr ngen::PredCtrl CG::any; \
PREFIX constexpr ngen::PredCtrl CG::all; \
\
PREFIX constexpr ngen::InstructionModifier CG::x_repl; \
PREFIX constexpr ngen::InstructionModifier CG::y_repl; \
PREFIX constexpr ngen::InstructionModifier CG::z_repl; \
PREFIX constexpr ngen::InstructionModifier CG::w_repl; \
\
PREFIX constexpr ngen::InstructionModifier CG::ze; \
PREFIX constexpr ngen::InstructionModifier CG::eq; \
PREFIX constexpr ngen::InstructionModifier CG::nz; \
PREFIX constexpr ngen::InstructionModifier CG::ne; \
PREFIX constexpr ngen::InstructionModifier CG::gt; \
PREFIX constexpr ngen::InstructionModifier CG::ge; \
PREFIX constexpr ngen::InstructionModifier CG::lt; \
PREFIX constexpr ngen::InstructionModifier CG::le; \
PREFIX constexpr ngen::InstructionModifier CG::ov; \
PREFIX constexpr ngen::InstructionModifier CG::un; \
PREFIX constexpr ngen::InstructionModifier CG::eo; \
\
PREFIX constexpr ngen::InstructionModifier CG::M0; \
PREFIX constexpr ngen::InstructionModifier CG::M4; \
PREFIX constexpr ngen::InstructionModifier CG::M8; \
PREFIX constexpr ngen::InstructionModifier CG::M12; \
PREFIX constexpr ngen::InstructionModifier CG::M16; \
PREFIX constexpr ngen::InstructionModifier CG::M20; \
PREFIX constexpr ngen::InstructionModifier CG::M24; \
PREFIX constexpr ngen::InstructionModifier CG::M28; \
\
PREFIX constexpr ngen::SBID CG::sb0; \
PREFIX constexpr ngen::SBID CG::sb1; \
PREFIX constexpr ngen::SBID CG::sb2; \
PREFIX constexpr ngen::SBID CG::sb3; \
PREFIX constexpr ngen::SBID CG::sb4; \
PREFIX constexpr ngen::SBID CG::sb5; \
PREFIX constexpr ngen::SBID CG::sb6; \
PREFIX constexpr ngen::SBID CG::sb7; \
PREFIX constexpr ngen::SBID CG::sb8; \
PREFIX constexpr ngen::SBID CG::sb9; \
PREFIX constexpr ngen::SBID CG::sb10; \
PREFIX constexpr ngen::SBID CG::sb11; \
PREFIX constexpr ngen::SBID CG::sb12; \
PREFIX constexpr ngen::SBID CG::sb13; \
PREFIX constexpr ngen::SBID CG::sb14; \
PREFIX constexpr ngen::SBID CG::sb15; \
PREFIX constexpr ngen::SBID CG::sb16; \
PREFIX constexpr ngen::SBID CG::sb17; \
PREFIX constexpr ngen::SBID CG::sb18; \
PREFIX constexpr ngen::SBID CG::sb19; \
PREFIX constexpr ngen::SBID CG::sb20; \
PREFIX constexpr ngen::SBID CG::sb21; \
PREFIX constexpr ngen::SBID CG::sb22; \
PREFIX constexpr ngen::SBID CG::sb23; \
PREFIX constexpr ngen::SBID CG::sb24; \
PREFIX constexpr ngen::SBID CG::sb25; \
PREFIX constexpr ngen::SBID CG::sb26; \
PREFIX constexpr ngen::SBID CG::sb27; \
PREFIX constexpr ngen::SBID CG::sb28; \
PREFIX constexpr ngen::SBID CG::sb29; \
PREFIX constexpr ngen::SBID CG::sb30; \
PREFIX constexpr ngen::SBID CG::sb31; \
PREFIX constexpr ngen::SWSBInfo CG::NoAccSBSet; \
\
PREFIX constexpr ngen::AddressBase CG::A32; \
PREFIX constexpr ngen::AddressBase CG::A32NC; \
PREFIX constexpr ngen::AddressBase CG::A64; \
PREFIX constexpr ngen::AddressBase CG::A64NC; \
PREFIX constexpr ngen::AddressBase CG::SLM; \
\
PREFIX constexpr ngen::DataSpecLSC CG::D8; \
PREFIX constexpr ngen::DataSpecLSC CG::D16; \
PREFIX constexpr ngen::DataSpecLSC CG::D32; \
PREFIX constexpr ngen::DataSpecLSC CG::D64; \
PREFIX constexpr ngen::DataSpecLSC CG::D8U32; \
PREFIX constexpr ngen::DataSpecLSC CG::D16U32; \
PREFIX constexpr ngen::DataSpecLSC CG::D8T; \
PREFIX constexpr ngen::DataSpecLSC CG::D16T; \
PREFIX constexpr ngen::DataSpecLSC CG::D32T; \
PREFIX constexpr ngen::DataSpecLSC CG::D64T; \
PREFIX constexpr ngen::DataSpecLSC CG::D8U32T; \
PREFIX constexpr ngen::DataSpecLSC CG::D16U32T; \
PREFIX constexpr ngen::DataSpecLSC CG::V1; \
PREFIX constexpr ngen::DataSpecLSC CG::V2; \
PREFIX constexpr ngen::DataSpecLSC CG::V3; \
PREFIX constexpr ngen::DataSpecLSC CG::V4; \
PREFIX constexpr ngen::DataSpecLSC CG::V8; \
PREFIX constexpr ngen::DataSpecLSC CG::V16; \
PREFIX constexpr ngen::DataSpecLSC CG::V32; \
PREFIX constexpr ngen::DataSpecLSC CG::V64; \
PREFIX constexpr ngen::DataSpecLSC CG::V1T; \
PREFIX constexpr ngen::DataSpecLSC CG::V2T; \
PREFIX constexpr ngen::DataSpecLSC CG::V3T; \
PREFIX constexpr ngen::DataSpecLSC CG::V4T; \
PREFIX constexpr ngen::DataSpecLSC CG::V8T; \
PREFIX constexpr ngen::DataSpecLSC CG::V16T; \
PREFIX constexpr ngen::DataSpecLSC CG::V32T; \
PREFIX constexpr ngen::DataSpecLSC CG::V64T; \
PREFIX constexpr ngen::DataSpecLSC CG::transpose; \
PREFIX constexpr ngen::DataSpecLSC CG::vnni; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1IAR_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WT_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WT_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WB_L3WB;

#ifndef NGEN_SHORT_NAMES
#define NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX)
#else
#define NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX) \
PREFIX constexpr const ngen::IndirectRegisterFrame &CG::r; \
PREFIX constexpr const ngen::InstructionModifier &CG::W;
#endif

#define NGEN_REGISTER_DECL_EXTRA2A(CG,PREFIX) \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L3CC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L3CC;

#if !XE3
#define NGEN_REGISTER_DECL_EXTRA2(CG,PREFIX)
#else
#define NGEN_REGISTER_DECL_EXTRA2(CG,PREFIX) \
PREFIX constexpr ngen::ScalarRegister CG::s0;
#endif

#ifndef PRERELEASE_HW
#define NGEN_REGISTER_DECL_EXTRA3(CG,PREFIX)
#else
#define NGEN_REGISTER_DECL_EXTRA3(CG,PREFIX) \
PREFIX constexpr ngen::InstructionModifier CG::Fwd;
#endif

#if !XE3P
#define NGEN_REGISTER_DECL_EXTRA4(CG,PREFIX)
#else
#define NGEN_REGISTER_DECL_EXTRA4(CG,PREFIX) \
PREFIX constexpr ngen::GRF CG::r256; \
PREFIX constexpr ngen::GRF CG::r257; \
PREFIX constexpr ngen::GRF CG::r258; \
PREFIX constexpr ngen::GRF CG::r259; \
PREFIX constexpr ngen::GRF CG::r260; \
PREFIX constexpr ngen::GRF CG::r261; \
PREFIX constexpr ngen::GRF CG::r262; \
PREFIX constexpr ngen::GRF CG::r263; \
PREFIX constexpr ngen::GRF CG::r264; \
PREFIX constexpr ngen::GRF CG::r265; \
PREFIX constexpr ngen::GRF CG::r266; \
PREFIX constexpr ngen::GRF CG::r267; \
PREFIX constexpr ngen::GRF CG::r268; \
PREFIX constexpr ngen::GRF CG::r269; \
PREFIX constexpr ngen::GRF CG::r270; \
PREFIX constexpr ngen::GRF CG::r271; \
PREFIX constexpr ngen::GRF CG::r272; \
PREFIX constexpr ngen::GRF CG::r273; \
PREFIX constexpr ngen::GRF CG::r274; \
PREFIX constexpr ngen::GRF CG::r275; \
PREFIX constexpr ngen::GRF CG::r276; \
PREFIX constexpr ngen::GRF CG::r277; \
PREFIX constexpr ngen::GRF CG::r278; \
PREFIX constexpr ngen::GRF CG::r279; \
PREFIX constexpr ngen::GRF CG::r280; \
PREFIX constexpr ngen::GRF CG::r281; \
PREFIX constexpr ngen::GRF CG::r282; \
PREFIX constexpr ngen::GRF CG::r283; \
PREFIX constexpr ngen::GRF CG::r284; \
PREFIX constexpr ngen::GRF CG::r285; \
PREFIX constexpr ngen::GRF CG::r286; \
PREFIX constexpr ngen::GRF CG::r287; \
PREFIX constexpr ngen::GRF CG::r288; \
PREFIX constexpr ngen::GRF CG::r289; \
PREFIX constexpr ngen::GRF CG::r290; \
PREFIX constexpr ngen::GRF CG::r291; \
PREFIX constexpr ngen::GRF CG::r292; \
PREFIX constexpr ngen::GRF CG::r293; \
PREFIX constexpr ngen::GRF CG::r294; \
PREFIX constexpr ngen::GRF CG::r295; \
PREFIX constexpr ngen::GRF CG::r296; \
PREFIX constexpr ngen::GRF CG::r297; \
PREFIX constexpr ngen::GRF CG::r298; \
PREFIX constexpr ngen::GRF CG::r299; \
PREFIX constexpr ngen::GRF CG::r300; \
PREFIX constexpr ngen::GRF CG::r301; \
PREFIX constexpr ngen::GRF CG::r302; \
PREFIX constexpr ngen::GRF CG::r303; \
PREFIX constexpr ngen::GRF CG::r304; \
PREFIX constexpr ngen::GRF CG::r305; \
PREFIX constexpr ngen::GRF CG::r306; \
PREFIX constexpr ngen::GRF CG::r307; \
PREFIX constexpr ngen::GRF CG::r308; \
PREFIX constexpr ngen::GRF CG::r309; \
PREFIX constexpr ngen::GRF CG::r310; \
PREFIX constexpr ngen::GRF CG::r311; \
PREFIX constexpr ngen::GRF CG::r312; \
PREFIX constexpr ngen::GRF CG::r313; \
PREFIX constexpr ngen::GRF CG::r314; \
PREFIX constexpr ngen::GRF CG::r315; \
PREFIX constexpr ngen::GRF CG::r316; \
PREFIX constexpr ngen::GRF CG::r317; \
PREFIX constexpr ngen::GRF CG::r318; \
PREFIX constexpr ngen::GRF CG::r319; \
PREFIX constexpr ngen::GRF CG::r320; \
PREFIX constexpr ngen::GRF CG::r321; \
PREFIX constexpr ngen::GRF CG::r322; \
PREFIX constexpr ngen::GRF CG::r323; \
PREFIX constexpr ngen::GRF CG::r324; \
PREFIX constexpr ngen::GRF CG::r325; \
PREFIX constexpr ngen::GRF CG::r326; \
PREFIX constexpr ngen::GRF CG::r327; \
PREFIX constexpr ngen::GRF CG::r328; \
PREFIX constexpr ngen::GRF CG::r329; \
PREFIX constexpr ngen::GRF CG::r330; \
PREFIX constexpr ngen::GRF CG::r331; \
PREFIX constexpr ngen::GRF CG::r332; \
PREFIX constexpr ngen::GRF CG::r333; \
PREFIX constexpr ngen::GRF CG::r334; \
PREFIX constexpr ngen::GRF CG::r335; \
PREFIX constexpr ngen::GRF CG::r336; \
PREFIX constexpr ngen::GRF CG::r337; \
PREFIX constexpr ngen::GRF CG::r338; \
PREFIX constexpr ngen::GRF CG::r339; \
PREFIX constexpr ngen::GRF CG::r340; \
PREFIX constexpr ngen::GRF CG::r341; \
PREFIX constexpr ngen::GRF CG::r342; \
PREFIX constexpr ngen::GRF CG::r343; \
PREFIX constexpr ngen::GRF CG::r344; \
PREFIX constexpr ngen::GRF CG::r345; \
PREFIX constexpr ngen::GRF CG::r346; \
PREFIX constexpr ngen::GRF CG::r347; \
PREFIX constexpr ngen::GRF CG::r348; \
PREFIX constexpr ngen::GRF CG::r349; \
PREFIX constexpr ngen::GRF CG::r350; \
PREFIX constexpr ngen::GRF CG::r351; \
PREFIX constexpr ngen::GRF CG::r352; \
PREFIX constexpr ngen::GRF CG::r353; \
PREFIX constexpr ngen::GRF CG::r354; \
PREFIX constexpr ngen::GRF CG::r355; \
PREFIX constexpr ngen::GRF CG::r356; \
PREFIX constexpr ngen::GRF CG::r357; \
PREFIX constexpr ngen::GRF CG::r358; \
PREFIX constexpr ngen::GRF CG::r359; \
PREFIX constexpr ngen::GRF CG::r360; \
PREFIX constexpr ngen::GRF CG::r361; \
PREFIX constexpr ngen::GRF CG::r362; \
PREFIX constexpr ngen::GRF CG::r363; \
PREFIX constexpr ngen::GRF CG::r364; \
PREFIX constexpr ngen::GRF CG::r365; \
PREFIX constexpr ngen::GRF CG::r366; \
PREFIX constexpr ngen::GRF CG::r367; \
PREFIX constexpr ngen::GRF CG::r368; \
PREFIX constexpr ngen::GRF CG::r369; \
PREFIX constexpr ngen::GRF CG::r370; \
PREFIX constexpr ngen::GRF CG::r371; \
PREFIX constexpr ngen::GRF CG::r372; \
PREFIX constexpr ngen::GRF CG::r373; \
PREFIX constexpr ngen::GRF CG::r374; \
PREFIX constexpr ngen::GRF CG::r375; \
PREFIX constexpr ngen::GRF CG::r376; \
PREFIX constexpr ngen::GRF CG::r377; \
PREFIX constexpr ngen::GRF CG::r378; \
PREFIX constexpr ngen::GRF CG::r379; \
PREFIX constexpr ngen::GRF CG::r380; \
PREFIX constexpr ngen::GRF CG::r381; \
PREFIX constexpr ngen::GRF CG::r382; \
PREFIX constexpr ngen::GRF CG::r383; \
PREFIX constexpr ngen::GRF CG::r384; \
PREFIX constexpr ngen::GRF CG::r385; \
PREFIX constexpr ngen::GRF CG::r386; \
PREFIX constexpr ngen::GRF CG::r387; \
PREFIX constexpr ngen::GRF CG::r388; \
PREFIX constexpr ngen::GRF CG::r389; \
PREFIX constexpr ngen::GRF CG::r390; \
PREFIX constexpr ngen::GRF CG::r391; \
PREFIX constexpr ngen::GRF CG::r392; \
PREFIX constexpr ngen::GRF CG::r393; \
PREFIX constexpr ngen::GRF CG::r394; \
PREFIX constexpr ngen::GRF CG::r395; \
PREFIX constexpr ngen::GRF CG::r396; \
PREFIX constexpr ngen::GRF CG::r397; \
PREFIX constexpr ngen::GRF CG::r398; \
PREFIX constexpr ngen::GRF CG::r399; \
PREFIX constexpr ngen::GRF CG::r400; \
PREFIX constexpr ngen::GRF CG::r401; \
PREFIX constexpr ngen::GRF CG::r402; \
PREFIX constexpr ngen::GRF CG::r403; \
PREFIX constexpr ngen::GRF CG::r404; \
PREFIX constexpr ngen::GRF CG::r405; \
PREFIX constexpr ngen::GRF CG::r406; \
PREFIX constexpr ngen::GRF CG::r407; \
PREFIX constexpr ngen::GRF CG::r408; \
PREFIX constexpr ngen::GRF CG::r409; \
PREFIX constexpr ngen::GRF CG::r410; \
PREFIX constexpr ngen::GRF CG::r411; \
PREFIX constexpr ngen::GRF CG::r412; \
PREFIX constexpr ngen::GRF CG::r413; \
PREFIX constexpr ngen::GRF CG::r414; \
PREFIX constexpr ngen::GRF CG::r415; \
PREFIX constexpr ngen::GRF CG::r416; \
PREFIX constexpr ngen::GRF CG::r417; \
PREFIX constexpr ngen::GRF CG::r418; \
PREFIX constexpr ngen::GRF CG::r419; \
PREFIX constexpr ngen::GRF CG::r420; \
PREFIX constexpr ngen::GRF CG::r421; \
PREFIX constexpr ngen::GRF CG::r422; \
PREFIX constexpr ngen::GRF CG::r423; \
PREFIX constexpr ngen::GRF CG::r424; \
PREFIX constexpr ngen::GRF CG::r425; \
PREFIX constexpr ngen::GRF CG::r426; \
PREFIX constexpr ngen::GRF CG::r427; \
PREFIX constexpr ngen::GRF CG::r428; \
PREFIX constexpr ngen::GRF CG::r429; \
PREFIX constexpr ngen::GRF CG::r430; \
PREFIX constexpr ngen::GRF CG::r431; \
PREFIX constexpr ngen::GRF CG::r432; \
PREFIX constexpr ngen::GRF CG::r433; \
PREFIX constexpr ngen::GRF CG::r434; \
PREFIX constexpr ngen::GRF CG::r435; \
PREFIX constexpr ngen::GRF CG::r436; \
PREFIX constexpr ngen::GRF CG::r437; \
PREFIX constexpr ngen::GRF CG::r438; \
PREFIX constexpr ngen::GRF CG::r439; \
PREFIX constexpr ngen::GRF CG::r440; \
PREFIX constexpr ngen::GRF CG::r441; \
PREFIX constexpr ngen::GRF CG::r442; \
PREFIX constexpr ngen::GRF CG::r443; \
PREFIX constexpr ngen::GRF CG::r444; \
PREFIX constexpr ngen::GRF CG::r445; \
PREFIX constexpr ngen::GRF CG::r446; \
PREFIX constexpr ngen::GRF CG::r447; \
PREFIX constexpr ngen::GRF CG::r448; \
PREFIX constexpr ngen::GRF CG::r449; \
PREFIX constexpr ngen::GRF CG::r450; \
PREFIX constexpr ngen::GRF CG::r451; \
PREFIX constexpr ngen::GRF CG::r452; \
PREFIX constexpr ngen::GRF CG::r453; \
PREFIX constexpr ngen::GRF CG::r454; \
PREFIX constexpr ngen::GRF CG::r455; \
PREFIX constexpr ngen::GRF CG::r456; \
PREFIX constexpr ngen::GRF CG::r457; \
PREFIX constexpr ngen::GRF CG::r458; \
PREFIX constexpr ngen::GRF CG::r459; \
PREFIX constexpr ngen::GRF CG::r460; \
PREFIX constexpr ngen::GRF CG::r461; \
PREFIX constexpr ngen::GRF CG::r462; \
PREFIX constexpr ngen::GRF CG::r463; \
PREFIX constexpr ngen::GRF CG::r464; \
PREFIX constexpr ngen::GRF CG::r465; \
PREFIX constexpr ngen::GRF CG::r466; \
PREFIX constexpr ngen::GRF CG::r467; \
PREFIX constexpr ngen::GRF CG::r468; \
PREFIX constexpr ngen::GRF CG::r469; \
PREFIX constexpr ngen::GRF CG::r470; \
PREFIX constexpr ngen::GRF CG::r471; \
PREFIX constexpr ngen::GRF CG::r472; \
PREFIX constexpr ngen::GRF CG::r473; \
PREFIX constexpr ngen::GRF CG::r474; \
PREFIX constexpr ngen::GRF CG::r475; \
PREFIX constexpr ngen::GRF CG::r476; \
PREFIX constexpr ngen::GRF CG::r477; \
PREFIX constexpr ngen::GRF CG::r478; \
PREFIX constexpr ngen::GRF CG::r479; \
PREFIX constexpr ngen::GRF CG::r480; \
PREFIX constexpr ngen::GRF CG::r481; \
PREFIX constexpr ngen::GRF CG::r482; \
PREFIX constexpr ngen::GRF CG::r483; \
PREFIX constexpr ngen::GRF CG::r484; \
PREFIX constexpr ngen::GRF CG::r485; \
PREFIX constexpr ngen::GRF CG::r486; \
PREFIX constexpr ngen::GRF CG::r487; \
PREFIX constexpr ngen::GRF CG::r488; \
PREFIX constexpr ngen::GRF CG::r489; \
PREFIX constexpr ngen::GRF CG::r490; \
PREFIX constexpr ngen::GRF CG::r491; \
PREFIX constexpr ngen::GRF CG::r492; \
PREFIX constexpr ngen::GRF CG::r493; \
PREFIX constexpr ngen::GRF CG::r494; \
PREFIX constexpr ngen::GRF CG::r495; \
PREFIX constexpr ngen::GRF CG::r496; \
PREFIX constexpr ngen::GRF CG::r497; \
PREFIX constexpr ngen::GRF CG::r498; \
PREFIX constexpr ngen::GRF CG::r499; \
PREFIX constexpr ngen::GRF CG::r500; \
PREFIX constexpr ngen::GRF CG::r501; \
PREFIX constexpr ngen::GRF CG::r502; \
PREFIX constexpr ngen::GRF CG::r503; \
PREFIX constexpr ngen::GRF CG::r504; \
PREFIX constexpr ngen::GRF CG::r505; \
PREFIX constexpr ngen::GRF CG::r506; \
PREFIX constexpr ngen::GRF CG::r507; \
PREFIX constexpr ngen::GRF CG::r508; \
PREFIX constexpr ngen::GRF CG::r509; \
PREFIX constexpr ngen::GRF CG::r510; \
PREFIX constexpr ngen::GRF CG::r511; \
PREFIX constexpr ngen::AddressBase CG::A64_A32U; \
PREFIX constexpr ngen::AddressBase CG::A64_A32S; \
PREFIX constexpr ngen::DataSpecLSC CG::Overfetch; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L2UC_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L2UC_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L2C_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L2C_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L2UC_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L2UC_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L2C_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1C_L2C_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2UC_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2UC_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2C_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2C_L3C; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1IAR_L2IAR_L3IAR; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L2UC_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1UC_L2WB_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WT_L2UC_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WT_L2UC_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WT_L2WB_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2UC_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2WB_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1S_L2WB_L3WB; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WB_L2WB_L3UC; \
PREFIX constexpr ngen::CacheSettingsLSC CG::L1WB_L2UC_L3WB;
#endif

#define NGEN_REGISTER_DECL(CG,PREFIX) \
NGEN_REGISTER_DECL_MAIN(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA2A(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA2(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA3(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA4(CG,PREFIX)

#include "ngen.hpp"
NGEN_REGISTER_DECL(ngen::BinaryCodeGenerator<hw>, template <ngen::HW hw>)

#ifdef NGEN_ASM
#include "ngen_asm.hpp"
NGEN_REGISTER_DECL(ngen::AsmCodeGenerator, /* nothing */)
#endif

template class ngen::BinaryCodeGenerator<ngen::HW::Unknown>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen9>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen10>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen11>;
template class ngen::BinaryCodeGenerator<ngen::HW::Gen12LP>;
template class ngen::BinaryCodeGenerator<ngen::HW::XeHP>;
template class ngen::BinaryCodeGenerator<ngen::HW::XeHPG>;
template class ngen::BinaryCodeGenerator<ngen::HW::XeHPC>;
template class ngen::BinaryCodeGenerator<ngen::HW::Xe2>;
#if XE3
template class ngen::BinaryCodeGenerator<ngen::HW::Xe3>;
#endif
#if XE3P
template class ngen::BinaryCodeGenerator<ngen::HW::Xe3p>;
#endif

#endif /* (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS) */
