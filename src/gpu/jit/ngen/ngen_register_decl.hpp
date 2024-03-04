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
PREFIX constexpr NGEN_NAMESPACE::IndirectRegisterFrame CG::indirect; \
\
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r0; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r1; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r2; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r3; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r4; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r5; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r6; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r7; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r8; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r9; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r10; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r11; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r12; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r13; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r14; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r15; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r16; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r17; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r18; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r19; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r20; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r21; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r22; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r23; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r24; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r25; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r26; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r27; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r28; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r29; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r30; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r31; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r32; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r33; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r34; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r35; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r36; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r37; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r38; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r39; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r40; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r41; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r42; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r43; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r44; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r45; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r46; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r47; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r48; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r49; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r50; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r51; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r52; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r53; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r54; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r55; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r56; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r57; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r58; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r59; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r60; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r61; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r62; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r63; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r64; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r65; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r66; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r67; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r68; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r69; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r70; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r71; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r72; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r73; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r74; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r75; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r76; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r77; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r78; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r79; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r80; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r81; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r82; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r83; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r84; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r85; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r86; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r87; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r88; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r89; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r90; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r91; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r92; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r93; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r94; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r95; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r96; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r97; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r98; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r99; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r100; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r101; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r102; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r103; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r104; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r105; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r106; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r107; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r108; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r109; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r110; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r111; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r112; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r113; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r114; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r115; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r116; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r117; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r118; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r119; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r120; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r121; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r122; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r123; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r124; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r125; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r126; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r127; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r128; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r129; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r130; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r131; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r132; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r133; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r134; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r135; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r136; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r137; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r138; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r139; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r140; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r141; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r142; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r143; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r144; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r145; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r146; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r147; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r148; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r149; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r150; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r151; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r152; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r153; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r154; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r155; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r156; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r157; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r158; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r159; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r160; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r161; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r162; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r163; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r164; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r165; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r166; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r167; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r168; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r169; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r170; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r171; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r172; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r173; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r174; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r175; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r176; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r177; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r178; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r179; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r180; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r181; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r182; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r183; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r184; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r185; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r186; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r187; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r188; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r189; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r190; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r191; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r192; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r193; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r194; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r195; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r196; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r197; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r198; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r199; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r200; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r201; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r202; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r203; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r204; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r205; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r206; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r207; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r208; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r209; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r210; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r211; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r212; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r213; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r214; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r215; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r216; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r217; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r218; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r219; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r220; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r221; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r222; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r223; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r224; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r225; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r226; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r227; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r228; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r229; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r230; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r231; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r232; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r233; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r234; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r235; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r236; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r237; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r238; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r239; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r240; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r241; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r242; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r243; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r244; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r245; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r246; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r247; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r248; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r249; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r250; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r251; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r252; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r253; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r254; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r255; \
\
PREFIX constexpr NGEN_NAMESPACE::NullRegister CG::null; \
PREFIX constexpr NGEN_NAMESPACE::AddressRegister CG::a0; \
PREFIX constexpr NGEN_NAMESPACE::AccumulatorRegister CG::acc0; \
PREFIX constexpr NGEN_NAMESPACE::AccumulatorRegister CG::acc1; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc2; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc3; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc4; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc5; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc6; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc7; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc8; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc9; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme0; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme1; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme2; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme3; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme4; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme5; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme6; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme7; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::nomme; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::noacc; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f0; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f1; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f2; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f3; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f0_0; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f0_1; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f1_0; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f1_1; \
PREFIX constexpr NGEN_NAMESPACE::ChannelEnableRegister CG::ce0; \
PREFIX constexpr NGEN_NAMESPACE::StackPointerRegister CG::sp; \
PREFIX constexpr NGEN_NAMESPACE::StateRegister CG::sr0; \
PREFIX constexpr NGEN_NAMESPACE::StateRegister CG::sr1; \
PREFIX constexpr NGEN_NAMESPACE::ControlRegister CG::cr0; \
PREFIX constexpr NGEN_NAMESPACE::NotificationRegister CG::n0; \
PREFIX constexpr NGEN_NAMESPACE::InstructionPointerRegister CG::ip; \
PREFIX constexpr NGEN_NAMESPACE::ThreadDependencyRegister CG::tdr0; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm0; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm1; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm2; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm3; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm4; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::pm0; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tp0; \
PREFIX constexpr NGEN_NAMESPACE::DebugRegister CG::dbg0; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc0; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc1; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc2; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc3; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoDDClr; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoDDChk; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::AccWrEn; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoSrcDepSet; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Breakpoint; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::sat; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoMask; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ExBSO; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::AutoSWSB; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Serialize; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::EOT; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Align1; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Align16; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Atomic; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Switch; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoPreempt; \
\
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::anyv; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::allv; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any2h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all2h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any4h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all4h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any8h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all8h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any16h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all16h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any32h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all32h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::x_repl; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::y_repl; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::z_repl; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::w_repl; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ze; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::eq; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::nz; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ne; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::gt; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ge; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::lt; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::le; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ov; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::un; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::eo; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M0; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M4; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M8; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M12; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M16; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M20; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M24; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M28; \
\
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb0; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb1; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb2; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb3; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb4; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb5; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb6; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb7; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb8; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb9; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb10; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb11; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb12; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb13; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb14; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb15; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb16; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb17; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb18; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb19; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb20; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb21; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb22; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb23; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb24; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb25; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb26; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb27; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb28; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb29; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb30; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb31; \
PREFIX constexpr NGEN_NAMESPACE::SWSBInfo CG::NoAccSBSet; \
\
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A32; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A32NC; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A64; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A64NC; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::SLM; \
\
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D64; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8U32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16U32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D64T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8U32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16U32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V1; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V2; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V3; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V4; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V8; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V16; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V64; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V1T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V2T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V3T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V4T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V8T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V16T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V64T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::transpose; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::vnni; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1C_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1C_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1S_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1S_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1IAR_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3WB; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1WT_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1WT_L3WB; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1S_L3WB; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1WB_L3WB;

#ifndef NGEN_SHORT_NAMES
#define NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX)
#else
#define NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX) \
PREFIX constexpr const NGEN_NAMESPACE::IndirectRegisterFrame &CG::r; \
PREFIX constexpr const NGEN_NAMESPACE::InstructionModifier &CG::W;
#endif

#define NGEN_REGISTER_DECL_EXTRA2A(CG,PREFIX) \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1C_L3CC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3CC;

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
NGEN_REGISTER_DECL(NGEN_NAMESPACE::BinaryCodeGenerator<hw>, template <NGEN_NAMESPACE::HW hw>)

#ifdef NGEN_ASM
#include "ngen_asm.hpp"
NGEN_REGISTER_DECL(NGEN_NAMESPACE::AsmCodeGenerator, /* nothing */)
#endif

template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Unknown>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen9>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen10>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen11>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen12LP>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::XeHP>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::XeHPG>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::XeHPC>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Xe2>;
#if XE3
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Xe3>;
#endif
#if XE3P
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Xe3p>;
#endif

#endif /* (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS) */
