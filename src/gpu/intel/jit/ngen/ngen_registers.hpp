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
 * Do not #include this file directly; ngen uses it internally.
 */

#if defined(NGEN_CPP11) && defined(NGEN_GLOBAL_REGS)
#define constexpr_reg const
#define constexpr_reg_const const
#else
#define constexpr_reg constexpr
#define constexpr_reg_const constexpr const
#endif

static constexpr_reg IndirectRegisterFrame indirect{};
#ifdef NGEN_SHORT_NAMES
static constexpr_reg_const IndirectRegisterFrame &r = indirect;
#endif

static constexpr_reg GRF r0{0}, r1{1}, r2{2}, r3{3}, r4{4}, r5{5}, r6{6}, r7{7};
static constexpr_reg GRF r8{8}, r9{9}, r10{10}, r11{11}, r12{12}, r13{13}, r14{14}, r15{15};
static constexpr_reg GRF r16{16}, r17{17}, r18{18}, r19{19}, r20{20}, r21{21}, r22{22}, r23{23};
static constexpr_reg GRF r24{24}, r25{25}, r26{26}, r27{27}, r28{28}, r29{29}, r30{30}, r31{31};
static constexpr_reg GRF r32{32}, r33{33}, r34{34}, r35{35}, r36{36}, r37{37}, r38{38}, r39{39};
static constexpr_reg GRF r40{40}, r41{41}, r42{42}, r43{43}, r44{44}, r45{45}, r46{46}, r47{47};
static constexpr_reg GRF r48{48}, r49{49}, r50{50}, r51{51}, r52{52}, r53{53}, r54{54}, r55{55};
static constexpr_reg GRF r56{56}, r57{57}, r58{58}, r59{59}, r60{60}, r61{61}, r62{62}, r63{63};
static constexpr_reg GRF r64{64}, r65{65}, r66{66}, r67{67}, r68{68}, r69{69}, r70{70}, r71{71};
static constexpr_reg GRF r72{72}, r73{73}, r74{74}, r75{75}, r76{76}, r77{77}, r78{78}, r79{79};
static constexpr_reg GRF r80{80}, r81{81}, r82{82}, r83{83}, r84{84}, r85{85}, r86{86}, r87{87};
static constexpr_reg GRF r88{88}, r89{89}, r90{90}, r91{91}, r92{92}, r93{93}, r94{94}, r95{95};
static constexpr_reg GRF r96{96}, r97{97}, r98{98}, r99{99}, r100{100}, r101{101}, r102{102}, r103{103};
static constexpr_reg GRF r104{104}, r105{105}, r106{106}, r107{107}, r108{108}, r109{109}, r110{110}, r111{111};
static constexpr_reg GRF r112{112}, r113{113}, r114{114}, r115{115}, r116{116}, r117{117}, r118{118}, r119{119};
static constexpr_reg GRF r120{120}, r121{121}, r122{122}, r123{123}, r124{124}, r125{125}, r126{126}, r127{127};
static constexpr_reg GRF r128{128}, r129{129}, r130{130}, r131{131}, r132{132}, r133{133}, r134{134}, r135{135};
static constexpr_reg GRF r136{136}, r137{137}, r138{138}, r139{139}, r140{140}, r141{141}, r142{142}, r143{143};
static constexpr_reg GRF r144{144}, r145{145}, r146{146}, r147{147}, r148{148}, r149{149}, r150{150}, r151{151};
static constexpr_reg GRF r152{152}, r153{153}, r154{154}, r155{155}, r156{156}, r157{157}, r158{158}, r159{159};
static constexpr_reg GRF r160{160}, r161{161}, r162{162}, r163{163}, r164{164}, r165{165}, r166{166}, r167{167};
static constexpr_reg GRF r168{168}, r169{169}, r170{170}, r171{171}, r172{172}, r173{173}, r174{174}, r175{175};
static constexpr_reg GRF r176{176}, r177{177}, r178{178}, r179{179}, r180{180}, r181{181}, r182{182}, r183{183};
static constexpr_reg GRF r184{184}, r185{185}, r186{186}, r187{187}, r188{188}, r189{189}, r190{190}, r191{191};
static constexpr_reg GRF r192{192}, r193{193}, r194{194}, r195{195}, r196{196}, r197{197}, r198{198}, r199{199};
static constexpr_reg GRF r200{200}, r201{201}, r202{202}, r203{203}, r204{204}, r205{205}, r206{206}, r207{207};
static constexpr_reg GRF r208{208}, r209{209}, r210{210}, r211{211}, r212{212}, r213{213}, r214{214}, r215{215};
static constexpr_reg GRF r216{216}, r217{217}, r218{218}, r219{219}, r220{220}, r221{221}, r222{222}, r223{223};
static constexpr_reg GRF r224{224}, r225{225}, r226{226}, r227{227}, r228{228}, r229{229}, r230{230}, r231{231};
static constexpr_reg GRF r232{232}, r233{233}, r234{234}, r235{235}, r236{236}, r237{237}, r238{238}, r239{239};
static constexpr_reg GRF r240{240}, r241{241}, r242{242}, r243{243}, r244{244}, r245{245}, r246{246}, r247{247};
static constexpr_reg GRF r248{248}, r249{249}, r250{250}, r251{251}, r252{252}, r253{253}, r254{254}, r255{255};
#if XE3P
static constexpr_reg GRF r256{256}, r257{257}, r258{258}, r259{259}, r260{260}, r261{261}, r262{262}, r263{263};
static constexpr_reg GRF r264{264}, r265{265}, r266{266}, r267{267}, r268{268}, r269{269}, r270{270}, r271{271};
static constexpr_reg GRF r272{272}, r273{273}, r274{274}, r275{275}, r276{276}, r277{277}, r278{278}, r279{279};
static constexpr_reg GRF r280{280}, r281{281}, r282{282}, r283{283}, r284{284}, r285{285}, r286{286}, r287{287};
static constexpr_reg GRF r288{288}, r289{289}, r290{290}, r291{291}, r292{292}, r293{293}, r294{294}, r295{295};
static constexpr_reg GRF r296{296}, r297{297}, r298{298}, r299{299}, r300{300}, r301{301}, r302{302}, r303{303};
static constexpr_reg GRF r304{304}, r305{305}, r306{306}, r307{307}, r308{308}, r309{309}, r310{310}, r311{311};
static constexpr_reg GRF r312{312}, r313{313}, r314{314}, r315{315}, r316{316}, r317{317}, r318{318}, r319{319};
static constexpr_reg GRF r320{320}, r321{321}, r322{322}, r323{323}, r324{324}, r325{325}, r326{326}, r327{327};
static constexpr_reg GRF r328{328}, r329{329}, r330{330}, r331{331}, r332{332}, r333{333}, r334{334}, r335{335};
static constexpr_reg GRF r336{336}, r337{337}, r338{338}, r339{339}, r340{340}, r341{341}, r342{342}, r343{343};
static constexpr_reg GRF r344{344}, r345{345}, r346{346}, r347{347}, r348{348}, r349{349}, r350{350}, r351{351};
static constexpr_reg GRF r352{352}, r353{353}, r354{354}, r355{355}, r356{356}, r357{357}, r358{358}, r359{359};
static constexpr_reg GRF r360{360}, r361{361}, r362{362}, r363{363}, r364{364}, r365{365}, r366{366}, r367{367};
static constexpr_reg GRF r368{368}, r369{369}, r370{370}, r371{371}, r372{372}, r373{373}, r374{374}, r375{375};
static constexpr_reg GRF r376{376}, r377{377}, r378{378}, r379{379}, r380{380}, r381{381}, r382{382}, r383{383};
static constexpr_reg GRF r384{384}, r385{385}, r386{386}, r387{387}, r388{388}, r389{389}, r390{390}, r391{391};
static constexpr_reg GRF r392{392}, r393{393}, r394{394}, r395{395}, r396{396}, r397{397}, r398{398}, r399{399};
static constexpr_reg GRF r400{400}, r401{401}, r402{402}, r403{403}, r404{404}, r405{405}, r406{406}, r407{407};
static constexpr_reg GRF r408{408}, r409{409}, r410{410}, r411{411}, r412{412}, r413{413}, r414{414}, r415{415};
static constexpr_reg GRF r416{416}, r417{417}, r418{418}, r419{419}, r420{420}, r421{421}, r422{422}, r423{423};
static constexpr_reg GRF r424{424}, r425{425}, r426{426}, r427{427}, r428{428}, r429{429}, r430{430}, r431{431};
static constexpr_reg GRF r432{432}, r433{433}, r434{434}, r435{435}, r436{436}, r437{437}, r438{438}, r439{439};
static constexpr_reg GRF r440{440}, r441{441}, r442{442}, r443{443}, r444{444}, r445{445}, r446{446}, r447{447};
static constexpr_reg GRF r448{448}, r449{449}, r450{450}, r451{451}, r452{452}, r453{453}, r454{454}, r455{455};
static constexpr_reg GRF r456{456}, r457{457}, r458{458}, r459{459}, r460{460}, r461{461}, r462{462}, r463{463};
static constexpr_reg GRF r464{464}, r465{465}, r466{466}, r467{467}, r468{468}, r469{469}, r470{470}, r471{471};
static constexpr_reg GRF r472{472}, r473{473}, r474{474}, r475{475}, r476{476}, r477{477}, r478{478}, r479{479};
static constexpr_reg GRF r480{480}, r481{481}, r482{482}, r483{483}, r484{484}, r485{485}, r486{486}, r487{487};
static constexpr_reg GRF r488{488}, r489{489}, r490{490}, r491{491}, r492{492}, r493{493}, r494{494}, r495{495};
static constexpr_reg GRF r496{496}, r497{497}, r498{498}, r499{499}, r500{500}, r501{501}, r502{502}, r503{503};
static constexpr_reg GRF r504{504}, r505{505}, r506{506}, r507{507}, r508{508}, r509{509}, r510{510}, r511{511};
#endif

static constexpr_reg NullRegister null{};
static constexpr_reg AddressRegister a0{0};
static constexpr_reg AccumulatorRegister acc0{0}, acc1{1};
static constexpr_reg SpecialAccumulatorRegister acc2{2,0}, acc3{3,1}, acc4{4,2}, acc5{5,3}, acc6{6,4}, acc7{7,5}, acc8{8,6}, acc9{9,7};
static constexpr_reg SpecialAccumulatorRegister mme0{4,0}, mme1{5,1}, mme2{6,2}, mme3{7,3}, mme4{8,4}, mme5{9,5}, mme6{10,6}, mme7{11,7};
static constexpr_reg SpecialAccumulatorRegister nomme = SpecialAccumulatorRegister::createNoMME();
static constexpr_reg SpecialAccumulatorRegister noacc = nomme;
static constexpr_reg FlagRegister f0{0}, f1{1};
static constexpr_reg FlagRegister f0_0{0,0}, f0_1{0,1}, f1_0{1,0}, f1_1{1,1};
static constexpr_reg FlagRegister f2{2}, f3{3};
static constexpr_reg ChannelEnableRegister ce0{0};
static constexpr_reg StackPointerRegister sp{0};
#if XE3
static constexpr_reg ScalarRegister s0{0};
#endif
static constexpr_reg StateRegister sr0{0}, sr1{1};
static constexpr_reg ControlRegister cr0{0};
static constexpr_reg NotificationRegister n0{0};
static constexpr_reg InstructionPointerRegister ip{};
static constexpr_reg ThreadDependencyRegister tdr0{0};
static constexpr_reg PerformanceRegister tm0{0};
static constexpr_reg PerformanceRegister tm1{1};
static constexpr_reg PerformanceRegister tm2{2};
static constexpr_reg PerformanceRegister tm3{3};
static constexpr_reg PerformanceRegister tm4{4};
static constexpr_reg PerformanceRegister pm0{0,3}, tp0{0,4};
static constexpr_reg DebugRegister dbg0{0};
static constexpr_reg FlowControlRegister fc0{0}, fc1{1}, fc2{2}, fc3{3};

static constexpr_reg InstructionModifier NoDDClr = InstructionModifier::createNoDDClr();
static constexpr_reg InstructionModifier NoDDChk = InstructionModifier::createNoDDChk();
static constexpr_reg InstructionModifier AccWrEn = InstructionModifier::createAccWrCtrl();
static constexpr_reg InstructionModifier NoSrcDepSet = AccWrEn;
#ifdef PRERELEASE_HW
static constexpr_reg InstructionModifier Fwd = InstructionModifier::createFwd();
#endif
static constexpr_reg InstructionModifier Breakpoint = InstructionModifier::createDebugCtrl();
static constexpr_reg InstructionModifier sat = InstructionModifier::createSaturate();
static constexpr_reg InstructionModifier NoMask = InstructionModifier::createMaskCtrl(true);
static constexpr_reg InstructionModifier ExBSO = InstructionModifier::createExBSO();
static constexpr_reg InstructionModifier AutoSWSB = InstructionModifier::createAutoSWSB();
static constexpr_reg InstructionModifier Serialize = InstructionModifier::createSerialized();
static constexpr_reg InstructionModifier EOT = InstructionModifier::createEOT();
static constexpr_reg InstructionModifier Align1 = InstructionModifier::createAccessMode(0);
static constexpr_reg InstructionModifier Align16 = InstructionModifier::createAccessMode(1);

static constexpr_reg InstructionModifier Switch{ThreadCtrl::Switch};
static constexpr_reg InstructionModifier Atomic{ThreadCtrl::Atomic};
static constexpr_reg InstructionModifier NoPreempt{ThreadCtrl::NoPreempt};

#ifdef NGEN_SHORT_NAMES
static constexpr_reg_const InstructionModifier &W = NoMask;
#endif

static constexpr_reg PredCtrl anyv = PredCtrl::anyv;
static constexpr_reg PredCtrl allv = PredCtrl::allv;
static constexpr_reg PredCtrl any2h = PredCtrl::any2h;
static constexpr_reg PredCtrl all2h = PredCtrl::all2h;
static constexpr_reg PredCtrl any4h = PredCtrl::any4h;
static constexpr_reg PredCtrl all4h = PredCtrl::all4h;
static constexpr_reg PredCtrl any8h = PredCtrl::any8h;
static constexpr_reg PredCtrl all8h = PredCtrl::all8h;
static constexpr_reg PredCtrl any16h = PredCtrl::any16h;
static constexpr_reg PredCtrl all16h = PredCtrl::all16h;
static constexpr_reg PredCtrl any32h = PredCtrl::any32h;
static constexpr_reg PredCtrl all32h = PredCtrl::all32h;
static constexpr_reg PredCtrl any = PredCtrl::any;
static constexpr_reg PredCtrl all = PredCtrl::all;

static constexpr_reg InstructionModifier x_repl = InstructionModifier{PredCtrl::x};
static constexpr_reg InstructionModifier y_repl = InstructionModifier{PredCtrl::y};
static constexpr_reg InstructionModifier z_repl = InstructionModifier{PredCtrl::z};
static constexpr_reg InstructionModifier w_repl = InstructionModifier{PredCtrl::w};

static constexpr_reg InstructionModifier ze{ConditionModifier::ze};
static constexpr_reg InstructionModifier eq{ConditionModifier::eq};
static constexpr_reg InstructionModifier nz{ConditionModifier::ne};
static constexpr_reg InstructionModifier ne{ConditionModifier::nz};
static constexpr_reg InstructionModifier gt{ConditionModifier::gt};
static constexpr_reg InstructionModifier ge{ConditionModifier::ge};
static constexpr_reg InstructionModifier lt{ConditionModifier::lt};
static constexpr_reg InstructionModifier le{ConditionModifier::le};
static constexpr_reg InstructionModifier ov{ConditionModifier::ov};
static constexpr_reg InstructionModifier un{ConditionModifier::un};
static constexpr_reg InstructionModifier eo{ConditionModifier::eo};

static constexpr_reg InstructionModifier M0 = InstructionModifier::createChanOff(0);
static constexpr_reg InstructionModifier M4 = InstructionModifier::createChanOff(4);
static constexpr_reg InstructionModifier M8 = InstructionModifier::createChanOff(8);
static constexpr_reg InstructionModifier M12 = InstructionModifier::createChanOff(12);
static constexpr_reg InstructionModifier M16 = InstructionModifier::createChanOff(16);
static constexpr_reg InstructionModifier M20 = InstructionModifier::createChanOff(20);
static constexpr_reg InstructionModifier M24 = InstructionModifier::createChanOff(24);
static constexpr_reg InstructionModifier M28 = InstructionModifier::createChanOff(28);
static inline InstructionModifier ExecutionOffset(int off) { return InstructionModifier::createChanOff(off); }
#ifdef NGEN_SHORT_NAMES
static inline InstructionModifier M(int off) { return ExecutionOffset(off); }
#endif

static constexpr_reg SBID sb0{0}, sb1{1}, sb2{2}, sb3{3}, sb4{4}, sb5{5}, sb6{6}, sb7{7};
static constexpr_reg SBID sb8{8}, sb9{9}, sb10{10}, sb11{11}, sb12{12}, sb13{13}, sb14{14}, sb15{15};
static constexpr_reg SBID sb16{16}, sb17{17}, sb18{18}, sb19{19}, sb20{20}, sb21{21}, sb22{22}, sb23{23};
static constexpr_reg SBID sb24{24}, sb25{25}, sb26{16}, sb27{27}, sb28{28}, sb29{29}, sb30{30}, sb31{31};
static constexpr_reg SWSBInfo NoAccSBSet = SWSBInfo::createNoAccSBSet();

static constexpr_reg AddressBase A32 = AddressBase::createA32(true);
static constexpr_reg AddressBase A32NC = AddressBase::createA32(false);
static constexpr_reg AddressBase A64 = AddressBase::createA64(true);
static constexpr_reg AddressBase A64NC = AddressBase::createA64(false);
static constexpr_reg AddressBase SLM = AddressBase::createSLM();

#if XE3P
static constexpr_reg AddressBase A64_A32U = AddressBase::createA64A32U();
static constexpr_reg AddressBase A64_A32S = AddressBase::createA64A32S();
#endif

static inline AddressBase Surface(uint8_t index) { return AddressBase::createBTS(index); }
static inline AddressBase CC(uint8_t index) { return AddressBase::createCC(index); }
static inline AddressBase SC(uint8_t index) { return AddressBase::createSC(index); }

static inline AddressBase BTI(uint8_t index) { return AddressBase::createBTS(index); }
static inline AddressBase SS(uint32_t index) { return AddressBase::createSS(index); }
static inline AddressBase BSS(uint32_t index) { return AddressBase::createBSS(index); }

static constexpr_reg DataSpecLSC D8{DataSizeLSC::D8};
static constexpr_reg DataSpecLSC D16{DataSizeLSC::D16};
static constexpr_reg DataSpecLSC D32{DataSizeLSC::D32};
static constexpr_reg DataSpecLSC D64{DataSizeLSC::D64};
static constexpr_reg DataSpecLSC D8U32{DataSizeLSC::D8U32};
static constexpr_reg DataSpecLSC D16U32{DataSizeLSC::D16U32};

static constexpr_reg DataSpecLSC D8T = DataSpecLSC(DataSizeLSC::D8) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D16T = DataSpecLSC(DataSizeLSC::D16) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D32T = DataSpecLSC(DataSizeLSC::D32) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D64T = DataSpecLSC(DataSizeLSC::D64) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D8U32T = DataSpecLSC(DataSizeLSC::D8U32) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC D16U32T = DataSpecLSC(DataSizeLSC::D16U32) | DataSpecLSC::createTranspose();

static constexpr_reg DataSpecLSC V1 = DataSpecLSC::createV(1,0);
static constexpr_reg DataSpecLSC V2 = DataSpecLSC::createV(2,1);
static constexpr_reg DataSpecLSC V3 = DataSpecLSC::createV(3,2);
static constexpr_reg DataSpecLSC V4 = DataSpecLSC::createV(4,3);
static constexpr_reg DataSpecLSC V8 = DataSpecLSC::createV(8,4);
static constexpr_reg DataSpecLSC V16 = DataSpecLSC::createV(16,5);
static constexpr_reg DataSpecLSC V32 = DataSpecLSC::createV(32,6);
static constexpr_reg DataSpecLSC V64 = DataSpecLSC::createV(64,7);

static constexpr_reg DataSpecLSC V1T = DataSpecLSC::createV(1,0) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V2T = DataSpecLSC::createV(2,1) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V3T = DataSpecLSC::createV(3,2) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V4T = DataSpecLSC::createV(4,3) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V8T = DataSpecLSC::createV(8,4) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V16T = DataSpecLSC::createV(16,5) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V32T = DataSpecLSC::createV(32,6) | DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC V64T = DataSpecLSC::createV(64,7) | DataSpecLSC::createTranspose();

static constexpr_reg DataSpecLSC transpose = DataSpecLSC::createTranspose();
static constexpr_reg DataSpecLSC vnni = DataSpecLSC::createVNNI();
#if XE3P
static constexpr_reg DataSpecLSC Overfetch = DataSpecLSC::createOverfetch();
#endif

static constexpr_reg CacheSettingsLSC L1UC_L3UC = CacheSettingsLSC::L1UC_L3UC;
static constexpr_reg CacheSettingsLSC L1UC_L3C  = CacheSettingsLSC::L1UC_L3C;
static constexpr_reg CacheSettingsLSC L1C_L3UC  = CacheSettingsLSC::L1C_L3UC;
static constexpr_reg CacheSettingsLSC L1C_L3C   = CacheSettingsLSC::L1C_L3C;
static constexpr_reg CacheSettingsLSC L1S_L3UC  = CacheSettingsLSC::L1S_L3UC;
static constexpr_reg CacheSettingsLSC L1S_L3C   = CacheSettingsLSC::L1S_L3C;
static constexpr_reg CacheSettingsLSC L1IAR_L3C = CacheSettingsLSC::L1IAR_L3C;
static constexpr_reg CacheSettingsLSC L1UC_L3WB = CacheSettingsLSC::L1UC_L3WB;
static constexpr_reg CacheSettingsLSC L1WT_L3UC = CacheSettingsLSC::L1WT_L3UC;
static constexpr_reg CacheSettingsLSC L1WT_L3WB = CacheSettingsLSC::L1WT_L3WB;
static constexpr_reg CacheSettingsLSC L1S_L3WB  = CacheSettingsLSC::L1S_L3WB;
static constexpr_reg CacheSettingsLSC L1WB_L3WB = CacheSettingsLSC::L1WB_L3WB;
static constexpr_reg CacheSettingsLSC L1C_L3CC  = CacheSettingsLSC::L1C_L3CC;
static constexpr_reg CacheSettingsLSC L1UC_L3CC = CacheSettingsLSC::L1UC_L3CC;
#if XE3P
static constexpr_reg CacheSettingsLSC L1UC_L2UC_L3UC    = CacheSettingsLSC::L1UC_L2UC_L3UC;
static constexpr_reg CacheSettingsLSC L1UC_L2UC_L3C     = CacheSettingsLSC::L1UC_L2UC_L3C;
static constexpr_reg CacheSettingsLSC L1UC_L2C_L3UC     = CacheSettingsLSC::L1UC_L2C_L3UC;
static constexpr_reg CacheSettingsLSC L1UC_L2C_L3C      = CacheSettingsLSC::L1UC_L2C_L3C;
static constexpr_reg CacheSettingsLSC L1C_L2UC_L3UC     = CacheSettingsLSC::L1C_L2UC_L3UC;
static constexpr_reg CacheSettingsLSC L1C_L2UC_L3C      = CacheSettingsLSC::L1C_L2UC_L3C;
static constexpr_reg CacheSettingsLSC L1C_L2C_L3UC      = CacheSettingsLSC::L1C_L2C_L3UC;
static constexpr_reg CacheSettingsLSC L1C_L2C_L3C       = CacheSettingsLSC::L1C_L2C_L3C;
static constexpr_reg CacheSettingsLSC L1S_L2UC_L3UC     = CacheSettingsLSC::L1S_L2UC_L3UC;
static constexpr_reg CacheSettingsLSC L1S_L2UC_L3C      = CacheSettingsLSC::L1S_L2UC_L3C;
static constexpr_reg CacheSettingsLSC L1S_L2C_L3UC      = CacheSettingsLSC::L1S_L2C_L3UC;
static constexpr_reg CacheSettingsLSC L1S_L2C_L3C       = CacheSettingsLSC::L1S_L2C_L3C;
static constexpr_reg CacheSettingsLSC L1IAR_L2IAR_L3IAR = CacheSettingsLSC::L1IAR_L2IAR_L3IAR;
static constexpr_reg CacheSettingsLSC L1UC_L2UC_L3WB    = CacheSettingsLSC::L1UC_L2UC_L3WB;
static constexpr_reg CacheSettingsLSC L1UC_L2WB_L3UC    = CacheSettingsLSC::L1UC_L2WB_L3UC;
static constexpr_reg CacheSettingsLSC L1WT_L2UC_L3UC    = CacheSettingsLSC::L1WT_L2UC_L3UC;
static constexpr_reg CacheSettingsLSC L1WT_L2UC_L3WB    = CacheSettingsLSC::L1WT_L2UC_L3WB;
static constexpr_reg CacheSettingsLSC L1WT_L2WB_L3UC    = CacheSettingsLSC::L1WT_L2WB_L3UC;
static constexpr_reg CacheSettingsLSC L1S_L2UC_L3WB     = CacheSettingsLSC::L1S_L2UC_L3WB;
static constexpr_reg CacheSettingsLSC L1S_L2WB_L3UC     = CacheSettingsLSC::L1S_L2WB_L3UC;
static constexpr_reg CacheSettingsLSC L1S_L2WB_L3WB     = CacheSettingsLSC::L1S_L2WB_L3WB;
static constexpr_reg CacheSettingsLSC L1WB_L2WB_L3UC    = CacheSettingsLSC::L1WB_L2WB_L3UC;
static constexpr_reg CacheSettingsLSC L1WB_L2UC_L3WB    = CacheSettingsLSC::L1WB_L2UC_L3WB;
#endif
