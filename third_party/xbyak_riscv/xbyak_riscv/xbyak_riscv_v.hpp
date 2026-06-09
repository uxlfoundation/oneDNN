/*
	Copyright (C), 2023, MITSUNARI Shigeo
	Copyright (C), 2023, KNS Group LLC (YADRO)
	Licensed under the 3-Clause BSD License
	You may obtain a copy of the License at https://opensource.org/license/bsd-3-clause/
*/
void vaadd_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x24002057, vm, vs2, vs1, vd); }
void vaadd_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x24006057, vm, vs2, rs1, vd); }
void vaaddu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x20002057, vm, vs2, vs1, vd); }
void vaaddu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x20006057, vm, vs2, rs1, vd); }
void vadc_vim(const VReg& vd, const VReg& vs2, int32_t simm5) { opIVI(0x40003057, 0, vs2, simm5, vd); }
void vadc_vvm(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x40000057, 0, vs2, vs1, vd); }
void vadc_vxm(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x40004057, 0, vs2, rs1, vd); }
void vadd_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x3057, vm, vs2, simm5, vd); }
void vadd_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x57, vm, vs2, vs1, vd); }
void vadd_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x4057, vm, vs2, rs1, vd); }
void vand_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x24003057, vm, vs2, simm5, vd); }
void vand_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x24000057, vm, vs2, vs1, vd); }
void vand_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x24004057, vm, vs2, rs1, vd); }
void vasub_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x2c002057, vm, vs2, vs1, vd); }
void vasub_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x2c006057, vm, vs2, rs1, vd); }
void vasubu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x28002057, vm, vs2, vs1, vd); }
void vasubu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x28006057, vm, vs2, rs1, vd); }
void vcompress_vm(const VReg& vd, const VReg& vs2, const VReg& vs1) { opMVV(0x5e002057, 0, vs2, vs1, vd); }
void vcpop_m(const Reg& rd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x40082057, vm, vs2, 0, rd); }
void vdiv_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x84002057, vm, vs2, vs1, vd); }
void vdiv_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x84006057, vm, vs2, rs1, vd); }
void vdivu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x80002057, vm, vs2, vs1, vd); }
void vdivu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x80006057, vm, vs2, rs1, vd); }
void vfadd_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x5057, vm, vs2, rs1, vd); }
void vfadd_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x1057, vm, vs2, vs1, vd); }
void vfclass_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x4c081057, vm, vs2, 0, vd); }
void vfcvt_f_x_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48019057, vm, vs2, 0, vd); }
void vfcvt_f_xu_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48011057, vm, vs2, 0, vd); }
void vfcvt_rtz_x_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48039057, vm, vs2, 0, vd); }
void vfcvt_rtz_xu_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48031057, vm, vs2, 0, vd); }
void vfcvt_x_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48009057, vm, vs2, 0, vd); }
void vfcvt_xu_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48001057, vm, vs2, 0, vd); }
void vfdiv_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x80005057, vm, vs2, rs1, vd); }
void vfdiv_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x80001057, vm, vs2, vs1, vd); }
void vfirst_m(const Reg& rd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x4008a057, vm, vs2, 0, rd); }
void vfmacc_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xb0005057, vm, vs2, rs1, vd); }
void vfmacc_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xb0001057, vm, vs2, vs1, vd); }
void vfmadd_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xa0005057, vm, vs2, rs1, vd); }
void vfmadd_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xa0001057, vm, vs2, vs1, vd); }
void vfmax_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x18005057, vm, vs2, rs1, vd); }
void vfmax_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x18001057, vm, vs2, vs1, vd); }
void vfmerge_vfm(const VReg& vd, const VReg& vs2, const FReg& rs1) { opFVF(0x5c005057, 0, vs2, rs1, vd); }
void vfmin_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x10005057, vm, vs2, rs1, vd); }
void vfmin_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x10001057, vm, vs2, vs1, vd); }
void vfmsac_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xb8005057, vm, vs2, rs1, vd); }
void vfmsac_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xb8001057, vm, vs2, vs1, vd); }
void vfmsub_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xa8005057, vm, vs2, rs1, vd); }
void vfmsub_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xa8001057, vm, vs2, vs1, vd); }
void vfmul_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x90005057, vm, vs2, rs1, vd); }
void vfmul_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x90001057, vm, vs2, vs1, vd); }
void vfmv_f_s(const FReg& rd, const VReg& vs2) { opFVV(0x42001057, 0, vs2, 0, rd); }
void vfmv_s_f(const VReg& vd, const FReg& rs1) { opFVF(0x42005057, 0, 0, rs1, vd); }
void vfmv_v_f(const VReg& vd, const FReg& rs1) { opFVF(0x5e005057, 0, 0, rs1, vd); }
void vfncvt_f_f_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x480a1057, vm, vs2, 0, vd); }
void vfncvt_f_x_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48099057, vm, vs2, 0, vd); }
void vfncvt_f_xu_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48091057, vm, vs2, 0, vd); }
void vfncvt_rod_f_f_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x480a9057, vm, vs2, 0, vd); }
void vfncvt_rtz_x_f_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x480b9057, vm, vs2, 0, vd); }
void vfncvt_rtz_xu_f_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x480b1057, vm, vs2, 0, vd); }
void vfncvt_x_f_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48089057, vm, vs2, 0, vd); }
void vfncvt_xu_f_w(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48081057, vm, vs2, 0, vd); }
void vfnmacc_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xb4005057, vm, vs2, rs1, vd); }
void vfnmacc_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xb4001057, vm, vs2, vs1, vd); }
void vfnmadd_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xa4005057, vm, vs2, rs1, vd); }
void vfnmadd_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xa4001057, vm, vs2, vs1, vd); }
void vfnmsac_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xbc005057, vm, vs2, rs1, vd); }
void vfnmsac_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xbc001057, vm, vs2, vs1, vd); }
void vfnmsub_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xac005057, vm, vs2, rs1, vd); }
void vfnmsub_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xac001057, vm, vs2, vs1, vd); }
void vfrdiv_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x84005057, vm, vs2, rs1, vd); }
void vfrec7_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x4c029057, vm, vs2, 0, vd); }
void vfredmax_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x1c001057, vm, vs2, vs1, vd); }
void vfredmin_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x14001057, vm, vs2, vs1, vd); }
void vfredosum_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xc001057, vm, vs2, vs1, vd); }
void vfredusum_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x4001057, vm, vs2, vs1, vd); }
void vfrsqrt7_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x4c021057, vm, vs2, 0, vd); }
void vfrsub_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x9c005057, vm, vs2, rs1, vd); }
void vfsgnj_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x20005057, vm, vs2, rs1, vd); }
void vfsgnj_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x20001057, vm, vs2, vs1, vd); }
void vfsgnjn_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x24005057, vm, vs2, rs1, vd); }
void vfsgnjn_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x24001057, vm, vs2, vs1, vd); }
void vfsgnjx_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x28005057, vm, vs2, rs1, vd); }
void vfsgnjx_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x28001057, vm, vs2, vs1, vd); }
void vfslide1down_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x3c005057, vm, vs2, rs1, vd); }
void vfslide1up_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x38005057, vm, vs2, rs1, vd); }
void vfsqrt_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x4c001057, vm, vs2, 0, vd); }
void vfsub_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x8005057, vm, vs2, rs1, vd); }
void vfsub_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x8001057, vm, vs2, vs1, vd); }
void vfwadd_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0xc0005057, vm, vs2, rs1, vd); }
void vfwadd_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xc0001057, vm, vs2, vs1, vd); }
void vfwadd_wf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0xd0005057, vm, vs2, rs1, vd); }
void vfwadd_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xd0001057, vm, vs2, vs1, vd); }
void vfwcvt_f_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48061057, vm, vs2, 0, vd); }
void vfwcvt_f_x_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48059057, vm, vs2, 0, vd); }
void vfwcvt_f_xu_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48051057, vm, vs2, 0, vd); }
void vfwcvt_rtz_x_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48079057, vm, vs2, 0, vd); }
void vfwcvt_rtz_xu_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48071057, vm, vs2, 0, vd); }
void vfwcvt_x_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48049057, vm, vs2, 0, vd); }
void vfwcvt_xu_f_v(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0x48041057, vm, vs2, 0, vd); }
void vfwmacc_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xf0005057, vm, vs2, rs1, vd); }
void vfwmacc_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xf0001057, vm, vs2, vs1, vd); }
void vfwmsac_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xf8005057, vm, vs2, rs1, vd); }
void vfwmsac_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xf8001057, vm, vs2, vs1, vd); }
void vfwmul_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0xe0005057, vm, vs2, rs1, vd); }
void vfwmul_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xe0001057, vm, vs2, vs1, vd); }
void vfwnmacc_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xf4005057, vm, vs2, rs1, vd); }
void vfwnmacc_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xf4001057, vm, vs2, vs1, vd); }
void vfwnmsac_vf(const VReg& vd, const FReg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opFVF(0xfc005057, vm, vs2, rs1, vd); }
void vfwnmsac_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opFVV(0xfc001057, vm, vs2, vs1, vd); }
void vfwredosum_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xcc001057, vm, vs2, vs1, vd); }
void vfwredusum_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xc4001057, vm, vs2, vs1, vd); }
void vfwsub_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0xc8005057, vm, vs2, rs1, vd); }
void vfwsub_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xc8001057, vm, vs2, vs1, vd); }
void vfwsub_wf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0xd8005057, vm, vs2, rs1, vd); }
void vfwsub_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0xd8001057, vm, vs2, vs1, vd); }
void vid_v(const VReg& vd, VM vm=VM::unmasked) { opMVV(0x5008a057, vm, 0, 0, vd); }
void viota_m(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x50082057, vm, vs2, 0, vd); }
void vl1re16_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2805007, 0, 0, rs1, vd); }
void vl1re32_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2806007, 0, 0, rs1, vd); }
void vl1re64_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2807007, 0, 0, rs1, vd); }
void vl1re8_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2800007, 0, 0, rs1, vd); }
void vl2re16_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2805007, 0, 0, rs1, vd); }
void vl2re32_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2806007, 0, 0, rs1, vd); }
void vl2re64_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2807007, 0, 0, rs1, vd); }
void vl2re8_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2800007, 0, 0, rs1, vd); }
void vl4re16_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2805007, 0, 0, rs1, vd); }
void vl4re32_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2806007, 0, 0, rs1, vd); }
void vl4re64_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2807007, 0, 0, rs1, vd); }
void vl4re8_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2800007, 0, 0, rs1, vd); }
void vl8re16_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2805007, 0, 0, rs1, vd); }
void vl8re32_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2806007, 0, 0, rs1, vd); }
void vl8re64_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2807007, 0, 0, rs1, vd); }
void vl8re8_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2800007, 0, 0, rs1, vd); }
void vlseg1e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10007007, vm, 0, rs1, vd); }
void vlseg2e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x30007007, vm, 0, rs1, vd); }
void vlseg3e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x50007007, vm, 0, rs1, vd); }
void vlseg4e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x70007007, vm, 0, rs1, vd); }
void vlseg5e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x90007007, vm, 0, rs1, vd); }
void vlseg6e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb0007007, vm, 0, rs1, vd); }
void vlseg7e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd0007007, vm, 0, rs1, vd); }
void vlseg8e1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf0007007, vm, 0, rs1, vd); }
void vle1024_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10007007, vm, 0, rs1, vd); }
void vlseg1e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11007007, vm, 0, rs1, vd); }
void vlseg2e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x31007007, vm, 0, rs1, vd); }
void vlseg3e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x51007007, vm, 0, rs1, vd); }
void vlseg4e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x71007007, vm, 0, rs1, vd); }
void vlseg5e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x91007007, vm, 0, rs1, vd); }
void vlseg6e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb1007007, vm, 0, rs1, vd); }
void vlseg7e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd1007007, vm, 0, rs1, vd); }
void vlseg8e1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf1007007, vm, 0, rs1, vd); }
void vle1024ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11007007, vm, 0, rs1, vd); }
void vlseg1e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10000007, vm, 0, rs1, vd); }
void vlseg2e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x30000007, vm, 0, rs1, vd); }
void vlseg3e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x50000007, vm, 0, rs1, vd); }
void vlseg4e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x70000007, vm, 0, rs1, vd); }
void vlseg5e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x90000007, vm, 0, rs1, vd); }
void vlseg6e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb0000007, vm, 0, rs1, vd); }
void vlseg7e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd0000007, vm, 0, rs1, vd); }
void vlseg8e128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf0000007, vm, 0, rs1, vd); }
void vle128_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10000007, vm, 0, rs1, vd); }
void vlseg1e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11000007, vm, 0, rs1, vd); }
void vlseg2e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x31000007, vm, 0, rs1, vd); }
void vlseg3e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x51000007, vm, 0, rs1, vd); }
void vlseg4e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x71000007, vm, 0, rs1, vd); }
void vlseg5e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x91000007, vm, 0, rs1, vd); }
void vlseg6e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb1000007, vm, 0, rs1, vd); }
void vlseg7e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd1000007, vm, 0, rs1, vd); }
void vlseg8e128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf1000007, vm, 0, rs1, vd); }
void vle128ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11000007, vm, 0, rs1, vd); }
void vlseg1e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x5007, vm, 0, rs1, vd); }
void vlseg2e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x20005007, vm, 0, rs1, vd); }
void vlseg3e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x40005007, vm, 0, rs1, vd); }
void vlseg4e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x60005007, vm, 0, rs1, vd); }
void vlseg5e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x80005007, vm, 0, rs1, vd); }
void vlseg6e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa0005007, vm, 0, rs1, vd); }
void vlseg7e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc0005007, vm, 0, rs1, vd); }
void vlseg8e16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe0005007, vm, 0, rs1, vd); }
void vle16_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x5007, vm, 0, rs1, vd); }
void vlseg1e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1005007, vm, 0, rs1, vd); }
void vlseg2e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x21005007, vm, 0, rs1, vd); }
void vlseg3e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x41005007, vm, 0, rs1, vd); }
void vlseg4e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x61005007, vm, 0, rs1, vd); }
void vlseg5e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x81005007, vm, 0, rs1, vd); }
void vlseg6e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa1005007, vm, 0, rs1, vd); }
void vlseg7e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc1005007, vm, 0, rs1, vd); }
void vlseg8e16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe1005007, vm, 0, rs1, vd); }
void vle16ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1005007, vm, 0, rs1, vd); }
void vlseg1e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10005007, vm, 0, rs1, vd); }
void vlseg2e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x30005007, vm, 0, rs1, vd); }
void vlseg3e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x50005007, vm, 0, rs1, vd); }
void vlseg4e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x70005007, vm, 0, rs1, vd); }
void vlseg5e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x90005007, vm, 0, rs1, vd); }
void vlseg6e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb0005007, vm, 0, rs1, vd); }
void vlseg7e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd0005007, vm, 0, rs1, vd); }
void vlseg8e256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf0005007, vm, 0, rs1, vd); }
void vle256_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10005007, vm, 0, rs1, vd); }
void vlseg1e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11005007, vm, 0, rs1, vd); }
void vlseg2e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x31005007, vm, 0, rs1, vd); }
void vlseg3e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x51005007, vm, 0, rs1, vd); }
void vlseg4e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x71005007, vm, 0, rs1, vd); }
void vlseg5e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x91005007, vm, 0, rs1, vd); }
void vlseg6e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb1005007, vm, 0, rs1, vd); }
void vlseg7e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd1005007, vm, 0, rs1, vd); }
void vlseg8e256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf1005007, vm, 0, rs1, vd); }
void vle256ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11005007, vm, 0, rs1, vd); }
void vlseg1e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x6007, vm, 0, rs1, vd); }
void vlseg2e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x20006007, vm, 0, rs1, vd); }
void vlseg3e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x40006007, vm, 0, rs1, vd); }
void vlseg4e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x60006007, vm, 0, rs1, vd); }
void vlseg5e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x80006007, vm, 0, rs1, vd); }
void vlseg6e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa0006007, vm, 0, rs1, vd); }
void vlseg7e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc0006007, vm, 0, rs1, vd); }
void vlseg8e32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe0006007, vm, 0, rs1, vd); }
void vle32_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x6007, vm, 0, rs1, vd); }
void vlseg1e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1006007, vm, 0, rs1, vd); }
void vlseg2e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x21006007, vm, 0, rs1, vd); }
void vlseg3e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x41006007, vm, 0, rs1, vd); }
void vlseg4e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x61006007, vm, 0, rs1, vd); }
void vlseg5e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x81006007, vm, 0, rs1, vd); }
void vlseg6e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa1006007, vm, 0, rs1, vd); }
void vlseg7e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc1006007, vm, 0, rs1, vd); }
void vlseg8e32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe1006007, vm, 0, rs1, vd); }
void vle32ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1006007, vm, 0, rs1, vd); }
void vlseg1e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10006007, vm, 0, rs1, vd); }
void vlseg2e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x30006007, vm, 0, rs1, vd); }
void vlseg3e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x50006007, vm, 0, rs1, vd); }
void vlseg4e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x70006007, vm, 0, rs1, vd); }
void vlseg5e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x90006007, vm, 0, rs1, vd); }
void vlseg6e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb0006007, vm, 0, rs1, vd); }
void vlseg7e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd0006007, vm, 0, rs1, vd); }
void vlseg8e512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf0006007, vm, 0, rs1, vd); }
void vle512_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x10006007, vm, 0, rs1, vd); }
void vlseg1e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11006007, vm, 0, rs1, vd); }
void vlseg2e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x31006007, vm, 0, rs1, vd); }
void vlseg3e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x51006007, vm, 0, rs1, vd); }
void vlseg4e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x71006007, vm, 0, rs1, vd); }
void vlseg5e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x91006007, vm, 0, rs1, vd); }
void vlseg6e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xb1006007, vm, 0, rs1, vd); }
void vlseg7e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xd1006007, vm, 0, rs1, vd); }
void vlseg8e512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xf1006007, vm, 0, rs1, vd); }
void vle512ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x11006007, vm, 0, rs1, vd); }
void vlseg1e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x7007, vm, 0, rs1, vd); }
void vlseg2e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x20007007, vm, 0, rs1, vd); }
void vlseg3e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x40007007, vm, 0, rs1, vd); }
void vlseg4e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x60007007, vm, 0, rs1, vd); }
void vlseg5e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x80007007, vm, 0, rs1, vd); }
void vlseg6e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa0007007, vm, 0, rs1, vd); }
void vlseg7e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc0007007, vm, 0, rs1, vd); }
void vlseg8e64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe0007007, vm, 0, rs1, vd); }
void vle64_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x7007, vm, 0, rs1, vd); }
void vlseg1e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1007007, vm, 0, rs1, vd); }
void vlseg2e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x21007007, vm, 0, rs1, vd); }
void vlseg3e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x41007007, vm, 0, rs1, vd); }
void vlseg4e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x61007007, vm, 0, rs1, vd); }
void vlseg5e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x81007007, vm, 0, rs1, vd); }
void vlseg6e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa1007007, vm, 0, rs1, vd); }
void vlseg7e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc1007007, vm, 0, rs1, vd); }
void vlseg8e64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe1007007, vm, 0, rs1, vd); }
void vle64ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1007007, vm, 0, rs1, vd); }
void vlseg1e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x7, vm, 0, rs1, vd); }
void vlseg2e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x20000007, vm, 0, rs1, vd); }
void vlseg3e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x40000007, vm, 0, rs1, vd); }
void vlseg4e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x60000007, vm, 0, rs1, vd); }
void vlseg5e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x80000007, vm, 0, rs1, vd); }
void vlseg6e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa0000007, vm, 0, rs1, vd); }
void vlseg7e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc0000007, vm, 0, rs1, vd); }
void vlseg8e8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe0000007, vm, 0, rs1, vd); }
void vle8_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x7, vm, 0, rs1, vd); }
void vlseg1e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1000007, vm, 0, rs1, vd); }
void vlseg2e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x21000007, vm, 0, rs1, vd); }
void vlseg3e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x41000007, vm, 0, rs1, vd); }
void vlseg4e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x61000007, vm, 0, rs1, vd); }
void vlseg5e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x81000007, vm, 0, rs1, vd); }
void vlseg6e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xa1000007, vm, 0, rs1, vd); }
void vlseg7e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xc1000007, vm, 0, rs1, vd); }
void vlseg8e8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0xe1000007, vm, 0, rs1, vd); }
void vle8ff_v(const VReg& vd, const Reg& rs1, VM vm=VM::unmasked) { opVectorLoad(0x1000007, vm, 0, rs1, vd); }
void vlm_v(const VReg& vd, const Reg& rs1) { opVectorLoad(0x2b00007, 0, 0, rs1, vd); }
void vloxei1024_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x1c007007, vm, vs2, rs1, vd); }
void vloxei128_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x1c000007, vm, vs2, rs1, vd); }
void vloxei16_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0xc005007, vm, vs2, rs1, vd); }
void vloxei256_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x1c005007, vm, vs2, rs1, vd); }
void vloxei32_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0xc006007, vm, vs2, rs1, vd); }
void vloxei512_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x1c006007, vm, vs2, rs1, vd); }
void vloxei64_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0xc007007, vm, vs2, rs1, vd); }
void vloxei8_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0xc000007, vm, vs2, rs1, vd); }
void vlsseg1e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18007007, vm, rs2, rs1, vd); }
void vlsseg2e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x38007007, vm, rs2, rs1, vd); }
void vlsseg3e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x58007007, vm, rs2, rs1, vd); }
void vlsseg4e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x78007007, vm, rs2, rs1, vd); }
void vlsseg5e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x98007007, vm, rs2, rs1, vd); }
void vlsseg6e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xb8007007, vm, rs2, rs1, vd); }
void vlsseg7e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xd8007007, vm, rs2, rs1, vd); }
void vlsseg8e1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xf8007007, vm, rs2, rs1, vd); }
void vlse1024_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18007007, vm, rs2, rs1, vd); }
void vlsseg1e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18000007, vm, rs2, rs1, vd); }
void vlsseg2e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x38000007, vm, rs2, rs1, vd); }
void vlsseg3e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x58000007, vm, rs2, rs1, vd); }
void vlsseg4e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x78000007, vm, rs2, rs1, vd); }
void vlsseg5e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x98000007, vm, rs2, rs1, vd); }
void vlsseg6e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xb8000007, vm, rs2, rs1, vd); }
void vlsseg7e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xd8000007, vm, rs2, rs1, vd); }
void vlsseg8e128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xf8000007, vm, rs2, rs1, vd); }
void vlse128_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18000007, vm, rs2, rs1, vd); }
void vlsseg1e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8005007, vm, rs2, rs1, vd); }
void vlsseg2e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x28005007, vm, rs2, rs1, vd); }
void vlsseg3e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x48005007, vm, rs2, rs1, vd); }
void vlsseg4e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x68005007, vm, rs2, rs1, vd); }
void vlsseg5e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x88005007, vm, rs2, rs1, vd); }
void vlsseg6e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xa8005007, vm, rs2, rs1, vd); }
void vlsseg7e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xc8005007, vm, rs2, rs1, vd); }
void vlsseg8e16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xe8005007, vm, rs2, rs1, vd); }
void vlse16_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8005007, vm, rs2, rs1, vd); }
void vlsseg1e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18005007, vm, rs2, rs1, vd); }
void vlsseg2e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x38005007, vm, rs2, rs1, vd); }
void vlsseg3e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x58005007, vm, rs2, rs1, vd); }
void vlsseg4e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x78005007, vm, rs2, rs1, vd); }
void vlsseg5e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x98005007, vm, rs2, rs1, vd); }
void vlsseg6e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xb8005007, vm, rs2, rs1, vd); }
void vlsseg7e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xd8005007, vm, rs2, rs1, vd); }
void vlsseg8e256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xf8005007, vm, rs2, rs1, vd); }
void vlse256_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18005007, vm, rs2, rs1, vd); }
void vlsseg1e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8006007, vm, rs2, rs1, vd); }
void vlsseg2e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x28006007, vm, rs2, rs1, vd); }
void vlsseg3e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x48006007, vm, rs2, rs1, vd); }
void vlsseg4e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x68006007, vm, rs2, rs1, vd); }
void vlsseg5e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x88006007, vm, rs2, rs1, vd); }
void vlsseg6e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xa8006007, vm, rs2, rs1, vd); }
void vlsseg7e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xc8006007, vm, rs2, rs1, vd); }
void vlsseg8e32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xe8006007, vm, rs2, rs1, vd); }
void vlse32_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8006007, vm, rs2, rs1, vd); }
void vlsseg1e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18006007, vm, rs2, rs1, vd); }
void vlsseg2e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x38006007, vm, rs2, rs1, vd); }
void vlsseg3e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x58006007, vm, rs2, rs1, vd); }
void vlsseg4e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x78006007, vm, rs2, rs1, vd); }
void vlsseg5e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x98006007, vm, rs2, rs1, vd); }
void vlsseg6e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xb8006007, vm, rs2, rs1, vd); }
void vlsseg7e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xd8006007, vm, rs2, rs1, vd); }
void vlsseg8e512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xf8006007, vm, rs2, rs1, vd); }
void vlse512_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x18006007, vm, rs2, rs1, vd); }
void vlsseg1e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8007007, vm, rs2, rs1, vd); }
void vlsseg2e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x28007007, vm, rs2, rs1, vd); }
void vlsseg3e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x48007007, vm, rs2, rs1, vd); }
void vlsseg4e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x68007007, vm, rs2, rs1, vd); }
void vlsseg5e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x88007007, vm, rs2, rs1, vd); }
void vlsseg6e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xa8007007, vm, rs2, rs1, vd); }
void vlsseg7e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xc8007007, vm, rs2, rs1, vd); }
void vlsseg8e64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xe8007007, vm, rs2, rs1, vd); }
void vlse64_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8007007, vm, rs2, rs1, vd); }
void vlsseg1e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8000007, vm, rs2, rs1, vd); }
void vlsseg2e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x28000007, vm, rs2, rs1, vd); }
void vlsseg3e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x48000007, vm, rs2, rs1, vd); }
void vlsseg4e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x68000007, vm, rs2, rs1, vd); }
void vlsseg5e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x88000007, vm, rs2, rs1, vd); }
void vlsseg6e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xa8000007, vm, rs2, rs1, vd); }
void vlsseg7e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xc8000007, vm, rs2, rs1, vd); }
void vlsseg8e8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0xe8000007, vm, rs2, rs1, vd); }
void vlse8_v(const VReg& vd, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorLoad(0x8000007, vm, rs2, rs1, vd); }
void vluxei1024_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x14007007, vm, vs2, rs1, vd); }
void vluxei128_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x14000007, vm, vs2, rs1, vd); }
void vluxei16_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x4005007, vm, vs2, rs1, vd); }
void vluxei256_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x14005007, vm, vs2, rs1, vd); }
void vluxei32_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x4006007, vm, vs2, rs1, vd); }
void vluxei512_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x14006007, vm, vs2, rs1, vd); }
void vluxei64_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x4007007, vm, vs2, rs1, vd); }
void vluxei8_v(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorLoad(0x4000007, vm, vs2, rs1, vd); }
void vmacc_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xb4002057, vm, vs2, vs1, vd); }
void vmacc_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xb4006057, vm, vs2, rs1, vd); }
void vmadc_vi(const VReg& vd, const VReg& vs2, int32_t simm5) { opIVI(0x46003057, 0, vs2, simm5, vd); }
void vmadc_vim(const VReg& vd, const VReg& vs2, int32_t simm5) { opIVI(0x44003057, 0, vs2, simm5, vd); }
void vmadc_vv(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x46000057, 0, vs2, vs1, vd); }
void vmadc_vvm(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x44000057, 0, vs2, vs1, vd); }
void vmadc_vx(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x46004057, 0, vs2, rs1, vd); }
void vmadc_vxm(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x44004057, 0, vs2, rs1, vd); }
void vmadd_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xa4002057, vm, vs2, vs1, vd); }
void vmadd_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xa4006057, vm, vs2, rs1, vd); }
void vmand_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x64002057, vm, vs2, vs1, vd); }
void vmandn_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x60002057, vm, vs2, vs1, vd); }
void vmax_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x1c000057, vm, vs2, vs1, vd); }
void vmax_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x1c004057, vm, vs2, rs1, vd); }
void vmaxu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x18000057, vm, vs2, vs1, vd); }
void vmaxu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x18004057, vm, vs2, rs1, vd); }
void vmerge_vim(const VReg& vd, const VReg& vs2, int32_t simm5) { opIVI(0x5c003057, 0, vs2, simm5, vd); }
void vmerge_vvm(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x5c000057, 0, vs2, vs1, vd); }
void vmerge_vxm(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x5c004057, 0, vs2, rs1, vd); }
void vmfeq_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x60005057, vm, vs2, rs1, vd); }
void vmfeq_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x60001057, vm, vs2, vs1, vd); }
void vmfge_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x7c005057, vm, vs2, rs1, vd); }
void vmfgt_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x74005057, vm, vs2, rs1, vd); }
void vmfle_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x64005057, vm, vs2, rs1, vd); }
void vmfle_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x64001057, vm, vs2, vs1, vd); }
void vmflt_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x6c005057, vm, vs2, rs1, vd); }
void vmflt_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x6c001057, vm, vs2, vs1, vd); }
void vmfne_vf(const VReg& vd, const VReg& vs2, const FReg& rs1, VM vm=VM::unmasked) { opFVF(0x70005057, vm, vs2, rs1, vd); }
void vmfne_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opFVV(0x70001057, vm, vs2, vs1, vd); }
void vmin_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x14000057, vm, vs2, vs1, vd); }
void vmin_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x14004057, vm, vs2, rs1, vd); }
void vminu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x10000057, vm, vs2, vs1, vd); }
void vminu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x10004057, vm, vs2, rs1, vd); }
void vmnand_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x74002057, vm, vs2, vs1, vd); }
void vmnor_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x78002057, vm, vs2, vs1, vd); }
void vmor_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x68002057, vm, vs2, vs1, vd); }
void vmorn_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x70002057, vm, vs2, vs1, vd); }
void vmsbc_vv(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x4e000057, 0, vs2, vs1, vd); }
void vmsbc_vvm(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x4c000057, 0, vs2, vs1, vd); }
void vmsbc_vx(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x4e004057, 0, vs2, rs1, vd); }
void vmsbc_vxm(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x4c004057, 0, vs2, rs1, vd); }
void vmsbf_m(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x5000a057, vm, vs2, 0, vd); }
void vmseq_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x60003057, vm, vs2, simm5, vd); }
void vmseq_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x60000057, vm, vs2, vs1, vd); }
void vmseq_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x60004057, vm, vs2, rs1, vd); }
void vmsgt_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x7c003057, vm, vs2, simm5, vd); }
void vmsgt_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x7c004057, vm, vs2, rs1, vd); }
void vmsgtu_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x78003057, vm, vs2, simm5, vd); }
void vmsgtu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x78004057, vm, vs2, rs1, vd); }
void vmsif_m(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x5001a057, vm, vs2, 0, vd); }
void vmsle_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x74003057, vm, vs2, simm5, vd); }
void vmsle_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x74000057, vm, vs2, vs1, vd); }
void vmsle_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x74004057, vm, vs2, rs1, vd); }
void vmsleu_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x70003057, vm, vs2, simm5, vd); }
void vmsleu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x70000057, vm, vs2, vs1, vd); }
void vmsleu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x70004057, vm, vs2, rs1, vd); }
void vmslt_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x6c000057, vm, vs2, vs1, vd); }
void vmslt_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x6c004057, vm, vs2, rs1, vd); }
void vmsltu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x68000057, vm, vs2, vs1, vd); }
void vmsltu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x68004057, vm, vs2, rs1, vd); }
void vmsne_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x64003057, vm, vs2, simm5, vd); }
void vmsne_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x64000057, vm, vs2, vs1, vd); }
void vmsne_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x64004057, vm, vs2, rs1, vd); }
void vmsof_m(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x50012057, vm, vs2, 0, vd); }
void vmul_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x94002057, vm, vs2, vs1, vd); }
void vmul_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x94006057, vm, vs2, rs1, vd); }
void vmulh_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x9c002057, vm, vs2, vs1, vd); }
void vmulh_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x9c006057, vm, vs2, rs1, vd); }
void vmulhsu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x98002057, vm, vs2, vs1, vd); }
void vmulhsu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x98006057, vm, vs2, rs1, vd); }
void vmulhu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x90002057, vm, vs2, vs1, vd); }
void vmulhu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x90006057, vm, vs2, rs1, vd); }
void vmv1r_v(const VReg& vd, const VReg& vs2) { opIVI(0x9e003057, 0, vs2, 0, vd); }
void vmv2r_v(const VReg& vd, const VReg& vs2) { opIVI(0x9e00b057, 0, vs2, 0, vd); }
void vmv4r_v(const VReg& vd, const VReg& vs2) { opIVI(0x9e01b057, 0, vs2, 0, vd); }
void vmv8r_v(const VReg& vd, const VReg& vs2) { opIVI(0x9e03b057, 0, vs2, 0, vd); }
void vmv_s_x(const VReg& vd, const Reg& rs1) { opMVX(0x42006057, 0, 0, rs1, vd); }
void vmv_v_i(const VReg& vd, int32_t simm5) { opIVI(0x5e003057, 0, 0, simm5, vd); }
void vmv_v_v(const VReg& vd, const VReg& vs1) { opIVV(0x5e000057, 0, 0, vs1, vd); }
void vmv_v_x(const VReg& vd, const Reg& rs1) { opIVX(0x5e004057, 0, 0, rs1, vd); }
void vmv_x_s(const Reg& rd, const VReg& vs2) { opMVV(0x42002057, 0, vs2, 0, rd); }
void vmxnor_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x7c002057, vm, vs2, vs1, vd); }
void vmxor_mm(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x6c002057, vm, vs2, vs1, vd); }
void vnclip_wi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xbc003057, vm, vs2, simm5, vd); }
void vnclip_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xbc000057, vm, vs2, vs1, vd); }
void vnclip_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xbc004057, vm, vs2, rs1, vd); }
void vnclipu_wi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xb8003057, vm, vs2, simm5, vd); }
void vnclipu_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xb8000057, vm, vs2, vs1, vd); }
void vnclipu_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xb8004057, vm, vs2, rs1, vd); }
void vnmsac_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xbc002057, vm, vs2, vs1, vd); }
void vnmsac_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xbc006057, vm, vs2, rs1, vd); }
void vnmsub_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xac002057, vm, vs2, vs1, vd); }
void vnmsub_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xac006057, vm, vs2, rs1, vd); }
void vnsra_wi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xb4003057, vm, vs2, simm5, vd); }
void vnsra_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xb4000057, vm, vs2, vs1, vd); }
void vnsra_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xb4004057, vm, vs2, rs1, vd); }
void vnsrl_wi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xb0003057, vm, vs2, simm5, vd); }
void vnsrl_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xb0000057, vm, vs2, vs1, vd); }
void vnsrl_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xb0004057, vm, vs2, rs1, vd); }
void vor_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x28003057, vm, vs2, simm5, vd); }
void vor_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x28000057, vm, vs2, vs1, vd); }
void vor_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x28004057, vm, vs2, rs1, vd); }
void vredand_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x4002057, vm, vs2, vs1, vd); }
void vredmax_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x1c002057, vm, vs2, vs1, vd); }
void vredmaxu_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x18002057, vm, vs2, vs1, vd); }
void vredmin_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x14002057, vm, vs2, vs1, vd); }
void vredminu_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x10002057, vm, vs2, vs1, vd); }
void vredor_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x8002057, vm, vs2, vs1, vd); }
void vredsum_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x2057, vm, vs2, vs1, vd); }
void vredxor_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xc002057, vm, vs2, vs1, vd); }
void vrem_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x8c002057, vm, vs2, vs1, vd); }
void vrem_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x8c006057, vm, vs2, rs1, vd); }
void vremu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0x88002057, vm, vs2, vs1, vd); }
void vremu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x88006057, vm, vs2, rs1, vd); }
void vrgather_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x30003057, vm, vs2, simm5, vd); }
void vrgather_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x30000057, vm, vs2, vs1, vd); }
void vrgather_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x30004057, vm, vs2, rs1, vd); }
void vrgatherei16_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x38000057, vm, vs2, vs1, vd); }
void vrsub_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xc003057, vm, vs2, simm5, vd); }
void vrsub_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xc004057, vm, vs2, rs1, vd); }
void vs1r_v(VReg vs3, const Reg& rs1) { opVectorStore(0x2800027, 0, 0, rs1, vs3); }
void vs2r_v(VReg vs3, const Reg& rs1) { opVectorStore(0x2800027, 0, 0, rs1, vs3); }
void vs4r_v(VReg vs3, const Reg& rs1) { opVectorStore(0x2800027, 0, 0, rs1, vs3); }
void vs8r_v(VReg vs3, const Reg& rs1) { opVectorStore(0x2800027, 0, 0, rs1, vs3); }
void vsadd_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x84003057, vm, vs2, simm5, vd); }
void vsadd_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x84000057, vm, vs2, vs1, vd); }
void vsadd_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x84004057, vm, vs2, rs1, vd); }
void vsaddu_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x80003057, vm, vs2, simm5, vd); }
void vsaddu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x80000057, vm, vs2, vs1, vd); }
void vsaddu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x80004057, vm, vs2, rs1, vd); }
void vsbc_vvm(const VReg& vd, const VReg& vs2, const VReg& vs1) { opIVV(0x48000057, 0, vs2, vs1, vd); }
void vsbc_vxm(const VReg& vd, const VReg& vs2, const Reg& rs1) { opIVX(0x48004057, 0, vs2, rs1, vd); }
void vsseg1e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10007027, vm, 0, rs1, vs3); }
void vsseg2e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x30007027, vm, 0, rs1, vs3); }
void vsseg3e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x50007027, vm, 0, rs1, vs3); }
void vsseg4e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x70007027, vm, 0, rs1, vs3); }
void vsseg5e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x90007027, vm, 0, rs1, vs3); }
void vsseg6e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xb0007027, vm, 0, rs1, vs3); }
void vsseg7e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xd0007027, vm, 0, rs1, vs3); }
void vsseg8e1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xf0007027, vm, 0, rs1, vs3); }
void vse1024_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10007027, vm, 0, rs1, vs3); }
void vsseg1e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10000027, vm, 0, rs1, vs3); }
void vsseg2e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x30000027, vm, 0, rs1, vs3); }
void vsseg3e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x50000027, vm, 0, rs1, vs3); }
void vsseg4e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x70000027, vm, 0, rs1, vs3); }
void vsseg5e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x90000027, vm, 0, rs1, vs3); }
void vsseg6e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xb0000027, vm, 0, rs1, vs3); }
void vsseg7e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xd0000027, vm, 0, rs1, vs3); }
void vsseg8e128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xf0000027, vm, 0, rs1, vs3); }
void vse128_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10000027, vm, 0, rs1, vs3); }
void vsseg1e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x5027, vm, 0, rs1, vs3); }
void vsseg2e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x20005027, vm, 0, rs1, vs3); }
void vsseg3e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x40005027, vm, 0, rs1, vs3); }
void vsseg4e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x60005027, vm, 0, rs1, vs3); }
void vsseg5e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x80005027, vm, 0, rs1, vs3); }
void vsseg6e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xa0005027, vm, 0, rs1, vs3); }
void vsseg7e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xc0005027, vm, 0, rs1, vs3); }
void vsseg8e16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xe0005027, vm, 0, rs1, vs3); }
void vse16_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x5027, vm, 0, rs1, vs3); }
void vsseg1e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10005027, vm, 0, rs1, vs3); }
void vsseg2e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x30005027, vm, 0, rs1, vs3); }
void vsseg3e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x50005027, vm, 0, rs1, vs3); }
void vsseg4e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x70005027, vm, 0, rs1, vs3); }
void vsseg5e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x90005027, vm, 0, rs1, vs3); }
void vsseg6e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xb0005027, vm, 0, rs1, vs3); }
void vsseg7e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xd0005027, vm, 0, rs1, vs3); }
void vsseg8e256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xf0005027, vm, 0, rs1, vs3); }
void vse256_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10005027, vm, 0, rs1, vs3); }
void vsseg1e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x6027, vm, 0, rs1, vs3); }
void vsseg2e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x20006027, vm, 0, rs1, vs3); }
void vsseg3e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x40006027, vm, 0, rs1, vs3); }
void vsseg4e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x60006027, vm, 0, rs1, vs3); }
void vsseg5e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x80006027, vm, 0, rs1, vs3); }
void vsseg6e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xa0006027, vm, 0, rs1, vs3); }
void vsseg7e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xc0006027, vm, 0, rs1, vs3); }
void vsseg8e32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xe0006027, vm, 0, rs1, vs3); }
void vse32_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x6027, vm, 0, rs1, vs3); }
void vsseg1e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10006027, vm, 0, rs1, vs3); }
void vsseg2e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x30006027, vm, 0, rs1, vs3); }
void vsseg3e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x50006027, vm, 0, rs1, vs3); }
void vsseg4e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x70006027, vm, 0, rs1, vs3); }
void vsseg5e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x90006027, vm, 0, rs1, vs3); }
void vsseg6e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xb0006027, vm, 0, rs1, vs3); }
void vsseg7e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xd0006027, vm, 0, rs1, vs3); }
void vsseg8e512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xf0006027, vm, 0, rs1, vs3); }
void vse512_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x10006027, vm, 0, rs1, vs3); }
void vsseg1e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x7027, vm, 0, rs1, vs3); }
void vsseg2e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x20007027, vm, 0, rs1, vs3); }
void vsseg3e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x40007027, vm, 0, rs1, vs3); }
void vsseg4e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x60007027, vm, 0, rs1, vs3); }
void vsseg5e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x80007027, vm, 0, rs1, vs3); }
void vsseg6e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xa0007027, vm, 0, rs1, vs3); }
void vsseg7e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xc0007027, vm, 0, rs1, vs3); }
void vsseg8e64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xe0007027, vm, 0, rs1, vs3); }
void vse64_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x7027, vm, 0, rs1, vs3); }
void vsseg1e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x27, vm, 0, rs1, vs3); }
void vsseg2e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x20000027, vm, 0, rs1, vs3); }
void vsseg3e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x40000027, vm, 0, rs1, vs3); }
void vsseg4e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x60000027, vm, 0, rs1, vs3); }
void vsseg5e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x80000027, vm, 0, rs1, vs3); }
void vsseg6e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xa0000027, vm, 0, rs1, vs3); }
void vsseg7e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xc0000027, vm, 0, rs1, vs3); }
void vsseg8e8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0xe0000027, vm, 0, rs1, vs3); }
void vse8_v(VReg vs3, const Reg& rs1, VM vm=VM::unmasked) { opVectorStore(0x27, vm, 0, rs1, vs3); }
void vsext_vf2(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x4803a057, vm, vs2, 0, vd); }
void vsext_vf4(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x4802a057, vm, vs2, 0, vd); }
void vsext_vf8(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x4801a057, vm, vs2, 0, vd); }
void vslide1down_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x3c006057, vm, vs2, rs1, vd); }
void vslide1up_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0x38006057, vm, vs2, rs1, vd); }
void vslidedown_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x3c003057, vm, vs2, simm5, vd); }
void vslidedown_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x3c004057, vm, vs2, rs1, vd); }
void vslideup_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x38003057, vm, vs2, simm5, vd); }
void vslideup_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x38004057, vm, vs2, rs1, vd); }
void vsll_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x94003057, vm, vs2, simm5, vd); }
void vsll_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x94000057, vm, vs2, vs1, vd); }
void vsll_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x94004057, vm, vs2, rs1, vd); }
void vsm_v(VReg vs3, const Reg& rs1) { opVectorStore(0x2b00027, 0, 0, rs1, vs3); }
void vsmul_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x9c000057, vm, vs2, vs1, vd); }
void vsmul_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x9c004057, vm, vs2, rs1, vd); }
void vsoxei1024_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x1c007027, vm, vs2, rs1, vs3); }
void vsoxei128_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x1c000027, vm, vs2, rs1, vs3); }
void vsoxei16_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0xc005027, vm, vs2, rs1, vs3); }
void vsoxei256_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x1c005027, vm, vs2, rs1, vs3); }
void vsoxei32_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0xc006027, vm, vs2, rs1, vs3); }
void vsoxei512_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x1c006027, vm, vs2, rs1, vs3); }
void vsoxei64_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0xc007027, vm, vs2, rs1, vs3); }
void vsoxei8_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0xc000027, vm, vs2, rs1, vs3); }
void vsra_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xa4003057, vm, vs2, simm5, vd); }
void vsra_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xa4000057, vm, vs2, vs1, vd); }
void vsra_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xa4004057, vm, vs2, rs1, vd); }
void vsrl_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xa0003057, vm, vs2, simm5, vd); }
void vsrl_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xa0000057, vm, vs2, vs1, vd); }
void vsrl_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xa0004057, vm, vs2, rs1, vd); }
void vssseg1e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18007027, vm, rs2, rs1, vs3); }
void vssseg2e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x38007027, vm, rs2, rs1, vs3); }
void vssseg3e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x58007027, vm, rs2, rs1, vs3); }
void vssseg4e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x78007027, vm, rs2, rs1, vs3); }
void vssseg5e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x98007027, vm, rs2, rs1, vs3); }
void vssseg6e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xb8007027, vm, rs2, rs1, vs3); }
void vssseg7e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xd8007027, vm, rs2, rs1, vs3); }
void vssseg8e1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xf8007027, vm, rs2, rs1, vs3); }
void vsse1024_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18007027, vm, rs2, rs1, vs3); }
void vssseg1e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18000027, vm, rs2, rs1, vs3); }
void vssseg2e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x38000027, vm, rs2, rs1, vs3); }
void vssseg3e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x58000027, vm, rs2, rs1, vs3); }
void vssseg4e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x78000027, vm, rs2, rs1, vs3); }
void vssseg5e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x98000027, vm, rs2, rs1, vs3); }
void vssseg6e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xb8000027, vm, rs2, rs1, vs3); }
void vssseg7e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xd8000027, vm, rs2, rs1, vs3); }
void vssseg8e128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xf8000027, vm, rs2, rs1, vs3); }
void vsse128_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18000027, vm, rs2, rs1, vs3); }
void vssseg1e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8005027, vm, rs2, rs1, vs3); }
void vssseg2e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x28005027, vm, rs2, rs1, vs3); }
void vssseg3e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x48005027, vm, rs2, rs1, vs3); }
void vssseg4e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x68005027, vm, rs2, rs1, vs3); }
void vssseg5e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x88005027, vm, rs2, rs1, vs3); }
void vssseg6e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xa8005027, vm, rs2, rs1, vs3); }
void vssseg7e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xc8005027, vm, rs2, rs1, vs3); }
void vssseg8e16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xe8005027, vm, rs2, rs1, vs3); }
void vsse16_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8005027, vm, rs2, rs1, vs3); }
void vssseg1e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18005027, vm, rs2, rs1, vs3); }
void vssseg2e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x38005027, vm, rs2, rs1, vs3); }
void vssseg3e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x58005027, vm, rs2, rs1, vs3); }
void vssseg4e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x78005027, vm, rs2, rs1, vs3); }
void vssseg5e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x98005027, vm, rs2, rs1, vs3); }
void vssseg6e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xb8005027, vm, rs2, rs1, vs3); }
void vssseg7e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xd8005027, vm, rs2, rs1, vs3); }
void vssseg8e256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xf8005027, vm, rs2, rs1, vs3); }
void vsse256_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18005027, vm, rs2, rs1, vs3); }
void vssseg1e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8006027, vm, rs2, rs1, vs3); }
void vssseg2e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x28006027, vm, rs2, rs1, vs3); }
void vssseg3e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x48006027, vm, rs2, rs1, vs3); }
void vssseg4e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x68006027, vm, rs2, rs1, vs3); }
void vssseg5e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x88006027, vm, rs2, rs1, vs3); }
void vssseg6e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xa8006027, vm, rs2, rs1, vs3); }
void vssseg7e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xc8006027, vm, rs2, rs1, vs3); }
void vssseg8e32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xe8006027, vm, rs2, rs1, vs3); }
void vsse32_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8006027, vm, rs2, rs1, vs3); }
void vssseg1e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18006027, vm, rs2, rs1, vs3); }
void vssseg2e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x38006027, vm, rs2, rs1, vs3); }
void vssseg3e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x58006027, vm, rs2, rs1, vs3); }
void vssseg4e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x78006027, vm, rs2, rs1, vs3); }
void vssseg5e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x98006027, vm, rs2, rs1, vs3); }
void vssseg6e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xb8006027, vm, rs2, rs1, vs3); }
void vssseg7e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xd8006027, vm, rs2, rs1, vs3); }
void vssseg8e512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xf8006027, vm, rs2, rs1, vs3); }
void vsse512_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x18006027, vm, rs2, rs1, vs3); }
void vssseg1e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8007027, vm, rs2, rs1, vs3); }
void vssseg2e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x28007027, vm, rs2, rs1, vs3); }
void vssseg3e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x48007027, vm, rs2, rs1, vs3); }
void vssseg4e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x68007027, vm, rs2, rs1, vs3); }
void vssseg5e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x88007027, vm, rs2, rs1, vs3); }
void vssseg6e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xa8007027, vm, rs2, rs1, vs3); }
void vssseg7e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xc8007027, vm, rs2, rs1, vs3); }
void vssseg8e64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xe8007027, vm, rs2, rs1, vs3); }
void vsse64_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8007027, vm, rs2, rs1, vs3); }
void vssseg1e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8000027, vm, rs2, rs1, vs3); }
void vssseg2e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x28000027, vm, rs2, rs1, vs3); }
void vssseg3e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x48000027, vm, rs2, rs1, vs3); }
void vssseg4e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x68000027, vm, rs2, rs1, vs3); }
void vssseg5e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x88000027, vm, rs2, rs1, vs3); }
void vssseg6e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xa8000027, vm, rs2, rs1, vs3); }
void vssseg7e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xc8000027, vm, rs2, rs1, vs3); }
void vssseg8e8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0xe8000027, vm, rs2, rs1, vs3); }
void vsse8_v(VReg vs3, const Reg& rs1, const Reg& rs2, VM vm=VM::unmasked) { opVectorStore(0x8000027, vm, rs2, rs1, vs3); }
void vssra_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xac003057, vm, vs2, simm5, vd); }
void vssra_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xac000057, vm, vs2, vs1, vd); }
void vssra_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xac004057, vm, vs2, rs1, vd); }
void vssrl_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0xa8003057, vm, vs2, simm5, vd); }
void vssrl_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xa8000057, vm, vs2, vs1, vd); }
void vssrl_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0xa8004057, vm, vs2, rs1, vd); }
void vssub_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x8c000057, vm, vs2, vs1, vd); }
void vssub_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x8c004057, vm, vs2, rs1, vd); }
void vssubu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x88000057, vm, vs2, vs1, vd); }
void vssubu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x88004057, vm, vs2, rs1, vd); }
void vsub_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x8000057, vm, vs2, vs1, vd); }
void vsub_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x8004057, vm, vs2, rs1, vd); }
void vsuxei1024_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x14007027, vm, vs2, rs1, vs3); }
void vsuxei128_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x14000027, vm, vs2, rs1, vs3); }
void vsuxei16_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x4005027, vm, vs2, rs1, vs3); }
void vsuxei256_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x14005027, vm, vs2, rs1, vs3); }
void vsuxei32_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x4006027, vm, vs2, rs1, vs3); }
void vsuxei512_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x14006027, vm, vs2, rs1, vs3); }
void vsuxei64_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x4007027, vm, vs2, rs1, vs3); }
void vsuxei8_v(VReg vs3, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opVectorStore(0x4000027, vm, vs2, rs1, vs3); }
void vwadd_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xc4002057, vm, vs2, vs1, vd); }
void vwadd_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xc4006057, vm, vs2, rs1, vd); }
void vwadd_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xd4002057, vm, vs2, vs1, vd); }
void vwadd_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xd4006057, vm, vs2, rs1, vd); }
void vwaddu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xc0002057, vm, vs2, vs1, vd); }
void vwaddu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xc0006057, vm, vs2, rs1, vd); }
void vwaddu_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xd0002057, vm, vs2, vs1, vd); }
void vwaddu_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xd0006057, vm, vs2, rs1, vd); }
void vwmacc_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xf4002057, vm, vs2, vs1, vd); }
void vwmacc_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xf4006057, vm, vs2, rs1, vd); }
void vwmaccsu_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xfc002057, vm, vs2, vs1, vd); }
void vwmaccsu_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xfc006057, vm, vs2, rs1, vd); }
void vwmaccu_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0xf0002057, vm, vs2, vs1, vd); }
void vwmaccu_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xf0006057, vm, vs2, rs1, vd); }
void vwmaccus_vx(const VReg& vd, const Reg& rs1, const VReg& vs2, VM vm=VM::unmasked) { opMVX(0xf8006057, vm, vs2, rs1, vd); }
void vwmul_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xec002057, vm, vs2, vs1, vd); }
void vwmul_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xec006057, vm, vs2, rs1, vd); }
void vwmulsu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xe8002057, vm, vs2, vs1, vd); }
void vwmulsu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xe8006057, vm, vs2, rs1, vd); }
void vwmulu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xe0002057, vm, vs2, vs1, vd); }
void vwmulu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xe0006057, vm, vs2, rs1, vd); }
void vwredsum_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xc4000057, vm, vs2, vs1, vd); }
void vwredsumu_vs(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0xc0000057, vm, vs2, vs1, vd); }
void vwsub_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xcc002057, vm, vs2, vs1, vd); }
void vwsub_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xcc006057, vm, vs2, rs1, vd); }
void vwsub_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xdc002057, vm, vs2, vs1, vd); }
void vwsub_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xdc006057, vm, vs2, rs1, vd); }
void vwsubu_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xc8002057, vm, vs2, vs1, vd); }
void vwsubu_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xc8006057, vm, vs2, rs1, vd); }
void vwsubu_wv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opMVV(0xd8002057, vm, vs2, vs1, vd); }
void vwsubu_wx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opMVX(0xd8006057, vm, vs2, rs1, vd); }
void vxor_vi(const VReg& vd, const VReg& vs2, int32_t simm5, VM vm=VM::unmasked) { opIVI(0x2c003057, vm, vs2, simm5, vd); }
void vxor_vv(const VReg& vd, const VReg& vs2, const VReg& vs1, VM vm=VM::unmasked) { opIVV(0x2c000057, vm, vs2, vs1, vd); }
void vxor_vx(const VReg& vd, const VReg& vs2, const Reg& rs1, VM vm=VM::unmasked) { opIVX(0x2c004057, vm, vs2, rs1, vd); }
void vzext_vf2(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x48032057, vm, vs2, 0, vd); }
void vzext_vf4(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x48022057, vm, vs2, 0, vd); }
void vzext_vf8(const VReg& vd, const VReg& vs2, VM vm=VM::unmasked) { opMVV(0x48012057, vm, vs2, 0, vd); }

void vsetivli(const Reg& rd, uint32_t uimm, SEW sew, LMUL lmul=LMUL::m1, VTA vta=VTA::tu, VMA vma=VMA::mu) {
    uint32_t zimm = (static_cast<uint32_t>(vma)<<7) |
                    (static_cast<uint32_t>(vta)<<6) |
                    (static_cast<uint32_t>(sew)<<3) |
                    (static_cast<uint32_t>(lmul));
    uint32_t v = (0x3<<30) | (zimm<<20) | (uimm<<15) | (0x7<<12) | (rd.getIdx()<<7) | (0x57);
    append4B(v);
}

void vsetvli(const Reg& rd, const Reg& rs1, SEW sew, LMUL lmul=LMUL::m1, VTA vta=VTA::tu, VMA vma=VMA::mu) {
    uint32_t zimm = (static_cast<uint32_t>(vma)<<7) |
                    (static_cast<uint32_t>(vta)<<6) |
                    (static_cast<uint32_t>(sew)<<3) |
                    (static_cast<uint32_t>(lmul));
    uint32_t v = (0x0<<31) | (zimm<<20) | (rs1.getIdx()<<15) | (0x7<<12) | (rd.getIdx()<<7) | (0x57);
    append4B(v);
}

void vsetvl(const Reg& rd, const Reg& rs1, const Reg& rs2) {
    uint32_t v = (0x40<<25) | (rs2.getIdx()<<20) | (rs1.getIdx()<<15) | (0x7<<12) | (rd.getIdx()<<7) | (0x57);
    append4B(v);
}


// Copy mask register
void vmmv_m(const VReg& vd, const VReg& vs) { vmand_mm(vd, vs, vs); }
// Clear mask register
void vmclr_m(const VReg& vd) { vmxor_mm(vd, vd, vd); }
// Set mask register
void vmset_m(const VReg& vd) { vmxnor_mm(vd, vd, vd); }
// Invert bits
void vmnot_m(const VReg& vd, const VReg& vs) { vmnand_mm(vd, vs, vs); }


// vector compare pseudoinstructions
void vmfgt_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { vmflt_vv(vd, vs2, vs1, vm); }
void vmfge_vv(const VReg& vd, const VReg& vs1, const VReg& vs2, VM vm=VM::unmasked) { vmfle_vv(vd, vs2, vs1, vm); }

// sign-related pseudoinstructions
void vfabs_v(const VReg& vd, const VReg& vs, VM vm=VM::unmasked) { vfsgnjx_vv(vd, vs, vs, vm); }
void vfneg_v(const VReg& vd, const VReg& vs, VM vm=VM::unmasked) { vfsgnjn_vv(vd, vs, vs, vm); }
