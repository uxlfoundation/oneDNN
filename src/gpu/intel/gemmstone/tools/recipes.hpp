/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_RECIPES_HPP
#define GEMMSTONE_TOOLS_RECIPES_HPP

#include "ngen.hpp"

struct GEMMRecipe {
    ngen::HW hw;
    char precisions[4];
    char layouts[4];
    struct Extras {
        int aCP, bCP;
        int aAlign, bAlign;

        Extras() : aCP(1), bCP(1), aAlign(1), bAlign(1) {}
    } extras;
    int unrollM, unrollN;
    const char *strategyString;
    struct {
        int m = 0, n = 0, k = 0;
    } trsmBlocking;
    char tag = '\0';
};

inline namespace {
static GEMMRecipe::Extras make_crosspack(int a, int b) {
    GEMMRecipe::Extras result;
    result.aCP = a;
    result.bCP = b;
    return result;
}
[[maybe_unused]] static GEMMRecipe::Extras make_align(int a, int b) {
    GEMMRecipe::Extras result;
    result.aAlign = a;
    result.bAlign = b;
    return result;
}
} // inline namespace

GEMMRecipe gemmRecipes[] = {
    {ngen::HW::XeHP, "SSS", "ABN", {}, 32, 16, "sb2x2 sb4x2 sb ca1x2 wg 2x8 cs di hi bk1536/1536", {}, {}},
    {ngen::HW::XeHP, "SSS", "ABN", {}, 16, 32, "sb4x2 sb2x2 as cs di fn hi", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 64, 8,  "sb1x2 su4/2x2 sb ca1 wg 2x8 cs di hi bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 16, "ab4/2 ab8 ab cb1 wg 8x2 cs di nmk hi bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 16, "sb4/2x2 su1 su cb1 wg 16x2 cs di nmk", {2048, 16384, 2048}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 8,  "sb4 sb8 sb cab1 wg 4x4 cs di bm4096 bn4096 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 4,  "sb8 su8x2 sb ca1 wg 2x8 cs di bm4096 bn4096 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 16, 16, "sb8 sb16/8 sb cs di", {16384, 2048, 2048}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 16, 8,  "sb16 sb16 sb cab1 wg 4x4 cs di bm2048 bn2048 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 16, 4,  "sb8 su32 sb cab1 wg 4x4 cs di bm4096 bn4096 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 8,  8,  "sb16 sb16 sb cab1 wg 4x4 cs di bm2048 bn2048 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 8,  4,  "sb16 sb32 sb cab1 wg 2x8 cs di bm8192 bn8192 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 32, 16, "sb4/2x2 sb4x2 sb cs di hi bk2048", {}, {}},   // faster for large sizes: sb4 su8/4 sb cab1 wg 4x4 cs di
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 32, "sb2x2 sb4/2x2 sb cs di nmk", {16384, 4096, 4096}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 16, "sb4x3 sb4x3 sb cs di cab1 wg 4x4 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 16, "ab4x2 ab4x2 ab cs di cab1 wg 4x4 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 8,  "sb16 sb16 sb cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 8,  8,  "sb16 sb16 sb cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 8,  8,  "sb2x2 sb2x2 sb wg 2x2x16 kb kr cs di", {}, 'K'},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 16, 16, "sb8 sb8 su cab1 wg 4x4 cs di hi bk2048 ek", {2048, 16384, 2048}, {}},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 8,  16, "sb16 sb16 sb cab1 wg 4x4 cs di ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 8,  8,  "sb16 sb16 sb cab1 wg 4x4 cs di ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 8,  4,  "sb16 sb32 sb cab1 wg 2x8 cs di ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "TTN", {}, 8,  32, "sb8 sb4 sb cab1 wg 4x4 cs di bm4096 bn4096 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "TTN", {}, 8,  64, "su4/2x2 sb1x2 su cb1 wg 8x2 cs di fn hi bk2048", {}, {}},
    {ngen::HW::XeHP, "HHS", "ABN", make_crosspack(2, 16), 32, 32, "ab16x3 ab16x3 ab fs sc bo acb bk2048", {}, 'A'},
    {ngen::HW::XeHP, "HHS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 4x4 bo acb bk4096", {}, 'A'},
    {ngen::HW::XeHP, "HHS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 8x4 bo acb bk4096", {}, 'B'},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 32, "ab4/1x2 as4/1x2 ab l4 cb1 wg 8x2 cs nmk hi bk2048", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 16, "sb16 su4 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32,  4, "sb8 su16x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 16,  4, "sb4 su16x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NTN", {}, 32, 32, "ab4/1x2 ab2/1x2 ab l4 cs hi bk2048", {}, {}},  // slightly faster: ab8/2 ab4/1x2 ab l4 ca1 wg 2x8 cs np
    {ngen::HW::XeHP, "HHH", "NTN", {}, 32, 16, "sb8 sb16 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "TNN", {}, 16, 16, "su8 su32 sb l4 cab1 wg 4x4 cs di hi bk2048", {}, {}},
    {ngen::HW::XeHP, "HHH", "TNN", {}, 16,  8, "su32 su16 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "TTN", {}, 32, 32, "as4/1x2 ab4/1x2 ab l4 ca1 wg 2x8 cs nmk hi bk2048", {}, {}},
    {ngen::HW::XeHP, "BBS", "ABN", make_crosspack(2, 16), 32, 32, "ab16x3 ab16x3 ab fs sc bo acb bk2048", {}, 'A'},
    {ngen::HW::XeHP, "BBS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 4x4 bo acb bk4096", {}, 'A'},
    {ngen::HW::XeHP, "BBS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 8x4 bo acb bk4096", {}, 'B'},
    {ngen::HW::XeHP, "OOI", "ABN", make_crosspack(4, 32), 32, 32, "ab16x3 ab16x3 ab fs sc bo acb bk4096", {}, 'A'},
    {ngen::HW::XeHP, "OOI", "ABN", make_crosspack(4, 32), 32, 48, "ab16 ab16 ab fs wg 4x4 bo acb bk8192", {}, 'A'},
    {ngen::HW::XeHP, "OOI", "ABN", make_crosspack(4, 32), 32, 48, "ab16 ab16 ab fs wg 8x4 bo acb bk8192", {}, 'B'},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 32, 16, "sb8 sb32 sb l4 cab1 wg 4x4 cs di pab", {}, {}},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 16, 16, "sb32 sb32 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 32,  4, "sb8x2 su16x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 16,  4, "sb16 su32x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "NTN", {}, 32, 16, "sb8x2 sb8x2 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "TNN", {}, 16, 16, "sb32 sb32 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "TTN", {}, 16, 16, "sb32 sb32 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 16,  4, "ab2x2 as8 ab l4 cs di", {}, {}},                        // DLRM
    {ngen::HW::XeHP, "HHH", "TNN", {}, 16,  8, "sb16 sb16 ab cab2 wg 2x4 cs pab", {}, {}},              // DLRM
    {ngen::HW::XeHP, "BBS", "NNN", {}, 32,  8, "sb16 sb16 ab cab1 wg 4x4 fn nmk cs pab", {}, {}},       // DLRM
    {ngen::HW::XeHP, "BBS", "NTN", {}, 16, 16, "sb1x4 sb1x4 sb l4 cs di nmk fn pab", {}, {}},           // DLRM
    {ngen::HW::XeHP, "BBS", "TNN", {}, 32,  8, "sb16 sb16 ab ca2 wg 1x4 fn nmk cs pab", {}, {}},        // DLRM
    {ngen::HW::XeHP, "BBS", "TTN", {}, 8,  32, "sb16 sb16 as cab1 wg 4x4 cs pab", {}, {}},              // DLRM
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 16, "ab16/8 as16 ab l4 cab1 wg 4x4 cs di", {}, 'A'},         // BERT
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 16, "ab8 ab16 ab l4 ca1 wg 2x8 cs di", {}, 'B'},             // BERT
    {ngen::HW::XeHPG, "HHS", "ABN", make_crosspack(2, 16), 32, 48, "sB16 sB16 sb fs wg 4x4 bo acb bk4096", {}, 'A'},
    {ngen::HW::XeHPG, "HHS", "ABN", make_crosspack(2, 16), 32, 48, "sB16 sB16 sb fs wg 8x4 bo acb bk4096", {}, 'B'},
    {ngen::HW::XeHPG, "BBS", "ABN", make_crosspack(2, 16), 32, 48, "sB16 sB16 sb fs wg 4x4 bo acb bk4096", {}, 'A'},
    {ngen::HW::XeHPG, "BBS", "ABN", make_crosspack(2, 16), 32, 48, "sB16 sB16 sb fs wg 8x4 bo acb bk4096", {}, 'B'},
    {ngen::HW::XeHPG, "OOI", "ABN", make_crosspack(4, 32), 32, 48, "sB16 sB16 sb fs wg 4x4 bo acb bk8192", {}, 'A'},
    {ngen::HW::XeHPG, "OOI", "ABN", make_crosspack(4, 32), 32, 48, "sB16 sB16 sb fs wg 8x4 bo acb bk8192", {}, 'B'},
    {ngen::HW::XeHPC, "SSS", "ABN", {}, 32, 32, "aB8+B8@8 aB8+B8@8 aB nse di wg 4x8 bo pt sr sb256 kc8 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 64, 32, "aB8/4+B8@32/0 aS8+S1,16@32/0 aB nse di wg 4x8 bo pt sr kc8 sn sb256 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 16, 16, "aB8x2 aS16x2 aB nse di wg 4x8 sr kc8 bk0 sb256", {16384, 2048, 2048}, {}},
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 16, 16, "aB8x2 aS16x2 aU nse di wg 8x4 sr kc8 bk0 sb256", {2048, 16384, 2048}, {}},
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 16, 8,  "aB8x2+B8@32 aS8+S1,16@32 aB nse di wg 4x8 bo pt sr kc8 sn sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NTN", {}, 64, 32, "aB8/4x2+B8@32 aB8/4x2+B8@32 aB nse di wg 4x8 bo pt sr sb256 kc8 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NTN", {}, 16, 16, "aB8x2+B8@32 aB8x2+B8@32 aB nse di wg 4x8 bo pt sr kc8 sb256 bk0", {16384, 2048, 2048}, {}},
    {ngen::HW::XeHPC, "SSS", "TNN", {}, 64, 32, "aS8+S1,16@32/0 aS16+S1,16@32/0 aB nse di wg 4x8 bo pt sr kc8 sm sn sb256 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "TNN", {}, 16, 16, "aS8x2 aS8x2 aU nse di wg 8x4 sr kc8 bk0 sb256", {2048, 16384, 2048}, {}},
    {ngen::HW::XeHPC, "SSS", "TNN", {}, 16, 8,  "aS16+S1,16@32 aS16+S1,16@32 aB nse di wg 4x8 bo pt sr kc8 sm sn sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "TTN", {}, 32, 64, "aS8+S1,16@32/0 aB8/4+B8@32/0 aU nse di wg 8x4 bo pt sr kc8 sm sb256 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "ABN", make_crosspack(2, 16), 64, 32, "aB16+B16@48 aB16+B16@48 aB vav di sys grf256 af hi pt wg 4x8 sr sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "ABN", make_crosspack(2, 16), 16, 16, "aB16x2+B16@96 aB16x2+B16@96 aB vav sb256 wg 4x8 sr di bo pt sys bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NNN", make_align(128, 128),  64, 32, "av32+B32@64 am32+S1,32@64 aB vav di sys grf256 af wg 4x8 bo sb256 sn bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NNN", make_align(128, 128),  16, 16, "av32+B32@64 am64+S1,32@64 aB vav sb256 wg 8x4 sr di sn bo pt sys bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NTN", make_align(128, 128),  64, 32, "av16x2+B16@64/0 av16+B16@64 aB vav sys af wg 4x8 k64 ka64 cb1 grf256 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NTN", make_align(128, 128),  16, 16, "av16x2+B16@64/0 av16+B16@64 aB vav sys af wg 4x8 k64 ka64 cb1 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TNN", make_align(4,   128),  64, 32, "aS16+S1,32@32 am16+S1,32@32 aB vav di sys grf256 af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TNN", make_align(4,   128),  16, 16, "aS16+S1,32@32 am16+S1,32@32 aB vav di sys af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TTN", make_align(128, 128),  32, 64, "am32+S1,32@64 av32+B32@64 aS vav di sys grf256 af wg 8x4 bo sb256 sm bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "ABN", make_crosspack(2, 16), 64, 32, "aB16+B16@48 aB16+B16@48 aB vav di sys grf256 af hi pt wg 4x8 sr sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "ABN", make_crosspack(2, 16), 16, 16, "aB16x2+B16@96 aB16x2+B16@96 aB vav sb256 wg 4x8 sr di bo pt sys bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NNN", make_align(128, 128),  64, 32, "av32+B32@64 am32+S1,32@64 aB vav di sys grf256 af wg 4x8 bo sb256 sn bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NNN", make_align(128, 128),  16, 16, "av32+B32@64 am64+S1,32@64 aB vav sb256 wg 8x4 sr di sn bo pt sys bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NTN", make_align(128, 128),  64, 32, "av16x2+B16@64/0 av16+B16@64 aB vav sys af wg 4x8 k64 ka64 cb1 grf256 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NTN", make_align(128, 128),  16, 16, "av16x2+B16@64/0 av16+B16@64 aB vav sys af wg 4x8 k64 ka64 cb1 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TNN", make_align(4,   128),  64, 32, "aS16+S1,32@32 am16+S1,32@32 aB vav di sys grf256 af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TNN", make_align(4,   128),  16, 16, "aS16+S1,32@32 am16+S1,32@32 aB vav di sys af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TTN", make_align(128, 128),  32, 64, "am32+S1,32@64 av32+B32@64 aS vav di sys grf256 af wg 8x4 bo sb256 sm bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "ABN", make_crosspack(4, 32), 64, 32, "aB32+B32@96 aB32+B32@96 aB vav di sys grf256 af hi pt wg 4x8 sr sb512 bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "NNN", make_align(128, 128),  64, 32, "av64+B64@128 am64+S1,64@128/0 aB vav di sys grf256 af wg 4x8 bo sb512 sn bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "NTN", make_align(128, 128),  64, 32, "av32+B32@128 av32+B32@128 aB vav sys af wg 4x8 k128 ka128 cb1 grf256 bo sb512 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "TNN", make_align(4,   128),  64, 32, "aS32+S1,64@64 am32+S1,64@64 aB vav di sys grf256 af wg 4x8 bo sb512 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "TTN", make_align(128, 128),  32, 64, "am64+S1,64@128/0 av64+B64@128 aS vav di sys grf256 af wg 8x4 bo sb512 sm bk0", {}, {}},
};

constexpr int gemmRecipeCount = sizeof(gemmRecipes) / sizeof(GEMMRecipe);

#endif /* header guard */
