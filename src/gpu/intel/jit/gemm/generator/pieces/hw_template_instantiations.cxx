/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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


#ifdef _MSC_VER
#pragma warning (disable: 4661)     /* missing definition in template instatiation */
#endif

#if defined(DNNL_GPU_ISA_XELP)
REG_XELP_ISA(template class Generator<HW::Gen12LP>);
#elif defined(DNNL_GPU_ISA_XEHP)
REG_XEHP_ISA(template class Generator<HW::XeHP>);
#elif defined(DNNL_GPU_ISA_XEHPG)
REG_XEHPG_ISA(template class Generator<HW::XeHPG>);
#elif defined(DNNL_GPU_ISA_XEHPC)
REG_XEHPC_ISA(template class Generator<HW::XeHPC>);
#elif defined(DNNL_GPU_ISA_XE2)
REG_XE2_ISA(template class Generator<HW::Xe2>);
#elif defined(DNNL_GPU_ISA_XE3)
REG_XE3_ISA(template class Generator<HW::Xe3>);
#else
// Default to instantiating all classes
REG_XELP_ISA(template class Generator<HW::Gen12LP>);
REG_XEHP_ISA(template class Generator<HW::XeHP>);
REG_XEHPG_ISA(template class Generator<HW::XeHPG>);
REG_XEHPC_ISA(template class Generator<HW::XeHPC>);
REG_XE2_ISA(template class Generator<HW::Xe2>);
REG_XE3_ISA(template class Generator<HW::Xe3>);
#endif
