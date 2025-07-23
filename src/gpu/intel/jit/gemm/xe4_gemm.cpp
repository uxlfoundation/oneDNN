/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gpu/intel/jit/gemm/xe4_gemm.hpp"

#include "gpu/intel/jit/generator.hpp"
#include "ngen_register_allocator.hpp"

using namespace ngen;

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class xe4_gemm_kernel_t : public generator_t<gpu_xe4> {
public:
    static const uint32_t MMA_NULL_C = (1 << 8);

    struct Pipeline {
        Pipeline(xe4_gemm_kernel_t *host, int stages) : host(host) {
            a_cons_bar = 0;
            a_prod_bar = a_cons_bar + stages;
            b_cons_bar = a_prod_bar + stages;
            b_prod_bar = b_cons_bar + stages;
            d_bar = b_prod_bar + stages;
            store_bar = d_bar + 1;
            end = store_bar + 1;
        }

        void init(SRF one) {
            stage = host->alloc_srf();
            host->mov(stage, 0);
            auto tmp = host->alloc_srf();
            for (int i = 0; i < barrier_count(); i++) {
                host->mov(tmp, i * 8);
                host->abarrierinit(tmp, one);
            }
            host->barrier();
        }

        int barrier_count() const { return end; }

        void get_prod_bar(char ab, SRF out) {
            host->mul(out, stage, 8);
            host->add(out, out, 8 * ((ab == 'a') ? a_prod_bar : b_prod_bar));
        }
        void get_cons_bar(char ab, SRF out) {
            host->mul(out, stage, 8);
            host->add(out, out, 8 * ((ab == 'a') ? a_cons_bar : b_cons_bar));
        }
        void get_d_bar(SRF out) { host->mov(out, d_bar * 8); }
        void get_store_bar(SRF out) { host->mov(out, store_bar * 8); }

        xe4_gemm_kernel_t *host = nullptr;
        int a_cons_bar = 0;
        int a_prod_bar = 0;
        int b_cons_bar = 0;
        int b_prod_bar = 0;
        int d_bar = 0;
        int store_bar = 0;
        int end = 0;
        SRF stage;
    };

    xe4_gemm_kernel_t(const xe4_gemm_t::kernel_desc_t &desc)
        : generator_t<gpu_xe4>({GENERATOR_NAME, GENERATOR_LINE})
        , desc_(desc)
        , ra_(HW::Xe4) {
        Pipeline pipeline(this, desc_.stages);

        const uint32_t a_slm_bytes
                = desc_.bm * desc_.bk * ngen::getBytes(desc_.a_type);
        const uint32_t b_slm_bytes
                = desc_.bk * desc_.bn * ngen::getBytes(desc_.b_type);
        const uint32_t c_slm_bytes
                = desc_.bm * desc_.bn * ngen::getBytes(desc_.c_type);
        const uint32_t acc_slm_bytes
                = desc_.bm * desc_.bn * ngen::getBytes(desc_.acc_type());
        const uint32_t slm_bytes = a_slm_bytes * desc_.stages
                + b_slm_bytes * desc_.stages + c_slm_bytes + acc_slm_bytes;
        const uint32_t a_slm_off = 0;
        const uint32_t b_slm_off = a_slm_off + a_slm_bytes * desc_.stages;
        const uint32_t c_slm_off = b_slm_off + b_slm_bytes * desc_.stages;
        const uint32_t acc_slm_off = c_slm_off + c_slm_bytes;

        newArgument("a_buf", ExternalArgumentType::GlobalPtr);
        newArgument("b_buf", ExternalArgumentType::GlobalPtr);
        newArgument("c_buf", ExternalArgumentType::GlobalPtr);
        newArgument("M", DataType::u32);
        newArgument("N", DataType::u32);
        newArgument("K", DataType::u32);
        requireLocalID(3);
        requireLocalSize();
        requireABarriers(pipeline.barrier_count());
        requireBarrier();
        requireSLM(slm_bytes);
        externalName("xe4_gemm");
        finalizeInterface();
        setDefaultAutoSWSB();
        prologue();

        auto a_buf = getArgument("a_buf");
        auto b_buf = getArgument("b_buf");
        auto c_buf = getArgument("c_buf");
        auto M = to_srf(getArgument("M"));
        auto N = to_srf(getArgument("N"));
        auto K = to_srf(getArgument("K"));

        ra_.claim(SRF(0));
        ra_.claim(a_buf);
        ra_.claim(b_buf);
        ra_.claim(c_buf);
        ra_.claim(M);
        ra_.claim(N);
        ra_.claim(K);

        for (int i = 0; i < 3; i++)
            ra_.claim(getLocalID(i));
        for (int i = 0; i < 3; i++)
            ra_.claim(getGroupID(i));

        auto local_id = getLocalID(0);
        auto m_wg_idx = alloc_srf();
        auto n_wg_idx = alloc_srf();
        auto sg_idx = alloc_grf();
        auto one = alloc_srf();
        auto k = alloc_srf();
        auto k_last = alloc_srf();
        auto phase = alloc_srf();
        auto mma_flags = alloc_srf(DataType::u64);

        mul(n_wg_idx, getGroupID(0), desc_.bn);
        mul(m_wg_idx, getGroupID(1), desc_.bm);
        add(k_last, K, Immediate::d(-1));
        and_(k_last, k_last, ~(desc_.bk - 1));
        shr(sg_idx, local_id, 5);
        mov(one, 1);
        mov(k, 0);

        // Init barriers.
        pipeline.init(one);
        auto stage = pipeline.stage;

        // Producer subgroup.
        Label prod_label1;
        Label prod_label2;
        Label prod_label3;
        cmp(ne | p1, sg_idx, 0);
        goto_(p1, prod_label1, prod_label1);

        mov(phase, 1);
        mark(prod_label2);

        async_load('a', desc_.a_trans, M, K, desc_.bm, desc_.bk, to_srf(a_buf),
                m_wg_idx, k, a_slm_off, a_slm_bytes, pipeline, phase);
        async_load('b', desc_.b_trans, K, N, desc_.bk, desc_.bn, to_srf(b_buf),
                k, n_wg_idx, b_slm_off, b_slm_bytes, pipeline, phase);

        add(stage, stage, 1);
        cmp(eq | p2, stage, desc_.stages);
        sel(stage, stage, 0, !p2);
        xor_(p2, phase, phase, 0x1);

        add(k, k, desc_.bk);
        // FIXME: when using u32, XeSim complains on cm value which seems
        // related to XeISA Maintenance3 update.
        cmp(le | p2, k.s32(), k_last.s32());
        goto_(p2, prod_label3, prod_label2, true);
        mark(prod_label3);
        join(prod_label3);

        mark(prod_label1);
        join(prod_label1);

        // Consumer subgroup.
        Label cons_label1;
        Label cons_label2;
        Label cons_label3;
        cmp(ne | p1, sg_idx, 1);
        goto_(p1, cons_label1, cons_label1);

        mov(phase, 0);
        mov(mma_flags, MMA_NULL_C);
        mark(cons_label2);

        auto is_last = p2;
        cmp(eq | is_last, k, k_last);

        Label l1, l2;
        goto_(is_last, l1, l1);
        async_mma(desc_.a_trans, desc_.b_trans, a_slm_off, a_slm_bytes,
                b_slm_off, b_slm_bytes, c_slm_off, acc_slm_off, mma_flags,
                pipeline, phase);
        mark(l1);
        join(l1);
        goto_(~is_last, l2, l2);
        async_mma(desc_.a_trans, desc_.b_trans, a_slm_off, a_slm_bytes,
                b_slm_off, b_slm_bytes, c_slm_off, acc_slm_off, mma_flags,
                pipeline, phase,
                /*is_last=*/true);
        mark(l2);
        join(l2);

        mov(mma_flags, 0);

        add(stage, stage, 1);
        cmp(eq | p2, stage, desc_.stages);
        sel(stage, stage, 0, !p2);
        xor_(p2, phase, phase, 0x1);

        add(k, k, desc_.bk);
        cmp(le | p2, k.s32(), k_last.s32());
        goto_(p2, cons_label3, cons_label2, true);
        mark(cons_label3);
        join(cons_label3);

        mark(cons_label1);
        join(cons_label1);

        // Epilogue subgroup.
        Label epi_label1;
        cmp(ne | p1, sg_idx, 2);
        goto_(p1, epi_label1, epi_label1);

        async_store(M, N, desc_.bm, desc_.bn, to_srf(c_buf), m_wg_idx, n_wg_idx,
                c_slm_off, pipeline);

        mark(epi_label1);
        join(epi_label1);

        barrier();

        threadend();
        for (int i = 0; i < 8; i++)
            nop();
    }

    SRF alloc_srf(DataType type = DataType::u32) {
        auto ret = ra_.allocSRF(type);
        ret = ret.retype(type);
        return ret;
    }

    GRF alloc_grf(DataType type = DataType::u32) {
        auto ret = ra_.allocRange(ngen::getBytes(type) == 8 ? 2 : 1)[0];
        ret = ret.retype(type);
        return ret;
    }

    static SRF to_srf(Subregister sub) {
        return SRF(sub.getBase()).retype(sub.getType());
    }

    static SRF to_srf(Register reg) { return SRF(reg.getBase()); }

    void zero_out(RegisterRange range) {
        for (int i = 0; i < range.getLen(); i++) {
            mov(SRF(range[i].getBase()).b32(), 0);
        }
    }

    // Loads MxN 2D tensor from global memory to SLM.
    //   notrans: row-major (N-major)
    //   trans  : column-major (M-major)
    template <typename MOFF, typename NOFF>
    void async_load(char ab, transpose_t trans, SRF m, SRF n, uint32_t m_blk,
            uint32_t n_blk, Register addr_base, MOFF m_off, NOFF n_off,
            uint32_t slm_off, uint32_t slm_bytes, Pipeline pipeline,
            SRF phase) {
        bool row_major = (trans == transpose::notrans);
        auto ops = alloc_srf();
        auto tmp = alloc_srf(DataType::b64);
        auto payload = ra_.allocSRFRange(7);
        auto tdesc = ra_.allocSRFRange(16);
        mov(ops,
                m_blk * n_blk
                        * ngen::getBytes(
                                ab == 'a' ? desc_.a_type : desc_.b_type));
        mov<uint32_t>(payload[0], row_major ? n_off : m_off);
        mov<uint32_t>(payload[1], row_major ? m_off : n_off);
        mul<uint32_t>(payload[5], pipeline.stage, slm_bytes);
        add<uint32_t>(payload[5], payload[5], slm_off);
        shr<uint32_t>(payload[5], payload[5], 9);
        if (row_major) {
            mov<uint32_t>(tdesc[0], ((m_blk - 1) << 16) + (n_blk - 1));
            add<uint32_t>(tdesc[3], n, Immediate::d(-1));
            add<uint32_t>(tdesc[4], m, Immediate::d(-1));
        } else {
            mov<uint32_t>(tdesc[0], ((n_blk - 1) << 16) + (m_blk - 1));
            add<uint32_t>(tdesc[3], m, Immediate::d(-1));
            add<uint32_t>(tdesc[4], n, Immediate::d(-1));
        }
        mov<uint32_t>(tdesc[2], 0); // element strides
        shl<uint32_t>(tdesc[8], (row_major ? n : m),
                ngen::getLog2Bytes(ab == 'a'
                                ? desc_.a_type
                                : desc_.b_type)); // 40 bits per stride
        mov<uint32_t>(tdesc[9], 0);

        auto cons_bar = alloc_srf();
        auto prod_bar = to_srf(payload[6]).u32();
        pipeline.get_prod_bar(ab, prod_bar);
        pipeline.get_cons_bar(ab, cons_bar);

        abarriertry(0, cons_bar, phase);
        abarrierwait(0);

        auto data_type = (ab == 'a' ? desc_.a_type : desc_.b_type);
        bool is_type1 = (row_major || ab == 'b');
        auto cm_type = (is_type1 ? Type1 : Type2);
        bfi<uint32_t>(2, 28, payload[5], payload[5], is_type1 ? 0 : 1);
        bfi<uint32_t>(11, 16, payload[5], payload[5],
                (row_major ? n_blk : m_blk) >> 2);
        admatg2l(data_type | cm_type | ABarrier
                        | ADMAOptions::createTensorDims(2),
                payload[0], tdesc[0], addr_base);

        abarrierarriveexp(tmp, prod_bar, ops);

        ra_.release(ops);
        ra_.release(tmp);
        ra_.release(cons_bar);
        ra_.release(payload);
        ra_.release(tdesc);
    }

    // Stores MxN (N-major) 2D tensor from SLM to global memory.
    template <typename MOFF, typename NOFF>
    void async_store(SRF m, SRF n, uint32_t m_blk, uint32_t n_blk,
            Register addr_base, MOFF m_off, NOFF n_off, uint32_t slm_off,
            Pipeline pipeline) {
        auto ops = alloc_srf();
        auto tmp = alloc_srf(DataType::b64);
        auto zero = alloc_srf();
        auto payload = ra_.allocSRFRange(7);
        auto tdesc = ra_.allocSRFRange(16);
        mov(zero, 0);
        mov(ops, m_blk * n_blk * ngen::getBytes(desc_.c_type));
        mov<uint32_t>(payload[0], n_off);
        mov<uint32_t>(payload[1], m_off);
        mov<uint32_t>(payload[5], slm_off);
        shr<uint32_t>(payload[5], payload[5], 9);
        bfi<uint32_t>(2, 28, payload[5], payload[5], 0);
        bfi<uint32_t>(11, 16, payload[5], payload[5], n_blk >> 2);
        mov<uint32_t>(tdesc[0], ((m_blk - 1) << 16) + (n_blk - 1));
        mov<uint32_t>(tdesc[2], 0); // element strides
        add<uint32_t>(tdesc[3], n, Immediate::d(-1));
        add<uint32_t>(tdesc[4], m, Immediate::d(-1));
        shl<uint32_t>(tdesc[8], n,
                ngen::getLog2Bytes(desc_.c_type)); // 40 bits per stride
        mov<uint32_t>(tdesc[9], 0);

        auto d_bar = alloc_srf();
        auto store_bar = to_srf(payload[6]).u32();
        pipeline.get_d_bar(d_bar);
        pipeline.get_store_bar(store_bar);

        abarriertry(0, d_bar, zero);
        abarrierwait(0);

        admatl2g(desc_.c_type | Type1 | ABarrier
                        | ADMAOptions::createTensorDims(2),
                payload[0], tdesc[0], addr_base);

        abarrierarriveexp(tmp, store_bar, ops);
        abarriertry(0, store_bar, zero);
        abarrierwait(0);

        ra_.release(ops);
        ra_.release(tmp);
        ra_.release(zero);
        ra_.release(d_bar);
        ra_.release(payload);
        ra_.release(tdesc);
    }

    void async_mma(transpose_t a_trans, transpose_t b_trans, uint32_t a_slm_off,
            uint32_t a_slm_bytes, uint32_t b_slm_off, uint32_t b_slm_bytes,
            uint32_t c_slm_off, uint32_t acc_slm_off, SRF flags,
            Pipeline pipeline, SRF phase, bool is_last = false) {
        auto one = alloc_srf();
        auto tmp = alloc_srf(DataType::b64);
        auto tmp0 = SRF(tmp.getBase());
        auto tmp1 = SRF(tmp.getBase() + 1);
        auto desc = ra_.allocSRFRange(4);
        auto barriers = ra_.allocSRFRange(3);
        bool a_row_major = (a_trans == transpose::notrans);
        bool b_row_major = (b_trans == transpose::notrans);
        int a_bytes = ngen::getBytes(desc_.a_type);
        uint32_t acc_slm_stride = desc_.bn >> 2;
        uint32_t c_slm_stride = desc_.bn >> 2;
        uint32_t a_slm_stride = (a_row_major ? desc_.bk : desc_.bm) >> 2;
        uint32_t b_slm_stride = (b_row_major ? desc_.bn : desc_.bk) >> 2;
        // Order: DABC
        if (is_last) {
            mov<uint32_t>(desc[0], c_slm_off >> 9 | (c_slm_stride << 16));
            mov<uint32_t>(desc[3], acc_slm_off >> 9 | (acc_slm_stride << 16));
        } else {
            mov<uint32_t>(desc[0], acc_slm_off >> 9 | (acc_slm_stride << 16));
            mov<uint32_t>(desc[3], acc_slm_off >> 9 | (acc_slm_stride << 16));
        }

        mul<uint32_t>(tmp0, pipeline.stage, a_slm_bytes);
        mul<uint32_t>(tmp1, pipeline.stage, b_slm_bytes);
        add<uint32_t>(tmp0, tmp0, a_slm_off);
        add<uint32_t>(tmp1, tmp1, b_slm_off);

        shr<uint32_t>(desc[1], tmp0, 9);
        shr<uint32_t>(desc[2], tmp1, 9);

        or_<uint32_t>(desc[1], desc[1],
                (a_slm_stride << 16) | (a_row_major ? 0 : (1 << 28)));
        or_<uint32_t>(desc[2], desc[2], (b_slm_stride << 16));

        mov(one, 1);

        auto a_cons_bar = to_srf(barriers[0]).u32();
        auto b_cons_bar = to_srf(barriers[1]).u32();
        auto d_bar = to_srf(barriers[2]).u32();
        auto a_prod_bar = alloc_srf();
        auto b_prod_bar = alloc_srf();

        pipeline.get_cons_bar('a', a_cons_bar);
        pipeline.get_cons_bar('b', b_cons_bar);
        if (is_last) pipeline.get_d_bar(d_bar);
        pipeline.get_prod_bar('a', a_prod_bar);
        pipeline.get_prod_bar('b', b_prod_bar);

        abarriertry(0, a_prod_bar, phase);
        abarrierwait(0);
        abarriertry(0, b_prod_bar, phase);
        abarrierwait(0);

        auto opts = ATrack | BTrack;
        if (is_last) opts |= DTrack;
        if (!a_row_major) opts |= ATranspose;
        if (!b_row_major) opts |= BTranspose;
        uint32_t k_bytes = desc_.bk * a_bytes;
        bool both_u8 = (desc_.a_type == ngen::DataType::u8
                && desc_.b_type == ngen::DataType::u8);
        // XXX: Adjust when fixed in XeSim.
        if (a_bytes == 1 && !both_u8) k_bytes *= 2;
        amma(false, desc_.bm, desc_.bn, k_bytes,
                is_last ? desc_.c_type : desc_.acc_type(), desc_.a_type,
                desc_.b_type, desc_.acc_type(), opts, desc, barriers, flags);
        if (is_last) abarrierarriveexp(tmp, d_bar, one);
        abarrierarriveexp(tmp, a_cons_bar, one);
        abarrierarriveexp(tmp, b_cons_bar, one);

        ra_.release(one);
        ra_.release(tmp);
        ra_.release(desc);
        ra_.release(barriers);
        ra_.release(a_prod_bar);
        ra_.release(b_prod_bar);
    }

private:
    xe4_gemm_t::kernel_desc_t desc_;
    RegisterAllocator ra_;
};

status_t xe4_gemm_t::kernel_desc_t::create_generator(
        const compute::compute_engine_t &engine,
        compute::kernel_t &kernel) const {
    xe4_gemm_kernel_t gemm_kernel(*this);
    return engine.create_kernel(&kernel, &gemm_kernel);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
