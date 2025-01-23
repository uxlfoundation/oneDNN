#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "gated_mlp_pd.hpp"
#include "gated_mlp_utils.hpp"

dnnl_status_t dnnl_gmlp_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t W_gate_desc,
        const_dnnl_memory_desc_t W_up_desc,
        const_dnnl_memory_desc_t W_down_desc, const_dnnl_memory_desc_t dst_desc,
        const_dnnl_primitive_attr_t attr, const_dnnl_primitive_attr_t gate_attr,
        const_dnnl_primitive_attr_t up_attr,
        const_dnnl_primitive_attr_t down_attr) {
    auto gated_mlp_desc = dnnl::impl::create_gated_mlp_desc(src_desc,
            W_gate_desc, W_up_desc, W_down_desc, dst_desc, gate_attr, up_attr,
            down_attr);
    return dnnl::impl::primitive_desc_create(primitive_desc_iface, engine,
            (const dnnl::impl::op_desc_t *)&gated_mlp_desc, nullptr, attr);
}
