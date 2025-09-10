/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_PROBLEM_HPP
#define GPU_INTEL_JIT_IR_PROBLEM_HPP

#include <string>
#include <vector>

#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

enum class tensor_kind_t {
    undef,
    src,
    wei,
    dst,
    bias,
    a,
    b,
    c,
};

std::string to_string(tensor_kind_t tensor);

class pvar_name_t {
public:
    pvar_name_t() = default;
    explicit pvar_name_t(dim_idx_t idx) { data_[0] = into<char>('a' + idx); }
    explicit pvar_name_t(const std::string &s);

    bool operator==(const pvar_name_t &other) const {
        return std::strncmp(data_, other.data_, max_len) == 0;
    }

    bool operator!=(const pvar_name_t &other) const {
        return !operator==(other);
    }

    bool operator<(const pvar_name_t &other) const {
        return std::strncmp(data_, other.data_, max_len) < 0;
    }

    dim_idx_t index() const {
        gpu_assert(into<int>(length()) == 1);
        dim_idx_t idx = into<dim_idx_t>(data_[0]);
        gpu_assert('a' <= idx && idx <= 'z');
        return into<dim_idx_t>(idx - 'a');
    }

    size_t length() const { return std::strlen(data_); }
    bool is_empty() const { return std::strlen(data_) == 0; }
    size_t get_hash() const;
    std::string str() const;

    IR_DEFINE_DUMP()

private:
    static const size_t max_len = 8;
    char data_[max_len + 1] = {};
};

class pvar_t {
public:
    pvar_t() = default;
    pvar_t(size_t idx) : name_(into<dim_idx_t>(idx)) {}
    explicit pvar_t(const std::string &name) : name_(name) {}
    bool is_undef() const { return name_.is_empty(); }
    bool operator==(const pvar_t &other) const { return name_ == other.name_; }
    bool operator!=(const pvar_t &other) const { return name_ != other.name_; }
    bool operator<(const pvar_t &other) const { return name_ < other.name_; }
    dim_idx_t index() const { return name_.index(); }
    operator dim_idx_t() const { return index(); }
    size_t get_hash() const { return ir_utils::get_hash(name_); }
    const pvar_name_t &name() const { return name_; }
    std::string str() const { return name_.str(); }

    IR_DEFINE_DUMP()

    const expr_t &index_var() const;
    const expr_t &var() const;
    static pvar_t from_index_var(const expr_t &ndex_var);
    static pvar_t from_var(const expr_t &var);

    char to_spatial() const;
    int spatial_index() const;

private:
    pvar_name_t name_;
};

namespace pvars {
extern pvar_t g;
extern pvar_t ic;
extern pvar_t id;
extern pvar_t ih;
extern pvar_t iw;
extern pvar_t kd;
extern pvar_t kh;
extern pvar_t kw;
extern pvar_t mb;
extern pvar_t oc;
extern pvar_t od;
extern pvar_t oh;
extern pvar_t ow;
extern pvar_t sd;
extern pvar_t sh;
extern pvar_t sw;
extern pvar_t dd;
extern pvar_t dh;
extern pvar_t dw;
extern pvar_t pd;
extern pvar_t ph;
extern pvar_t pw;
extern pvar_t b;
extern pvar_t m;
extern pvar_t n;
extern pvar_t k;
} // namespace pvars

template <typename ValueT>
class pvar_map_t {
public:
    class iterator_t {
    public:
        iterator_t(typename std::map<pvar_t, ValueT>::const_iterator it)
            : it_(it) {}

        iterator_t &operator++() {
            it_++;
            return *this;
        }
        bool operator!=(const iterator_t &other) const {
            return it_ != other.it_;
        }

        const pvar_t &operator*() const { return it_->first; }

    private:
        typename std::map<pvar_t, ValueT>::const_iterator it_;
    };

    pvar_map_t() = default;
    pvar_map_t(const std::initializer_list<pvar_t> &keys, const ValueT &value) {
        for (auto &k : keys)
            operator[](k) = value;
    }

    pvar_map_t(const std::initializer_list<std::pair<pvar_t, ValueT>> &keys) {
        for (auto &k : keys)
            operator[](k.first) = k.second;
    }

    explicit pvar_map_t(const std::string &s) {
        for (auto &kv : ir_utils::to_string_int_pairs(s)) {
            operator[](pvar_t(kv.first)) = ValueT(kv.second);
        }
    }

    virtual ~pvar_map_t() = default;

    virtual ValueT default_value() const { return ValueT(); }

    bool has(const pvar_t &key) const { return map_.count(key) != 0; }
    iterator_t begin() const { return iterator_t(map_.begin()); }
    iterator_t end() const { return iterator_t(map_.end()); }
    size_t size() const { return map_.size(); }
    bool is_empty() const { return map_.empty(); }
    bool is_valid() const { return is_valid_; }
    bool is_invalid() const { return !is_valid_; }

    void set(const pvar_t &key, const ValueT &value) { map_[key] = value; }
    void set_valid(bool value) { is_valid_ = value; }

    void unset(const pvar_t &key) {
        if (!has(key)) return;
        map_.erase(key);
    }

    std::vector<pvar_t> keys() const {
        std::vector<pvar_t> ret;
        for (auto &key : *this)
            ret.push_back(key);
        return ret;
    }

    const ValueT &operator[](const pvar_t &key) const {
        gpu_assert(has(key)) << "Key not found: " << key;
        return map_.at(key);
    }

    ValueT &operator[](const pvar_t &key) {
        if (!has(key)) set(key, default_value());
        return map_[key];
    }

    template <typename DerivedT>
    DerivedT with_impl(const pvar_t &key, const ValueT &value) const {
        DerivedT ret = static_cast<const DerivedT &>(*this);
        ret[key] = value;
        return ret;
    }

    const ValueT &at(const pvar_t &key) const { return operator[](key); }
    const ValueT &at(size_t idx) const {
        return operator[](into<dim_idx_t>(idx));
    }
    ValueT get(const pvar_t &key, const ValueT &_default) const {
        if (!has(key)) return _default;
        return at(key);
    }
    ValueT get(const pvar_t &key) const { return get(key, default_value()); }
    void erase(const pvar_t &key) { map_.erase(key); }

    std::vector<ValueT> values() const {
        std::vector<ValueT> ret(size());
        for (size_t i = 0; i < size(); i++)
            ret[i] = at(i);
        return ret;
    }

    std::unordered_map<std::string, dim_t> to_string_map() const {
        std::unordered_map<std::string, dim_t> ret;
        for (auto &kv : map_)
            ret[kv.first.str()] = kv.second;
        return ret;
    }

    pvar_map_t operator|(const pvar_map_t &other) const {
        pvar_map_t ret = *this;
        for (auto &kv : other.map_) {
            auto it = map_.find(kv.first);
            if (it != map_.end()) {
                gpu_assert(it->second == kv.second);
                continue;
            }
            ret[kv.first] = kv.second;
        }
        return ret;
    }

    bool operator==(const pvar_map_t &other) const {
        if (size() != other.size()) return false;
        auto it1 = map_.begin();
        auto it2 = other.map_.begin();
        for (size_t i = 0; i < size(); i++) {
            if (it1->first != it2->first) return false;
            if (!ir_utils::is_equal_helper_t<ValueT, ValueT>::call(
                        it1->second, it2->second))
                return false;
            it1++;
            it2++;
        }
        return true;
    }

    bool operator!=(const pvar_map_t &other) const {
        return !operator==(other);
    }

    pvar_map_t drop_defaults() const {
        pvar_map_t ret;
        for (auto &d : *this) {
            if (ir_utils::is_equal_helper_t<ValueT, ValueT>::call(
                        at(d), default_value()))
                continue;
            ret[d] = at(d);
        }
        return ret;
    }

    size_t get_hash() const { return ir_utils::get_hash(map_); }

    void stringify(std::ostream &out) const {
        gpu_assert(is_valid());
        if (is_empty()) {
            out << "x";
            return;
        }
        for (auto &d : *this) {
            auto &value = operator[](d);
            out << d << value;
        }
    }

    void parse(std::istream &in) {
        auto s = stream_parse<std::string>(in);
        if (s == "x") return;
        for (auto &kv : ir_utils::to_string_int_pairs(s)) {
            operator[](pvar_t(kv.first)) = ValueT(kv.second);
        }
    }

    std::string str_impl(bool multiline) const {
        if (is_invalid()) return "(invalid)";
        if (is_empty()) return "x";
        ostringstream_t oss;
        bool is_first = true;
        for (auto &kv : map_) {
            auto &p = kv.first;
            auto &value = kv.second;
            if (multiline) {
                if (!is_first) oss << std::endl;
                oss << std::setw(4) << p << ": "
                    << ir_utils::str_helper_t<ValueT>::call(value);
                is_first = false;
            } else {
                oss << p << ir_utils::str_helper_t<ValueT>::call(value);
            }
        }
        return oss.str();
    }

    virtual std::string str() const {
        return str_impl(/*multiline=*/!std::is_integral<ValueT>::value);
    }

    IR_DEFINE_DUMP()

private:
    bool is_valid_ = true;
    std::map<pvar_t, ValueT> map_;
};

class tile_t : public pvar_map_t<dim_t> {
public:
    using pvar_map_t<dim_t>::pvar_map_t;

    tile_t() = default;
    tile_t(size_t size) : tile_t(std::vector<dim_t>(size, 1)) {}
    tile_t(const std::vector<dim_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(pvar_t(std::string(1, into<char>('a' + i))), values[i]);
    }

    dim_t default_value() const override { return 1; }

    dim_t elems() const {
        dim_t ret = 1;
        for (auto &d : *this)
            ret *= at(d);
        return ret;
    }

    bool try_factor(const pvar_t &key, dim_t factor) {
        if (factor == 1) return true;
        if (!has(key)) return false;
        dim_t &value = operator[](key);
        if (value % factor != 0) return false;
        value /= factor;
        return true;
    }

    bool is_equal(const std::vector<dim_t> &dims) const {
        if (size() != dims.size()) return false;
        for (size_t i = 0; i < size(); i++) {
            if (at(i) != dims[i]) return false;
        }
        return true;
    }

    bool is_divisible(const tile_t &other) const {
        if (size() != other.size()) return false;
        for (size_t i = 0; i < size(); i++) {
            if (at(i) % other.at(i) != 0) return false;
        }
        return true;
    }

#if __cplusplus >= 202002L
    bool operator==(const tile_t &other) const = default;
#endif

    std::string str() const override { return str_impl(/*multiline=*/false); }

    static tile_t invalid() {
        tile_t ret;
        ret.set_valid(false);
        return ret;
    }
};

class coord_t;

// Coordinate with integer values.
class icoord_t : public pvar_map_t<dim_t> {
public:
    using pvar_map_t<dim_t>::pvar_map_t;

    icoord_t() = default;
    icoord_t(size_t size) : icoord_t(std::vector<dim_t>(size, 0)) {}
    icoord_t(const std::vector<dim_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(pvar_t(std::string(1, into<char>('a' + i))), values[i]);
    }
    icoord_t(const coord_t &coord);
    dim_t default_value() const override { return 0; }
    static icoord_t invalid() {
        icoord_t ret;
        ret.set_valid(false);
        return ret;
    }
};

// Coordinate with expression values.
class coord_t : public pvar_map_t<expr_t> {
public:
    using pvar_map_t<expr_t>::pvar_map_t;

    coord_t() = default;
    coord_t(size_t size) : coord_t(std::vector<expr_t>(size, 0)) {}
    coord_t(const std::vector<expr_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(pvar_t(std::string(1, into<char>('a' + i))), values[i]);
    }
    coord_t(const icoord_t &icoord) {
        for (auto &d : icoord)
            set(d, icoord[d]);
    }
    expr_t default_value() const override { return expr_t(0); }
    static coord_t invalid() {
        coord_t ret;
        ret.set_valid(false);
        return ret;
    }
    coord_t with(const pvar_t &key, const expr_t &value) const {
        return pvar_map_t<expr_t>::with_impl<coord_t>(key, value);
    }
};

inline icoord_t::icoord_t(const coord_t &coord) {
    for (auto &d : coord)
        set(d, to_cpp<dim_t>(coord.at(d)));
}

inline coord_t operator+(const coord_t &a, const coord_t &b) {
    coord_t ret;
    for (auto &d : a) {
        ret[d] = a.get(d, expr_t(0)) + b.get(d, expr_t(0));
    }
    for (auto &d : b) {
        if (ret.has(d)) continue;
        ret[d] = a.get(d, expr_t(0)) + b.get(d, expr_t(0));
    }
    return ret;
}

inline icoord_t operator+(const icoord_t &a, const icoord_t &b) {
    icoord_t ret;
    for (auto &d : a) {
        ret[d] = a.get(d, 0) + b.get(d, 0);
    }
    for (auto &d : b) {
        if (ret.has(d)) continue;
        ret[d] = a.get(d, 0) + b.get(d, 0);
    }
    return ret;
}

struct tile_coord_t {
    tile_t tile;
    coord_t coord;

    tile_coord_t() = default;
    tile_coord_t(const tile_t &tile, const coord_t &coord = {})
        : tile(tile), coord(coord) {}
    bool is_invalid() const { return tile.is_invalid() || coord.is_invalid(); }
    size_t size() const { return tile.size(); }
    dim_t elems() const { return tile.elems(); }
    bool has_zero_coord() const {
        for (auto &d : coord) {
            if (!is_zero(coord.at(d))) return false;
        }
        return true;
    }

    tile_coord_t sub(
            const tile_t &sub_tile, const coord_t &sub_coord = {}) const {
        coord_t new_coord = coord;
        for (auto &d : sub_coord)
            new_coord[d] += sub_coord[d];
        return tile_coord_t(sub_tile, new_coord);
    }

    tile_coord_t sub(const tile_coord_t &sub_tile_coord) const {
        return sub(sub_tile_coord.tile, sub_tile_coord.coord);
    }

    std::string str() const {
        ostringstream_t oss;
        oss << "tile: " << tile.str();
        if (!has_zero_coord()) oss << " coord: " << coord.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static tile_coord_t invalid() {
        return tile_coord_t(tile_t::invalid(), coord_t::invalid());
    }
};

template <typename F>
void for_each(const tile_t &tile, const F &f) {
    ir_utils::for_each(tile.values(),
            [&](const std::vector<dim_t> &idxs) { f(icoord_t(idxs)); });
}

template <typename T>
bool has_spatial(const pvar_map_t<T> &map, char spatial) {
    for (auto &d : map) {
        if (d.to_spatial() == spatial) return true;
    }
    return false;
}

bool is_input_spatial(const pvar_t &pvar);
bool is_output_spatial(const pvar_t &pvar);
bool is_kernel_spatial(const pvar_t &pvar);
bool is_dilation(const pvar_t &pvar);
bool is_padding(const pvar_t &pvar);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::gpu::intel::jit::pvar_t> {
    size_t operator()(const dnnl::impl::gpu::intel::jit::pvar_t &pvar) const {
        return pvar.get_hash();
    }
};
} // namespace std

#endif
