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
#ifndef GPU_INTEL_JIT_DSL_TENSOR_HPP
#define GPU_INTEL_JIT_DSL_TENSOR_HPP

#include <cstdint>
#include <cstring>
#include <string>

#include "gpu/intel/jit/dsl/decl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace dsl {

// The idx_t class represents an index into a tensors basis with functionality
// for logging. This class has two modes of operation: as a key in a hash map or
// as an index into contiguous containers. When used as a key, idx_t can encode
// a custom name which is especially useful when the basis dimension is shared
// among tensor objects. When used for contiguous containers, the dimension name
// is implicitly determined from the integer index value.
class idx_t {
public:
    idx_t() = default;
    idx_t(size_t idx) : name_(idx) {}
    explicit idx_t(const std::string &name) : name_(name) {}
    bool is_undef() const { return name_.is_empty(); }
    bool operator==(const idx_t &other) const { return name_ == other.name_; }
    bool operator!=(const idx_t &other) const { return name_ != other.name_; }
    bool operator<(const idx_t &other) const { return name_ < other.name_; }
    size_t index() const { return name_.index(); }
    operator size_t() const { return index(); }
    size_t get_hash() const { return name_.get_hash(); }
    std::string str() const { return name_.str(); }

private:
    class name_t {
    public:
        name_t() = default;
        explicit name_t(size_t idx) { data_[0] = into<char>('a' + idx); }
        explicit name_t(const std::string &s) {
            gpu_assert(!s.empty() && s.length() <= max_len);
            s.copy(data_, s.length());
        }

        bool operator==(const name_t &other) const {
            return numeric_value() == other.numeric_value();
        }

        bool operator!=(const name_t &other) const {
            return !operator==(other);
        }

        bool operator<(const name_t &other) const {
            return std::strncmp(data_, other.data_, max_len) < 0;
        }

        size_t index() const {
            gpu_assert(length() == 1);
            gpu_assert('a' <= data_[0] && data_[0] <= 'z');
            return data_[0] - 'a';
        }

        size_t length() const { return std::strlen(data_); }
        bool is_empty() const { return data_[0] == 0; }
        size_t get_hash() const {
            return std::hash<uint64_t> {}(numeric_value());
        };
        std::string str() const { return data_; }

    private:
        uint64_t numeric_value() const {
            uint64_t ret;
            static_assert(sizeof(*this) == sizeof(ret),
                    "name_t is intended to be small enough to fit in a single "
                    "register");
            memcpy(&ret, this, sizeof(ret));
            return ret;
        }
        static constexpr uint32_t max_len = 7;
        char data_[max_len + 1] = {};
    };

    name_t name_;
};

class stride_t {
public:
    constexpr stride_t(int64_t stride = undefined_stride) : stride_(stride) {}

    constexpr bool operator==(const stride_t &other) const {
        return stride_ == other.stride_;
    }

    constexpr bool operator!=(const stride_t &other) const {
        return !operator==(other);
    }

    stride_t &operator*=(const stride_t &other) {
        gpu_assert(!(is_undefined() || other.is_undefined()));
        if (is_unknown() || other.is_unknown())
            *this = stride_t::unknown();
        else
            stride_ *= other.stride_;
        return *this;
    }
    stride_t &operator/=(const stride_t &other) {
        gpu_assert(!(is_undefined() || other.is_undefined()));
        if (is_unknown() || other.is_unknown())
            *this = stride_t::unknown();
        else
            stride_ /= other.stride_;
        return *this;
    }
    stride_t &operator%=(const stride_t &other) {
        gpu_assert(!(is_undefined() || other.is_undefined()));
        if (is_unknown() || other.is_unknown())
            *this = stride_t::unknown();
        else
            stride_ %= other.stride_;
        return *this;
    }

    size_t get_hash() const { return std::hash<int64_t> {}(stride_); }

    explicit operator int64_t() const {
        gpu_assert(is_fixed());
        return stride_;
    }
    explicit operator int() const {
        gpu_assert(is_fixed());
        return into<int>(stride_);
    }

    constexpr bool is_fixed() const { return !is_unknown() && !is_undefined(); }
    constexpr bool is_unknown() const { return stride_ == unknown_stride; }
    constexpr bool is_undefined() const { return stride_ == undefined_stride; }

    static constexpr stride_t unknown() { return stride_t(unknown_stride); }
    static constexpr stride_t undefined() { return stride_t(undefined_stride); }
    static constexpr stride_t max() { return stride_t(max_stride); }

    std::string str() const {
        if (is_undefined()) return "(invalid)";
        if (is_unknown()) return "(unknown)";
        return std::to_string(stride_);
    }

private:
    // Both negative sentinels: won't interfere with valid strides
    static constexpr int64_t unknown_stride
            = std::numeric_limits<int64_t>::min();
    static constexpr int64_t undefined_stride = unknown_stride + 1;
    static constexpr int64_t max_stride = std::numeric_limits<int64_t>::max();

    int64_t stride_ = undefined_stride;
};

static_assert(sizeof(stride_t) == 8, "stride_t is unexpectedly large");

inline stride_t operator*(stride_t a, stride_t b) {
    return a *= b;
}
inline stride_t operator/(stride_t a, stride_t b) {
    return a /= b;
}
inline stride_t operator%(stride_t a, stride_t b) {
    return a %= b;
}
inline bool operator<(stride_t a, stride_t b) {
    gpu_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) < int64_t(b);
}
inline bool operator<=(stride_t a, stride_t b) {
    gpu_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) <= int64_t(b);
}
inline bool operator>(stride_t a, stride_t b) {
    gpu_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) > int64_t(b);
}
inline bool operator>=(stride_t a, stride_t b) {
    gpu_assert(!a.is_undefined() && !b.is_undefined());
    if (a.is_unknown() || b.is_unknown()) return false;
    return int64_t(a) >= int64_t(b);
}

// The idx_map_t is a helper class to simplify constructing hash maps using
// idx_t as a key. On top of the normal hash map interfaces, this helper class
// includes features for logging and hashing.
template <typename ValueT>
class idx_map_t {
public:
    class iterator_t {
    public:
        iterator_t(typename std::map<idx_t, ValueT>::const_iterator it)
            : it_(it) {}

        iterator_t &operator++() {
            it_++;
            return *this;
        }
        bool operator!=(const iterator_t &other) const {
            return it_ != other.it_;
        }

        const idx_t &operator*() const { return it_->first; }

    private:
        typename std::map<idx_t, ValueT>::const_iterator it_;
    };

    idx_map_t() = default;
    idx_map_t(const std::initializer_list<idx_t> &keys, const ValueT &value) {
        for (auto &k : keys)
            operator[](k) = value;
    }

    idx_map_t(const std::initializer_list<std::pair<idx_t, ValueT>> &keys) {
        for (auto &k : keys)
            operator[](k.first) = k.second;
    }

    explicit idx_map_t(const std::string &s) {
        for (auto &kv : ir_utils::to_string_int_pairs(s)) {
            operator[](idx_t(kv.first)) = ValueT(kv.second);
        }
    }

    virtual ~idx_map_t() = default;

    virtual ValueT default_value() const { return ValueT(); }

    bool has(const idx_t &key) const { return map_.count(key) != 0; }
    iterator_t begin() const { return iterator_t(map_.begin()); }
    iterator_t end() const { return iterator_t(map_.end()); }
    size_t size() const { return map_.size(); }
    bool is_empty() const { return map_.empty(); }
    void set(const idx_t &key, const ValueT &value) { map_[key] = value; }

    void unset(const idx_t &key) {
        if (!has(key)) return;
        map_.erase(key);
    }

    std::vector<idx_t> keys() const {
        std::vector<idx_t> ret;
        for (auto &key : *this)
            ret.push_back(key);
        return ret;
    }

    const ValueT &operator[](const idx_t &key) const {
        gpu_assert(has(key)) << "Key not found: " << key;
        return map_.at(key);
    }

    ValueT &operator[](const idx_t &key) {
        if (!has(key)) set(key, default_value());
        return map_[key];
    }

    template <typename DerivedT>
    DerivedT with_impl(const idx_t &key, const ValueT &value) const {
        DerivedT ret = static_cast<const DerivedT &>(*this);
        ret[key] = value;
        return ret;
    }

    const ValueT &at(const idx_t &key) const { return operator[](key); }
    ValueT get(const idx_t &key, const ValueT &_default) const {
        if (!has(key)) return _default;
        return at(key);
    }
    ValueT get(const idx_t &key) const { return get(key, default_value()); }
    void erase(const idx_t &key) { map_.erase(key); }

    std::vector<ValueT> values() const {
        std::vector<ValueT> ret(size());
        for (size_t i = 0; i < size(); i++)
            ret[i] = at(i);
        return ret;
    }

    std::unordered_map<std::string, int64_t> to_string_map() const {
        std::unordered_map<std::string, int64_t> ret;
        for (auto &kv : map_)
            ret[kv.first.str()] = kv.second;
        return ret;
    }

    idx_map_t operator|(const idx_map_t &other) const {
        idx_map_t ret = *this;
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

    bool operator==(const idx_map_t &other) const {
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

    bool operator!=(const idx_map_t &other) const { return !operator==(other); }

    idx_map_t drop_defaults() const {
        idx_map_t ret;
        for (auto &d : *this) {
            if (ir_utils::is_equal_helper_t<ValueT, ValueT>::call(
                        at(d), default_value()))
                continue;
            ret[d] = at(d);
        }
        return ret;
    }

    size_t get_hash() const { return ir_utils::get_hash(map_); }

    void parse(std::istream &in) {
        auto s = stream_parse<std::string>(in);
        if (s == "x") return;
        for (auto &kv : ir_utils::to_string_int_pairs(s)) {
            operator[](idx_t(kv.first)) = ValueT(kv.second);
        }
    }

    std::string str_impl(bool multiline) const {
        if (is_empty()) return "x";
        ostringstream_t oss;
        bool is_first = true;
        for (auto &kv : map_) {
            auto &p = kv.first;
            auto &value = kv.second;
            if (multiline) {
                if (!is_first) oss << std::endl;
                oss << std::setw(4) << p << ": " << value;
                is_first = false;
            } else {
                oss << p << value;
            }
        }
        return oss.str();
    }

    virtual std::string str() const {
        return str_impl(/*multiline=*/!std::is_integral<ValueT>::value);
    }

private:
    std::map<idx_t, ValueT> map_;
};

class tile_t : public idx_map_t<int64_t> {
public:
    using idx_map_t<int64_t>::idx_map_t;

    tile_t() = default;
    tile_t(const std::vector<int64_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(idx_t(std::string(1, into<char>('a' + i))), values[i]);
    }

    int64_t default_value() const override { return 1; }

    int64_t elems() const {
        int64_t ret = 1;
        for (auto &d : *this)
            ret *= at(d);
        return ret;
    }

    bool try_factor(const idx_t &key, int64_t factor) {
        if (factor == 1) return true;
        if (!has(key)) return false;
        int64_t &value = operator[](key);
        if (value % factor != 0) return false;
        value /= factor;
        return true;
    }

    bool is_divisible(const tile_t &other) const {
        for (auto &i : other) {
            if (at(i) % other.at(i) != 0) return false;
        }
        return true;
    }

    tile_t with(const idx_t &key, dim_t value) const {
        return idx_map_t<int64_t>::with_impl<tile_t>(key, value);
    }

#if __cplusplus >= 202002L
    bool operator==(const tile_t &other) const = default;
#endif

    std::string str() const override { return str_impl(/*multiline=*/false); }
};

// Coordinate with integer values.
class icoord_t : public idx_map_t<int64_t> {
public:
    using idx_map_t<int64_t>::idx_map_t;

    icoord_t() = default;
    icoord_t(size_t size) : icoord_t(std::vector<int64_t>(size, 0)) {}
    icoord_t(const std::vector<int64_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(i, values[i]);
    }
    int64_t default_value() const override { return 0; }
};

// Coordinate with expression values.
class coord_t : public idx_map_t<expr_t> {
public:
    using idx_map_t<expr_t>::idx_map_t;

    coord_t() = default;
    coord_t(size_t size) : coord_t(std::vector<expr_t>(size, 0)) {}
    coord_t(const std::vector<expr_t> &values) {
        for (size_t i = 0; i < values.size(); i++)
            set(i, values[i]);
    }
    coord_t(const icoord_t &icoord) {
        for (auto &d : icoord)
            set(d, icoord[d]);
    }
    expr_t default_value() const override { return expr_t(0); }
    coord_t with(const idx_t &key, const expr_t &value) const {
        return idx_map_t<expr_t>::with_impl<coord_t>(key, value);
    }
};

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

} // namespace dsl
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
