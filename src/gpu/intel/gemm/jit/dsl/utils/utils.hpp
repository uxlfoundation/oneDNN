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

#ifndef GEMMSTONE_DSL_UTILS_UTILS_HPP
#define GEMMSTONE_DSL_UTILS_UTILS_HPP

#include <iostream>
#include <string>

#include "gemmstone/config.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

using ngen::utils::rounddown_pow2;
using ngen::utils::roundup_pow2;

namespace dsl {

template <typename derived_type, typename base_type>
inline derived_type downcast(base_type *base) {
    assume(dynamic_cast<derived_type>(base) == base);
    return static_cast<derived_type>(base);
}

class error_stream_t {
public:
    error_stream_t(const char *file, int line, const char *assert_msg)
        : data_(new data_t(file, line, assert_msg)) {}

    // This is to be able use a steam object in short-circuit evaluation with
    // booleans, see below.
    operator bool() const { return true; }

    template <typename T>
    error_stream_t &operator<<(const T &t) {
        data_->out << t;
        return *this;
    }

    ~error_stream_t() noexcept(false) {
        if (data_ == nullptr) return;

        printf("%s\n", data_->out.str().c_str());
#ifdef GPU_ABORT_ON_ERROR
        std::abort();
#else
        auto err = std::runtime_error(data_->out.str());
        delete data_;
        data_ = nullptr;

        // This is techincally unsafe. Since error_stream_t is only used in
        // debug builds and since it is only used by ir_assert() which signals
        // an ill-defined program state, nested throws is not a concern.
        throw err; // NOLINT
#endif
    }

private:
    struct data_t {
        data_t(const char *file, int line, const char *assert_msg)
            : file(file), line(line) {}

        const char *file;
        int line;
        ostringstream_t out;
    };

    data_t *data_;
};

#if !defined(NDEBUG) || defined(GEMMSTONE_ASSERTIONS)
#define dsl_assert(cond) \
    !(cond) && gemmstone::dsl::error_stream_t(__FILE__, __LINE__, #cond)
#else
#define dsl_assert(cond) \
    (false) && !(cond) \
            && gemmstone::dsl::error_stream_t(__FILE__, __LINE__, #cond)
#endif

#define dsl_error() dsl_assert(false) << "Not Expected. "

const std::string &to_string(ngen::ProductFamily family);
const std::string &to_string(ngen::HW hw);
std::string to_string(const ngen::Product &product);

template <typename T>
inline void maybe_unused(const T &x) {
    (void)(x);
}

namespace utils {

template <typename T, typename U, typename = void>
struct is_equal_helper_t {
    static bool call(const T &t, const U &u) { return t == u; }
};

template <typename T, typename U>
struct is_equal_helper_t<T, U,
        decltype(std::declval<T>().is_equal(std::declval<U>()), void())> {
    static bool call(const T &t, const U &u) { return t.is_equal(u); }
};

// Checks equality of objects:
// 1. Uses t.is_equal(u) if is_equal() is available
// 2. Uses (t == u) otherwise
template <typename T, typename U>
bool is_equal(const T &t, const U &u) {
    return is_equal_helper_t<T, U>::call(t, u);
}

// Checks equality of vector elements.
template <typename T, typename U>
bool is_equal(const std::vector<T> &a, const std::vector<U> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (!utils::is_equal(a[i], b[i])) return false;
    return true;
}

// Checks identity of vector elements.
template <typename T, typename U>
bool is_same(const std::vector<T> &a, const std::vector<U> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (!a[i].is_same(b[i])) return false;
    return true;
}
template <typename ContainerT>
struct seq_print_helper_t {
    seq_print_helper_t(const ContainerT &v, const std::string &sep, int width)
        : v(v), sep(sep), width(width) {}

    const ContainerT &v;
    const std::string sep;
    int width;
};

template <typename T>
seq_print_helper_t<T> make_seq_print_helper(
        const T &v, const std::string &sep = ", ", int width = 0) {
    return seq_print_helper_t<T>(v, sep, width);
}

template <typename T>
inline std::ostream &operator<<(
        std::ostream &out, const seq_print_helper_t<T> &seq) {
    for (auto it = seq.v.begin(); it != seq.v.end(); it++) {
        out << (it != seq.v.begin() ? seq.sep : "") << std::setw(seq.width)
            << *it;
    }
    return out;
}

} // namespace utils

inline bool stream_try_match(std::istream &in, const std::string &s) {
    in >> std::ws;
    auto pos = in.tellg();
    bool ok = true;
    for (auto &c : s) {
        if (in.get() != c || in.fail()) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        in.clear();
        in.seekg(pos);
    }
    return ok;
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
