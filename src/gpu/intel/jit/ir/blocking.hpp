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

#ifndef GPU_INTEL_JIT_IR_BLOCKING_HPP
#define GPU_INTEL_JIT_IR_BLOCKING_HPP

#include "gpu/intel/jit/ir/core.hpp"
#include "gpu/intel/jit/ir/problem.hpp"

#include <set>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class blocking_t {
public:
    int simd() const { return simd_; }
    const tile_t &loop() const { return loop_; }
    const tile_t &thread_group() const { return thread_group_; }
    const tile_t &iter() const { return iter_; }

    dim_t loop_dim(const pvar_t &d) const { return loop_[d]; }
    dim_t thread_group_dim(const pvar_t &d) const { return thread_group_[d]; }
    dim_t iter_dim(const pvar_t &d) const { return iter_[d]; }

    void set_simd(int simd) { simd_ = simd; }
    void set_loop(const pvar_t &d, dim_t value) { loop_[d] = value; }
    void set_thread_group(const pvar_t &d, dim_t value) {
        thread_group_[d] = value;
    }
    void set_iter(const pvar_t &d, dim_t value) { iter_[d] = value; }

    bool is_empty() const {
        return loop_.is_empty() && thread_group_.is_empty() && iter_.is_empty();
    }
    bool is_spatial() const {
        for (const auto &d : {pvars::iw, pvars::ow}) {
            if (iter_.has(d) && iter_[d] != 1) return true;
        }
        return false;
    }

    void unset(const pvar_t &d) {
        if (loop_.has(d)) loop_[d] = 1;
        if (thread_group_.has(d)) thread_group_[d] = 1;
        if (iter_.has(d)) iter_[d] = 1;
    }

    bool operator==(const blocking_t &other) const {
        return (loop_ == other.loop_) && (thread_group_ == other.thread_group_)
                && (iter_ == other.iter_);
    }

    void stringify(std::ostream &out) const {
        out << "simd=" << simd_;
        out << " l=";
        loop_.stringify(out);
        out << " T=";
        thread_group_.stringify(out);
        out << " i=";
        iter_.stringify(out);
    }

    void parse(std::istream &in) {
        stream_match(in, "simd=");
        simd_ = stream_parse<int>(in);
        stream_match(in, "l=");
        loop_.parse(in);
        stream_match(in, "T=");
        thread_group_.parse(in);
        stream_match(in, "i=");
        iter_.parse(in);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(loop_, thread_group_, iter_);
    }

    std::string str(bool csv = false) const {
        ostringstream_t oss;
        if (csv) {
            oss << simd_;
            oss << "," << loop_;
            oss << "," << thread_group_;
            oss << "," << iter_;
        } else {
            oss << "simd=" << simd_;
            oss << " l=" << loop_;
            oss << " T=" << thread_group_;
            oss << " i=" << iter_;
        }
        return oss.str();
    }

    // Returns the ratio of all operations (with padding) to "useful" operations
    double get_efficiency(const tile_t &shape) const {
        double ret = 1;
        for (auto &d : shape) {
            dim_t loop = loop_.get(d, 1);
            dim_t tg = thread_group_.get(d, 1);
            dim_t iter = iter_.get(d, 1);
            dim_t size = shape[d];
            dim_t size_padded = utils::rnd_up(size, loop * tg * iter);
            if (size_padded != size) ret *= double(size) / size_padded;
        }
        return ret;
    }

    IR_DEFINE_DUMP()

private:
    int simd_ = 0;
    tile_t loop_;
    tile_t thread_group_;
    tile_t iter_;
};

struct blocking_hash_t {
    size_t operator()(const blocking_t &b) const { return b.get_hash(); }
};

// Flags specifying blocking restrictions for a prb dimension.
enum class tile_flags_t : uint32_t {
    undef = 0,
    // Dimension participates in loop blocking.
    loop = (1 << 0),
    // Dimension participates in thread group blocking.
    thread_group = (1 << 1),
    // Dimension participates in iteration blocking.
    iter = (1 << 2),
    // Loop block spans the remaining dimension.
    loop_span = (1 << 3),
    // Loop block is fully unrolled.
    loop_iter_unroll = (1 << 4),
};

GPU_DEFINE_BIT_MASK_ENUM_OPS(tile_flags_t)

// Divisibility restrictions for a prb dimension.
struct div_info_t {
    // Iteration block must be divisible by this value.
    int iter_unit = 1;
    // (Iteration block) x (loop unroll) must be divisible by this value.
    int unroll_unit = 1;

    void set_iter_unit(int new_unit) {
        iter_unit = math::lcm(iter_unit, new_unit);
    }
    void set_unroll_unit(int new_unit) {
        unroll_unit = math::lcm(unroll_unit, new_unit);
    }

    bool is_iter_ok(dim_t blk) const {
        if (iter_unit != 1 && blk % iter_unit != 0) return false;
        if (iter_unit != 1 && !math::is_pow2(blk)) return false;
        return true;
    }
};

// Blocking restrictions for a prb dimension.
struct tile_info_t {
    tile_info_t() = default;
    tile_info_t(const pvar_t &dim) : dim(dim) {}
    void add(tile_flags_t f) { flags = flags | f; }
    void remove(tile_flags_t f) { flags = flags & ~f; }
    void set_iter_unit(int unit) { div_info.set_iter_unit(unit); }
    void set_unroll_unit(int unit) { div_info.set_unroll_unit(unit); }
    void set_min_iter_block(int block, int pow2_block = 0) {
        min_iter_blk = block;
        if (pow2_block != 0) min_iter_pow2_blk = pow2_block;
    }

    std::vector<int> iter_blocks(dim_t size) const;
    std::vector<int> thread_group_blocks(dim_t size) const;
    std::vector<dim_t> loop_blocks(dim_t size, int iter_blk) const;

    static bool block_ok(dim_t size, int blk, int target_eff) {
        dim_t size_padded = utils::rnd_up(size, blk);
        double eff = size / (double)size_padded;
        return eff * 100 >= target_eff;
    }

    static std::vector<dim_t> get_factors(dim_t n);
    static std::vector<dim_t> get_loop_blocks(dim_t n);

    pvar_t dim;
    tile_flags_t flags = tile_flags_t::undef;
    div_info_t div_info;

    int min_iter_blk = default_min_iter_blk;
    int min_iter_pow2_blk = default_min_iter_pow2_blk;
    int max_iter_blk = default_max_iter_blk;
    int max_thread_group_blk = default_max_thread_group_blk;

    static const int default_min_iter_blk = 6;
    static const int default_min_iter_pow2_blk = 8;
    static const int default_max_iter_blk = 64;
    static const int default_max_thread_group_blk = 16;
};

// Tile levels.
enum class level_t {
    undef = 0,
    loop,
    thread_group,
    iter,
};

class level_tile_t {
public:
    bool has(level_t level) const {
        switch (level) {
            case level_t::loop: return loop != 0;
            case level_t::thread_group: return thread_group != 0;
            case level_t::iter: return iter != 0;
            default: gpu_error_not_expected();
        }
        return false;
    }

    std::string str() const {
        if (utils::everyone_is(0, loop, thread_group, iter)) return "x";
        ostringstream_t oss;
        if (loop != 0) oss << "l" << loop;
        if (thread_group != 0) oss << "T" << thread_group;
        if (iter != 0) oss << "i" << iter;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    dim_t loop = 0;
    int thread_group = 0;
    int iter = 0;
};

void get_level_tiles(
        dim_t size, const tile_info_t &info, std::vector<level_tile_t> &ret);

class level_tile_set_t {
public:
    level_tile_set_t(const std::vector<std::vector<level_tile_t>> &tiles,
            const std::vector<int> &deps, const std::vector<pvar_t> &dims)
        : tiles_(tiles), deps_(deps), dims_(dims) {}

    int count() const;
    std::vector<blocking_t> product(int simd) const;
    std::vector<blocking_t> sample(int target,
            const std::function<bool(const blocking_t &)> &is_ok, int simd,
            int tries_mult_bound = 5) const;

private:
    static void set(
            blocking_t &blk, const pvar_t &dim, const level_tile_t &tile);

    void product_impl(int idx, std::vector<int> &cur_idxs, blocking_t &blk,
            std::vector<blocking_t> &ret) const;

    std::vector<level_tile_t> sample(ir_utils::fast_random_t &r) const;

    std::vector<std::vector<level_tile_t>> tiles_;
    std::vector<int> deps_;
    std::vector<pvar_t> dims_;
};

// Blocking scheme describing recipes to generate blockings.
class blocking_scheme_t {
public:
    virtual ~blocking_scheme_t() = default;
    blocking_scheme_t() = default;
    blocking_scheme_t(const std::string &s) {
        gpu_assert(s[s.length() - 1] == ']');
        auto parts = gpu_utils::split(s.substr(0, s.length() - 1), "],");
        for (auto &p : parts) {
            auto p_parts = gpu_utils::split(p, ":");
            auto &key = p_parts[0];
            auto &vec = p_parts[1];
            gpu_assert(vec[0] == '[');
            auto s_dims
                    = gpu_utils::split(vec.substr(1, vec.length() - 1), ",");
            for (auto &s : s_dims)
                set(key, s);
        }
    }

    virtual level_tile_set_t make_level_tile_set(
            const tile_t &padded_shape) const {
        const auto all_dims = dims();
        const int ndims = int(all_dims.size());
        const std::vector<int> deps(ndims, -1);
        std::vector<std::vector<level_tile_t>> tiles(ndims);

        for (int i = 0; i < ndims; i++) {
            auto &d = all_dims[i];
            get_level_tiles(padded_shape[d], tile_info(d), tiles[i]);
        }
        return level_tile_set_t(tiles, deps, all_dims);
    }

    tile_info_t &tile_info(const pvar_t &d) {
        auto it = tile_infos_.find(d);
        if (it != tile_infos_.end()) return it->second;
        auto &info = tile_infos_[d];
        info = tile_info_t(d);
        return info;
    }

    const tile_info_t &tile_info(const pvar_t &d) const {
        return tile_infos_.at(d);
    }

    std::vector<pvar_t> dims() const {
        std::set<pvar_t> dims;
        for (auto *t : {&loop_, &thread_group_, &iter_}) {
            for (auto &d : t->keys()) {
                dims.insert(d);
            }
        }
        return std::vector<pvar_t>(dims.begin(), dims.end());
    }

    std::string str() const {
        ostringstream_t oss;
        oss << "l:" << loop_;
        oss << " T:" << thread_group_;
        oss << " i:" << iter_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    void set(const std::string &s_tile, const std::string &_s_dim) {
        gpu_assert(!_s_dim.empty());
        bool no_min_check = (_s_dim[0] == '#');
        const auto &s_dim = no_min_check ? _s_dim.substr(1) : _s_dim;
        auto d = pvar_t(s_dim);
        if (no_min_check) gpu_assert(s_tile == "i");
        if (s_tile == "i") {
            add_iter_dim(d);
            if (no_min_check) tile_info(d).set_min_iter_block(1);
        } else if (s_tile == "T") {
            add_thread_group_dim(d);
        } else if (s_tile == "l") {
            add_loop_dim(d);
        } else if (s_tile == "ls") {
            add_loop_dim_with_span(d);
        } else if (s_tile == "li") {
            add_loop_dim_with_iter_unroll(d);
        } else {
            gpu_error_not_expected() << s_tile;
        }
    }

    void add_loop_dim(const pvar_t &d) {
        loop_[d] = 1;
        auto &info = tile_info(d);
        info.add(tile_flags_t::loop);
    }

    void add_loop_dim_with_span(const pvar_t &d) {
        add_loop_dim(d);
        tile_info(d).add(tile_flags_t::loop_span);
    }

    void add_loop_dim_with_iter_unroll(const pvar_t &d) {
        add_loop_dim(d);
        tile_info(d).add(tile_flags_t::loop_iter_unroll);
    }

    void add_thread_group_dim(const pvar_t &d) {
        thread_group_[d] = 1;
        auto &info = tile_info(d);
        info.add(tile_flags_t::thread_group);
    }

    void add_iter_dim(const pvar_t &d) {
        iter_[d] = 1;
        auto &info = tile_info(d);
        info.add(tile_flags_t::iter);
    }

protected:
    tile_t loop_;
    tile_t thread_group_;
    tile_t iter_;
    std::map<pvar_t, tile_info_t> tile_infos_;
};

template <class blocking_scheme_kind>
class blocking_scheme_list_impl_t {
public:
    blocking_scheme_list_impl_t() : blocking_scheme_list_impl_t(0) {}
    blocking_scheme_list_impl_t(int tune_level) : tune_level_(tune_level) {}
    void add(bool filter, const blocking_scheme_kind &scheme) {
        if ((tune_level_ == 0) && !filter) return;
        schemes_.push_back(scheme);
    }
    const std::vector<blocking_scheme_kind> &get() const { return schemes_; }

private:
    int tune_level_;
    std::vector<blocking_scheme_kind> schemes_;
};

using blocking_scheme_list_t = blocking_scheme_list_impl_t<blocking_scheme_t>;

class blocking_checker_t {
public:
    virtual ~blocking_checker_t() = default;
    virtual void reset_checks() = 0;
    virtual bool relax_checks() = 0;
    virtual bool is_ok(const blocking_t &blk) const = 0;
};

class blocking_generator_t {
public:
    blocking_generator_t(int vec_size, blocking_checker_t &chk,
            const std::vector<level_tile_set_t> &level_tile_sets) {
        for (auto &ts : level_tile_sets)
            generate_all(vec_size, chk, ts);
    }

    std::vector<blocking_t> blockings() const {
        return std::vector<blocking_t>(blockings_.begin(), blockings_.end());
    }

private:
    void generate_all(int vec_size, blocking_checker_t &chk,
            const level_tile_set_t &level_tile_set);

    // TODO: Remove.
    void generate_sample(int vec_size, const blocking_checker_t &chk,
            const level_tile_set_t &level_tile_set);

    std::unordered_set<blocking_t, blocking_hash_t> blockings_;
};

class blocking_params_t {
public:
    static const int bufs_hint_undef = -1;

    blocking_params_t() = default;
    blocking_params_t(
            const blocking_t &blocking, int bufs_hint = bufs_hint_undef)
        : blocking_(blocking), bufs_hint_(bufs_hint) {}

    int id() const { return id_; }
    int bufs_hint() const { return bufs_hint_; }
    bool is_empty() const { return blocking_.is_empty(); }
    const blocking_t &blocking() const { return blocking_; }
    void set_id(int id) { id_ = id; }

    void stringify(std::ostream &out) const {
        blocking_.stringify(out);
        if (bufs_hint_ != -1) out << " bufs=" << bufs_hint_;
    }

    void parse(std::istream &in) {
        // ID is always default in parsed keys.
        id_ = -1;
        blocking_.parse(in);

        if (stream_try_match(in, "bufs=")) {
            bufs_hint_ = stream_parse<int>(in);
        }
    }

    std::string str(bool csv = false) const {
        ostringstream_t oss;
        if (csv) {
            oss << blocking_.str(csv);
            oss << "," << bufs_hint_;
        } else {
            oss << "cfg=\"";
            oss << blocking_.str(csv);
            if (bufs_hint_ == 0) oss << " s=x0 p=x0";
            oss << "\"";
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static std::vector<std::string> csv_keys() {
        return {"simd", "loop", "tg", "iter", "bufs_hint"};
    }

private:
    int id_ = -1;
    blocking_t blocking_;
    int bufs_hint_ = bufs_hint_undef;
};

class prim_config_t;

class params_generator_t {
public:
    params_generator_t() = default;

    params_generator_t(const blocking_params_t &params);

    params_generator_t(int tune_level, int simd_size, blocking_checker_t &chk,
            const std::vector<level_tile_set_t> &level_tile_sets,
            const blocking_params_t &params);

    params_generator_t(int tune_level, int simd_size, blocking_checker_t &chk,
            const std::vector<level_tile_set_t> &level_tile_sets, int idx = -1);

    const std::vector<blocking_params_t> &params_vec() const {
        return params_vec_;
    }

    bool is_empty() const { return params_vec_.empty(); }

    bool is_valid() const { return cur_idx_ < configs(); }

    void move_next() { cur_idx_++; }

    int cur_index() const { return cur_idx_; }

    void set_cur_index(int idx) {
        gpu_assert(idx < configs());
        cur_idx_ = idx;
    }

    const blocking_params_t &cur_params() const { return at(cur_idx_); }

    const blocking_params_t &at(int idx) const {
        gpu_assert(idx >= 0 && idx < configs());
        return params_vec_[idx];
    }

    void set_params(prim_config_t &cfg);

    int configs() const { return (int)params_vec_.size(); }

    template <typename KeyFuncT>
    void sort(int beg, int end, const KeyFuncT &key_func) {
        gpu_assert(beg >= 0 && beg < configs());
        gpu_assert(end >= beg && end <= configs());
        std::sort(params_vec_.begin() + beg, params_vec_.begin() + end,
                [&](const blocking_params_t &a, const blocking_params_t &b) {
                    return key_func(a) < key_func(b);
                });
    }

    template <typename PredicateFuncT>
    void remove_if(const PredicateFuncT &func) {
        gpu_assert(cur_idx_ == -1);
        params_vec_.erase(
                std::remove_if(params_vec_.begin(), params_vec_.end(), func),
                params_vec_.end());
    }

    void shuffle(size_t seed);

    void print_all() const {
        using namespace ir_utils;
        std::vector<std::string> headers = {};
        table_t table("List of configs", headers);
        for (int i = 0; i < configs(); i++) {
            auto &params = params_vec_[i];
            gpu_trace() << "params #" << i << ": " << params;
        }
    }

private:
    static void assign_ids(std::vector<blocking_params_t> &vec);

    static void append_params(std::vector<blocking_params_t> &vec,
            const blocking_params_t &params);

    static void append_params(std::vector<blocking_params_t> &vec,
            const std::vector<level_tile_set_t> &level_tile_sets,
            blocking_checker_t &chk, int tune_level, int simd_size);

    std::vector<blocking_params_t> params_vec_;
    int cur_idx_ = 0;
};

enum class tiler_mode_t {
    undef,
    env_config,
    env_tiler,
    lookup,
    model,
    tune,
    default_mode = lookup
};

std::string to_string(tiler_mode_t mode);

struct tiler_params_t {
    tiler_mode_t mode = tiler_mode_t::default_mode;
    bool do_list = false;
    int tune_iters = 0;
    int env_params_idx = -1;
};

const tiler_params_t &tiler_params();

// Helper class to compute the distance between tiles with sizes.
//
// During initialization the number of blocking dimensions might be
// reduced for simplicity - e.g. for convolutions by converting them
// to BMNK values typical for GEMM - then these dims are converted to
// an 'indexed vector', see below. After that, L1 distances (sums of
// absolute differences) can be computed for any two blocking schemes.
// Simplified example:
//   B1: m8n8k16   -> [1, 0, 0]
//   B2: m16n16k16 -> [2, 1, 0]
//   B3: m1n32k16  -> [0, 2, 0]
// Here, each dimension sorts all possible sizes added to it - in the
// example above m ~ {1, 8, 16}, n ~ {8, 16, 32}, and k ~ {16, 16, 16},
// and then numerical indices are assigned to different values per dim
// (m ~ {1: 0; 8: 1; 16: 2},  n ~ {8: 0; 16: 1; 32: 2},  k ~ {16: 0}).
//   L1(B1, B2) = L1(B2, B1) = 1 + 1 + 0 = 2
//   L1(B1, B3) = L1(B3, B1) = 1 + 2 + 0 = 3
//   L1(B2, B3) = L1(B3, B2) = 2 + 1 + 0 = 3
//   L1(B1, B1) = L1(B2, B2) = L1(B3, B3) = 0
class tile_to_vec_t {
public:
    tile_to_vec_t() = default;
    tile_to_vec_t(const std::vector<std::vector<tile_t>> &tiles,
            const std::vector<int> &ids = {});

    float dist(int id0, int id1) const {
        auto &v0 = vecs_[id0];
        auto &v1 = vecs_[id1];
        float ret = 0;
        // Use L1 distance between coordinates.
        for (int i = 0; i < (int)v0.size(); i++) {
            ret += float(std::abs(v0[i] - v1[i]));
        }
        return ret;
    }

private:
    // assigns indices to the dimensions sizes added, one set of indices per dim
    struct indexed_tile_t {
        struct indexed_dim_t {
            indexed_dim_t() = default;
            indexed_dim_t(const pvar_t &dim) : dim_(dim) {}
            bool is_empty() const { return values_.empty(); }
            const pvar_t &dim() const { return dim_; }

            void add(dim_t value) { values_.emplace(value, dim_idx::invalid); }

            void finalize() {
                dim_idx_t idx = 0;
                add(1);
                for (auto &kv : values_) {
                    kv.second = idx++;
                }
            }

            dim_idx_t to_index(dim_t value) const {
                auto it = values_.find(value);
                gpu_assert(it != values_.end());
                return it->second;
            }

            pvar_t dim_;
            std::map<dim_t, dim_idx_t> values_;
        };

        void add(const pvar_t &d, dim_t value) {
            if (dim_mappers_.count(d) == 0) {
                dim_mappers_[d] = indexed_dim_t(d);
            }
            dim_mappers_[d].add(value);
        }

        void add(const tile_t &t) {
            for (auto &d : t) {
                add(d, t[d]);
            }
        }

        void finalize() {
            for (auto &kv : dim_mappers_)
                if (!kv.second.is_empty()) kv.second.finalize();
        }

        dim_idx_t to_index(const pvar_t &d, dim_t value) const {
            return dim_mappers_.at(d).to_index(value);
        }

        std::vector<dim_idx_t> to_index(const tile_t &t) const {
            std::vector<dim_idx_t> ret;
            for (auto &kv : dim_mappers_) {
                auto &m = kv.second;
                if (m.is_empty()) continue;
                ret.push_back(to_index(m.dim(), t.get(m.dim(), 1)));
            }
            return ret;
        }

        std::unordered_map<pvar_t, indexed_dim_t> dim_mappers_;
    };

    std::vector<std::vector<int>> vecs_;
};

// Helper class to track performance data collected during tuning.
class tune_data_t {
public:
    void add_time(int id, uint64_t nsec) {
        resize(id + 1);
        auto &p = points_[id];
        p.id = id;
        p.nsec = std::min(p.nsec, nsec);
        if (p.repeats == 0) reported_points_++;
        p.repeats++;
        if (nsec < best_point_.nsec) best_point_ = p;
    }

    int best_id() const { return best_point_.id; }
    uint64_t nsec(int id) const { return points_[id].nsec; }
    std::vector<int> best_ids(int n) const {
        auto sorted_points = points_;
        std::sort(sorted_points.begin(), sorted_points.end(),
                [&](const bench_point_t &a, const bench_point_t &b) {
                    return a.nsec < b.nsec;
                });
        std::vector<int> ret;
        for (int i = 0; i < std::min((int)sorted_points.size(), n); i++) {
            auto &p = sorted_points[i];
            if (p.id == -1) break;
            ret.push_back(p.id);
        }
        return ret;
    }
    int reported_points() const { return reported_points_; }

    void resize(int new_size) {
        int size = (int)points_.size();
        if (new_size <= size) return;
        points_.resize(new_size);
        for (int i = size; i < new_size; i++) {
            points_[i].id = i;
        }
    }

private:
    static const uint64_t max_nsec_ = std::numeric_limits<uint64_t>::max();

    struct bench_point_t {
        int id = -1;
        int repeats = 0;
        uint64_t nsec = max_nsec_;

        bool is_ok() const { return nsec != max_nsec_; }
    };

    std::vector<bench_point_t> points_;
    int reported_points_ = 0;
    bench_point_t best_point_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
