/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <tuple>

#include "tests/test_thread.hpp"
#include "tests/test_thread_decl.hpp"

std::ostream &operator<<(std::ostream &os, const thr_ctx_t &ctx) {
    if (ctx.max_concurrency == get_default_thr_ctx().max_concurrency)
        os << "auto:";
    else
        os << ctx.max_concurrency << ":";

    if (ctx.core_type == get_default_thr_ctx().core_type)
        os << "auto:";
    else
        os << ctx.core_type << ":";

    if (ctx.nthr_per_core == get_default_thr_ctx().nthr_per_core)
        os << "auto";
    else
        os << ctx.nthr_per_core;

    return os;
}

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
void *thr_ctx_t::get_interop_obj() const {
    return dnnl::testing::get_threadpool(*this);
}
#else
void *thr_ctx_t::get_interop_obj() const {
    return nullptr;
}
#endif

#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL

#include <mutex>
#include <unordered_map>

#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#if !defined(DNNL_TEST_THREADPOOL_USE_TBB)

#include "src/cpu/platform.hpp"
namespace dnnl {
namespace testing {
namespace {
inline int read_num_threads_from_env() {
    const char *env_num_threads = nullptr;
    const char *env_var_name = "OMP_NUM_THREADS";
#ifdef _WIN32
    // This is only required to avoid using _CRT_SECURE_NO_WARNINGS
    const size_t buf_size = 12;
    char buf[buf_size];
    size_t val_size = GetEnvironmentVariable(env_var_name, buf, buf_size);
    if (val_size > 0 && val_size < buf_size) env_num_threads = buf;
#else // ifdef _WIN32
    env_num_threads = ::getenv(env_var_name);
#endif

    int num_threads = 0;
    if (env_num_threads) {
        char *endp;
        int nt = strtol(env_num_threads, &endp, 10);
        if (*endp == '\0') num_threads = nt;
    }
    if (num_threads <= 0) {
        num_threads = (int)dnnl::impl::cpu::platform::get_max_threads_to_use();
    }
    return num_threads;
}
} // namespace
} // namespace testing
} // namespace dnnl
#endif // !defined(DNNL_TEST_THREADPOOL_USE_TBB)

#if defined(DNNL_TEST_THREADPOOL_USE_EIGEN)

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

#include "absl/synchronization/blocking_counter.h"

#include "common/compiler_workarounds.hpp"

#include <memory>

namespace dnnl {
namespace testing {

class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
private:
    std::unique_ptr<Eigen::ThreadPool> tp_;

    static void balance211(int n, int team, int tid, int *n_start, int *n_end) {
        if (team <= 1 || n == 0) {
            *n_start = 0;
            *n_end = n;
            return;
        }
        int min_per_team = n / team;
        int remainder = n - min_per_team * team; // i.e., n % teams.
        *n_start = tid * min_per_team + std::min(tid, remainder);
        *n_end = *n_start + min_per_team + (tid < remainder);
    }

    static void run_jobs(bool balance, int i, int n, int njobs,
            const std::function<void(int, int)> &fn) {
        if (balance) {
            int start, end;
            balance211(n, njobs, i, &start, &end);
            for (int j = start; j < end; j++)
                fn(j, n);
        } else {
            fn(i, n);
        }
    }

public:
    explicit threadpool_t(int num_threads = 0) {
        if (num_threads <= 0) num_threads = read_num_threads_from_env();
        tp_.reset(new Eigen::ThreadPool(num_threads));
    }
    int get_num_threads() const override { return tp_->NumThreads(); }
    bool get_in_parallel() const override {
        return tp_->CurrentThreadId() != -1;
    }
    uint64_t get_flags() const override { return 0; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        // Should never happen.
        if (n == 0) { return; }

        // Should never happen.
        if (n == 1) {
            fn(0, 1);
            return;
        }

        int nthr = get_num_threads();
        int njobs = std::min(n, nthr);
        bool balance = (nthr < n);

        absl::BlockingCounter counter(njobs);
        std::function<void(int, int)> handle_range
                = [= COMPAT_THIS_CAPTURE, &handle_range, &counter](
                          int first, int last) {
            while (last - first > 1) {
                const auto mid = first + (last - first) / 2;
                // Find something near the midpoint which is a
                // multiple of block size.
                tp_->ScheduleWithHint(
                        [=]() { handle_range(mid, last); }, mid, mid + 1);
                last = mid;
            }
            run_jobs(balance, first, n, njobs, fn);
            counter.DecrementCount();
        };

        // Eigen avoids a thread hop by running the root of the tree on the main
        // thread. We have disabled this because it actually slows things down
        // relative to base because base cheats and uses n threads while letting
        // main continue doing other work
        tp_->ScheduleWithHint([=]() { handle_range(0, njobs); }, 0, 1);

        counter.Wait();
    }

    void wait() override {
        // Nothing to do, runtime is synchronous
    }
};

} // namespace testing
} // namespace dnnl

#elif defined(DNNL_TEST_THREADPOOL_USE_EIGEN_ASYNC)

// absl sources define its own version of `CHECK` macro. oneDNN's version is not
// needed further the file, thus, disable it for compilation reason.
#undef CHECK

#define EIGEN_USE_THREADS
#include "Eigen/ThreadPool"

#include "xla/backends/cpu/runtime/work_queue.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace dnnl {
namespace testing {

static tsl::AsyncValueRef<tsl::Chain> OkDoneEventSingleton() {
    static std::unique_ptr<tsl::AsyncValueOwningRef<tsl::Chain>> singleton
            = [] {
        static auto storage = std::make_unique<
                tsl::internal::AsyncValueStorage<tsl::Chain>>();
        return std::make_unique<tsl::AsyncValueOwningRef<tsl::Chain>>(
                tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
    }();
    return singleton->AsRef();
}

class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
private:
    // Original `OneDnnThreadPool` at
    // `xla/backends/cpu/runtime/onednn/onednn_threadpool.h` takes
    // `Eigen::ThreadPoolInterface` instead. Since `Eigen::ThreadPool` is
    // a parent class, which is an alias to `NonBlockingThreadPool`, it fits
    // the need.
    std::unique_ptr<Eigen::ThreadPool> thread_pool_;

    // Async value that signals completion of the last scheduled parallel loop.
    //
    // Note: for an unknown reason if the `threadpool_t` is constructed but the
    // `parallel_for(...)` isn't called, ASan would bark on
    // `OkDoneEventSingleton()` internals during the program destruction.
    // It seems `done_event_` must be updated with its `FlatMap(...)` method,
    // but it's just a theory.
    // Some tests exposing this scenario have dummy parallel calls to avoid this
    // issue.
    // ANCHOR: DUMMY_PARALLEL.
    tsl::AsyncValueRef<tsl::Chain> done_event_;

public:
    explicit threadpool_t(int num_threads = 0) {
        if (num_threads <= 0) num_threads = read_num_threads_from_env();
        thread_pool_.reset(new Eigen::ThreadPool(num_threads));
        done_event_ = OkDoneEventSingleton();
    }
    int get_num_threads() const override { return thread_pool_->NumThreads(); }
    bool get_in_parallel() const override { return false; }
    uint64_t get_flags() const override { return ASYNCHRONOUS; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        // If we are using oneDNN with async support, we need to schedule the
        // parallel loop using the done_event_. This allows us to return
        // immediately and not block the caller thread.
        auto parallelize = [this, n, fn](tsl::Chain) {
            return xla::cpu::Worker::Parallelize(thread_pool_.get(),
                    thread_pool_->NumThreads(), n,
                    [fn, n](size_t i) { fn(static_cast<int>(i), n); });
        };

        done_event_ = done_event_.FlatMap(parallelize);
    }
    void wait() override {
        // While performing asynchronous execution, wait() method is needed to
        // notify the user that the output is ready. oneDNN will not call wait()
        // inside the library to avoid deadlock.
        tsl::BlockUntilReady(done_event_);
    }

    tsl::AsyncValueRef<tsl::Chain> done_event() const { return done_event_; }
};

} // namespace testing
} // namespace dnnl

#elif defined(DNNL_TEST_THREADPOOL_USE_TBB)
#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

namespace dnnl {
namespace testing {

class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
public:
    explicit threadpool_t(int num_threads) { (void)num_threads; }
    int get_num_threads() const override {
        return tbb::this_task_arena::max_concurrency();
    }
    bool get_in_parallel() const override { return 0; }
    uint64_t get_flags() const override { return 0; }
    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        tbb::parallel_for(
                0, n, [&](int i) { fn(i, n); }, tbb::static_partitioner());
    }
    void wait() override {}
};

} // namespace testing
} // namespace dnnl

#else

#include "src/common/counting_barrier.hpp"

#include <atomic>
#include <thread>
#include <vector>
#include <condition_variable>

namespace dnnl {
namespace testing {

// Naiive synchronous threadpool:
// - Only a single parallel_for is executed at the same time.
// - Recursive parallel_for results in sequential execution.
class threadpool_t : public dnnl::threadpool_interop::threadpool_iface {
public:
    using task_func = std::function<void(int, int)>;

    explicit threadpool_t(int num_threads = 0) {
        if (num_threads <= 0) num_threads = read_num_threads_from_env();
        num_threads_ = num_threads;
        master_sense_ = 0;

        for (int i = 0; i < 2; i++) {
            tasks_[i].go_flag.store(0);
            tasks_[i].fn = nullptr;
            tasks_[i].n = 0;
        }

        barrier_init();
        workers_.reset(new std::vector<worker_data_t>(num_threads_));
        for (int i = 0; i < num_threads_; i++) {
            auto wd = &workers_->at(i);
            wd->thread_id = i;
            wd->tp = this;
            wd->thread.reset(new std::thread(worker_loop, &workers_->at(i)));
        }
        barrier_wait();
    }

    ~threadpool_t() override {
        std::unique_lock<std::mutex> l(master_mutex_);
        barrier_init();
        task_submit(nullptr, 0);
        for (int i = 0; i < num_threads_; i++)
            workers_->at(i).thread->join();
        barrier_wait();
    }

    int get_num_threads() const override { return num_threads_; }

    bool get_in_parallel() const override { return worker_self() != nullptr; }

    uint64_t get_flags() const override { return 0; }

    void parallel_for(int n, const task_func &fn) override {
        if (worker_self() != nullptr)
            task_execute(0, 1, &fn, n);
        else {
            std::unique_lock<std::mutex> l(master_mutex_);
            barrier_init();
            task_submit(&fn, n);
            barrier_wait();
        }
    }

    void wait() override {}

private:
    int num_threads_;
    std::mutex master_mutex_;
    std::mutex master_submit_mutex_;

    struct worker_data_t {
        int thread_id;
        threadpool_t *tp;
        std::condition_variable cv;
        std::unique_ptr<std::thread> thread;
    };
    std::unique_ptr<std::vector<worker_data_t>> workers_;
    static thread_local worker_data_t *worker_self_;
    worker_data_t *worker_self() const {
        return worker_self_ != nullptr && worker_self_->tp == this
                ? worker_self_
                : nullptr;
    }

    struct task_data_t {
        std::atomic<int> go_flag;
        const task_func *fn;
        int n;
    };
    int master_sense_;
    task_data_t tasks_[2];

    dnnl::impl::counting_barrier_t barrier_;

    void barrier_init() { barrier_.init(num_threads_); }

    void barrier_wait() {
        barrier_.wait();
        tasks_[master_sense_].go_flag.store(0);
        master_sense_ = !master_sense_;
    }

    void barrier_notify(int worker_sense) { barrier_.notify(); }

    void task_submit(const task_func *fn, int n) {
        std::lock_guard<std::mutex> l(master_submit_mutex_);
        tasks_[master_sense_].fn = fn;
        tasks_[master_sense_].n = n;
        tasks_[master_sense_].go_flag.store(1);
        for (int i = 0; i < num_threads_; i++) {
            workers_->at(i).cv.notify_one();
        }
    }

    void task_execute(int ithr, int nthr, const task_func *fn, int n) {
        if (fn != nullptr && n > 0) {
            int start, end;
            impl::balance211(n, nthr, ithr, start, end);
            for (int i = start; i < end; i++)
                (*fn)(i, n);
        }
    }

    static void worker_loop(worker_data_t *wd) {
        worker_self_ = wd;
        int worker_sense = 0;

        wd->tp->barrier_notify(worker_sense);

        bool time_to_exit = false;
        std::unique_lock<std::mutex> l(wd->tp->master_submit_mutex_);

        do {
            worker_sense = !worker_sense;
            auto *t = &wd->tp->tasks_[worker_sense];
            wd->tp->workers_->at(wd->thread_id).cv.wait(l, [t]() {
                return t->go_flag.load() != 0;
            });
            wd->tp->task_execute(
                    wd->thread_id, wd->tp->num_threads_, t->fn, t->n);
            time_to_exit = t->fn == nullptr;
            wd->tp->barrier_notify(worker_sense);
        } while (!time_to_exit);
    }
};

thread_local threadpool_t::worker_data_t *threadpool_t::worker_self_ = nullptr;

} // namespace testing
} // namespace dnnl
#endif

namespace dnnl {

// Original threadpool utils are used by the scoped_tp_activation_t and thus
// need to be re-declared because of the hack above.
namespace impl {
namespace threadpool_utils {
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp);
void deactivate_threadpool();
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool();
} // namespace threadpool_utils
} // namespace impl

namespace testing {
// Threadpool singleton
dnnl::threadpool_interop::threadpool_iface *get_threadpool(
        const thr_ctx_t &ctx) {
    // global default threadpool is returned when thr context is
    // default
    static std::unordered_map<int, dnnl::testing::threadpool_t> tp_map;
    auto ret_val = tp_map.find(ctx.max_concurrency);
    if (ret_val != tp_map.end()) return &(ret_val->second);
    auto res = tp_map.emplace(std::piecewise_construct,
            std::forward_as_tuple(ctx.max_concurrency),
            std::forward_as_tuple(ctx.max_concurrency));
    if (!res.second) {
        fprintf(stderr, "get_threadpool failed to create a threadpool\n");
        exit(1);
    }
    return &(res.first->second);
}

scoped_tp_activation_t::scoped_tp_activation_t(
        dnnl::threadpool_interop::threadpool_iface *tp) {
    impl::threadpool_utils::activate_threadpool(tp);
}

scoped_tp_activation_t::~scoped_tp_activation_t() {
    impl::threadpool_utils::deactivate_threadpool();
}

scoped_tp_deactivation_t::scoped_tp_deactivation_t() {
    impl::threadpool_utils::deactivate_threadpool();
}

scoped_tp_deactivation_t::~scoped_tp_deactivation_t() {
    // we always use the same threadpool that is returned by `get_threadpool()`
    impl::threadpool_utils::activate_threadpool(get_threadpool());
}

} // namespace testing

// Implement a dummy threadpools_utils protocol here so that it is picked up
// by parallel*() calls from the tests.
namespace impl {
namespace testing_threadpool_utils {
void activate_threadpool(dnnl::threadpool_interop::threadpool_iface *tp) {}
void deactivate_threadpool() {}
dnnl::threadpool_interop::threadpool_iface *get_active_threadpool() {
    return testing::get_threadpool();
}

// here we return 0 so that parallel* calls use the
// default number of threads in the threadpool.
int get_max_concurrency() {
    return 0;
}

} // namespace testing_threadpool_utils

} // namespace impl
} // namespace dnnl

#endif

#ifdef TBB_INTERFACE_VERSION
// tbb constraints on core type appear in 2021.2
// tbb constraints on max_concurrency appear in 2020
// we check only for 2021.2 to enable thread context knobs
#define DNNL_TBB_CONSTRAINTS_ENABLED (TBB_INTERFACE_VERSION >= 12020)
// API to do explicit finalization was introduced in 2021.6.
#define DNNL_TBB_NEED_EXPLICIT_FINALIZE (TBB_INTERFACE_VERSION >= 12060)
#else
#define DNNL_TBB_CONSTRAINTS_ENABLED 0
#define DNNL_TBB_NEED_EXPLICIT_FINALIZE 0
#endif

#define DNNL_TBB_THREADING_WITH_CONSTRAINTS \
    (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB) \
            && DNNL_TBB_CONSTRAINTS_ENABLED
#define DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS \
    (DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_TBB) \
            && !DNNL_TBB_CONSTRAINTS_ENABLED

#if DNNL_TBB_THREADING_WITH_CONSTRAINTS
#include "oneapi/tbb/info.h"
#endif

void finalize_tbb() {
#if DNNL_TBB_NEED_EXPLICIT_FINALIZE
    oneapi::tbb::task_scheduler_handle handle
            = oneapi::tbb::task_scheduler_handle {oneapi::tbb::attach {}};
    oneapi::tbb::finalize(handle, std::nothrow);
#endif
}

const thr_ctx_t &get_default_thr_ctx() {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ \
        || DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL \
        || DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS
    static const thr_ctx_t default_thr_ctx = {0, -1, 0};
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    static const thr_ctx_t default_thr_ctx = {omp_get_max_threads(), -1, 0};
#elif DNNL_TBB_THREADING_WITH_CONSTRAINTS
    static const thr_ctx_t default_thr_ctx = {tbb::task_arena::automatic,
            tbb::task_arena::automatic, tbb::task_arena::automatic};
#endif
    return default_thr_ctx;
}

#define THR_CTX_ASSERT(check, msg_fmt, ...) \
    do { \
        if (!(check)) { \
            fprintf(stderr, msg_fmt, __VA_ARGS__); \
            exit(1); \
        } \
    } while (0)

// Single version of each function; the runtime-specific logic is handled inside
// via preprocessor branching.
int create_in_thr_ctx(const thr_ctx_t &ctx, const std::function<int()> &f) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ \
        || DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS
    THR_CTX_ASSERT(ctx.core_type == get_default_thr_ctx().core_type
                    && ctx.max_concurrency
                            == get_default_thr_ctx().max_concurrency
                    && ctx.nthr_per_core == get_default_thr_ctx().nthr_per_core,
            "Threading knobs not supported for this runtime: %s\n",
            DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
                    ? "sequential runtime has no threading"
                    : "TBB version is too old (>=2021.2 required)");
    return f();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    THR_CTX_ASSERT(ctx.core_type == get_default_thr_ctx().core_type,
            "core type %d is not supported for OMP runtime\n", ctx.core_type);

    auto max_nthr = omp_get_max_threads();
    omp_set_num_threads(ctx.max_concurrency);
    auto st = f();
    omp_set_num_threads(max_nthr);
    return st;
#elif DNNL_TBB_THREADING_WITH_CONSTRAINTS
    static auto core_types
            = tbb::info::core_types(); // sorted by the relative strength

    if ((ctx.core_type != get_default_thr_ctx().core_type)
            && ((size_t)ctx.core_type >= core_types.size()))
        printf("WARNING: TBB smallest core has index %d. Using this "
               "instead of %d.\n",
                (int)core_types.size() - 1, ctx.core_type);
    size_t core_type_id = (size_t)ctx.core_type < core_types.size()
            ? ctx.core_type
            : core_types.size() - 1;
    auto core_type = ctx.core_type == tbb::task_arena::automatic
            ? tbb::task_arena::automatic
            : core_types[core_type_id];
    auto arena = tbb::task_arena {
            tbb::task_arena::constraints {}
                    .set_core_type(core_type)
                    .set_max_threads_per_core(ctx.nthr_per_core)
                    .set_max_concurrency(ctx.max_concurrency)};
    return arena.execute([&] { return f(); });
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    THR_CTX_ASSERT(ctx.core_type == get_default_thr_ctx().core_type,
            "core type %d is not supported for TP runtime\n", ctx.core_type);

    auto tp = dnnl::testing::get_threadpool(ctx);
    auto stp = dnnl::testing::scoped_tp_activation_t(tp);
    return f();
#else
#error "unsupported threading runtime!"
#endif
}

int execute_in_thr_ctx(const thr_ctx_t &ctx, const std::function<int()> &f) {
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ \
        || DNNL_TBB_THREADING_WITHOUT_CONSTRAINTS
    THR_CTX_ASSERT(ctx.core_type == get_default_thr_ctx().core_type
                    && ctx.max_concurrency
                            == get_default_thr_ctx().max_concurrency
                    && ctx.nthr_per_core == get_default_thr_ctx().nthr_per_core,
            "Threading knobs not supported for this runtime: %s\n",
            DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_SEQ
                    ? "sequential runtime has no threading"
                    : "TBB version is too old (>=2021.2 required)");
    return f();
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_OMP
    THR_CTX_ASSERT(ctx.core_type == get_default_thr_ctx().core_type,
            "core type %d is not supported for OMP runtime\n", ctx.core_type);

    auto max_nthr = omp_get_max_threads();
    omp_set_num_threads(ctx.max_concurrency);
    auto st = f();
    omp_set_num_threads(max_nthr);
    return st;
#elif DNNL_TBB_THREADING_WITH_CONSTRAINTS
    static auto core_types
            = tbb::info::core_types(); // sorted by the relative strength

    if ((ctx.core_type != get_default_thr_ctx().core_type)
            && ((size_t)ctx.core_type >= core_types.size()))
        printf("WARNING: TBB smallest core has index %d. Using this "
               "instead of %d.\n",
                (int)core_types.size() - 1, ctx.core_type);
    size_t core_type_id = (size_t)ctx.core_type < core_types.size()
            ? ctx.core_type
            : core_types.size() - 1;
    auto core_type = ctx.core_type == tbb::task_arena::automatic
            ? tbb::task_arena::automatic
            : core_types[core_type_id];
    auto arena = tbb::task_arena {
            tbb::task_arena::constraints {}
                    .set_core_type(core_type)
                    .set_max_threads_per_core(ctx.nthr_per_core)
                    .set_max_concurrency(ctx.max_concurrency)};
    return arena.execute([&] { return f(); });
#elif DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
    // The function f shall take an interop obj as last argument.
    THR_CTX_ASSERT(ctx.core_type == get_default_thr_ctx().core_type,
            "core type %d is not supported for TP runtime\n", ctx.core_type);
    return f();
#else
#error "unsupported threading runtime!"
#endif
}
