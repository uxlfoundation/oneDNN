:orphan:

.. index:: pair: page; Using Threadpool-based Threading
.. _doxid-dev_guide_threadpool:

Using Threadpool-based Threading
================================

When oneDNN is built with the threadpool CPU runtime (see :ref:`Build Options <doxid-dev_guide_build_options>`), oneDNN requires the user to implement a threadpool interface to enable the library to perform computations using multiple threads.

The threadpool interface is defined in ``include/oneapi/dnnl/dnnl_threadpool_iface.hpp``. Below is a sample implementation based on the Eigen threadpool that is also used for testing (see ``tests/test_thread.cpp``).

.. ref-code-block:: cpp

	#include "dnnl_threadpool_iface.hpp"
	
	class threadpool_t : public :ref:`dnnl::threadpool_interop::threadpool_iface <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface>` {
	private:
	    // Change to Eigen::NonBlockingThreadPool if using Eigen <= 3.3.7
	    std::unique_ptr<Eigen::ThreadPool> tp_;
	
	public:
	    explicit threadpool_t(int num_threads = 0) {
	        if (num_threads <= 0)
	            num_threads = (int)std::thread::hardware_concurrency();
	        tp_.reset(new Eigen::ThreadPool(num_threads));
	    }
	    int :ref:`get_num_threads <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a1071371237ec5c98db140c1f1f1c0114>`() const override { return tp_->NumThreads(); }
	    bool :ref:`get_in_parallel <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a8279221c6e2f903a4c811688f7a033be>`() const override {
	        return tp_->CurrentThreadId() != -1;
	    }
	    uint64_t :ref:`get_flags <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a868267178f259cee5f1d5b33a8781a3e>`() const override { return :ref:`ASYNCHRONOUS <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a9e6d861d659445fe5abcf302e464d9e5>`; }
	    void :ref:`parallel_for <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1ac3d85ff935c11ec038ecabeeabd03ffb>`(int n, const std::function<void(int, int)> &fn) override {
	        int nthr = :ref:`get_num_threads <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__iface_1a1071371237ec5c98db140c1f1f1c0114>`();
	        int njobs = std::min(n, nthr);
	
	        for (int i = 0; i < njobs; i++) {
	            tp_->Schedule([i, n, njobs, fn]() {
	                int start, end;
	                impl::balance211(n, njobs, i, start, end);
	                for (int j = start; j < end; j++)
	                    fn(j, n);
	            });
	        }
	    };
	};

oneDNN supports non-blocked verbose profiling of primitive executions for threadpool runtimes provided the user define a threadpool event and event interface for the profiling mechanism to poll for event completion. An example of this event interfasce is also provided in ``include/oneapi/dnnl/dnnl_threadpool_iface.hpp`` and the following sample implementation from ``test/test_thread.cpp`` can be used for testing.

.. ref-code-block:: cpp

	// Lightweight completion event exposed to oneDNN's verbose profiler.
	// the start time for execution is stamped by the first parallel_for()
	// worker via a compare-and-swap atomic operation to avoid races
	// between workers. The end time and completion flag are set by the
	// AndThen callback registered in get_event() when the chain resolves.
	class threadpool_event_t
	    : public :ref:`dnnl::threadpool_interop::threadpool_event_iface_t <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t>` {
	public:
	    threadpool_event_t() = default;
	
	    bool :ref:`is_complete <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1acd8ff64ed52794a432a7eb7b3ae7acb8>`() const override {
	        return complete_.load(std::memory_order_acquire);
	    }
	
	    void :ref:`wait <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1a2edbf1e530f1747bf4be58728f591635>`() const override {
	        while (!:ref:`is_complete <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1acd8ff64ed52794a432a7eb7b3ae7acb8>`()) {}
	    }
	
	    double :ref:`exec_time_ms <doxid-structdnnl_1_1threadpool__interop_1_1threadpool__event__iface__t_1a1e411516ce3e1fb339c4507ce7d635bd>`() const override {
	        int64_t start = start_ns_.load(std::memory_order_relaxed);
	        int64_t end = end_ns_.load(std::memory_order_relaxed);
	        return static_cast<double>(end - start) * 1e-6;
	    }
	
	    void stamp_start(int64_t ns) {
	        int64_t expected = 0;
	        start_ns_.compare_exchange_strong(
	                expected, ns, std::memory_order_relaxed);
	    }
	
	    // Records the end timestamp and sets complete_.
	    // end_ns_ is written before complete_ to ensure exec_time_ms() is
	    // valid when is_complete() returns true.
	    void mark_complete() {
	        end_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                              std::chrono::high_resolution_clock::now()
	                                      .time_since_epoch())
	                              .count(),
	                std::memory_order_relaxed);
	        complete_.store(true, std::memory_order_release);
	    }
	
	private:
	    // Completion flag set by mark_complete() in the AndThen callback.
	    std::atomic<bool> complete_ {false};
	
	    // Start timestamp in nanoseconds. Stamped by the first worker via a
	    // a compare-and-swap atomic operation.
	    std::atomic<int64_t> start_ns_ {0};
	
	    // End timestamp in nanoseconds. Written by mark_complete() before
	    // complete_ is set.
	    std::atomic<int64_t> end_ns_ {0};
	};

