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

#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

status_t preprocess_headers(stringstream_t &pp_code, const char *code,
        const compute::kernel_ctx_t &kernel_ctx) {
    stringstream_t code_stream(code);

    for (std::string line; std::getline(code_stream, line);) {
        const size_t include_pos = line.find("#include");
        if (include_pos != std::string::npos) {
            static constexpr size_t include_len = 8;
            const size_t first_quote_pos
                    = line.find("\"", include_pos + include_len);
            const size_t second_quote_pos
                    = line.find("\"", first_quote_pos + 1);
            const size_t kernel_name_len
                    = second_quote_pos - first_quote_pos - 1;
            const auto header_name
                    = line.substr(first_quote_pos + 1, kernel_name_len);
            const char *header_source
                    = kernel_ctx.get_custom_header(header_name);
            if (!header_source) header_source = get_kernel_header(header_name);
            CHECK(preprocess_headers(pp_code, header_source, kernel_ctx));
        } else {
            pp_code << line << std::endl;
        }
    }
    return status::success;
}

void debugdump_processed_source(const std::string &source,
        const std::string &options, const std::string &cl_options) {
#if defined(__linux__) && defined(DNNL_DEV_MODE)
    if (get_verbose(verbose_t::debuginfo) >= 10) {
        auto get_defines = [](const std::string &from) {
            std::string ret;
            size_t pos = 0;
            while (pos < from.length()) {
                // Find next define argument
                pos = from.find("-D", pos);

                // Generate argument, quotes are interpreted literally, but
                // other special shell characters need escaped. Does not
                // currently handle quotes with the ' character or nested quotes
                char quote_parity = true;
                while (pos < from.length()) {
                    if (quote_parity
                            && utils::one_of(from[pos], '~', '#', '$', '&', '*',
                                    '(', ')', '\\', '|', '[', ']', '{', '}',
                                    ';', '\'', '<', '>', '/', '?', '!')) {
                        ret += '\\';
                    }
                    ret += from[pos];
                    if (from[pos] == '"') quote_parity ^= true;
                    if (from[pos] == ' ' && quote_parity) break;

                    pos++;
                }
            }
            return ret;
        };
        auto execute_command = [](const std::string &cmd,
                                       const std::string &stdin) {
            std::string result;
            std::array<char, 256> buffer;
            FILE *pipe = popen(cmd.c_str(), "w");
            fputs(stdin.c_str(), pipe);
            if (pipe) {
                while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                    result += buffer.data();
                }
            }
            pclose(pipe);
            return result;
        };

        // Run utilities to evaluate preprocessor defines and format the file
        // Theoretically, we can accomplish this task with libclang, but it
        // seems more work than it is worth. Instead, wrapping this in OCL_DEBUG
        // so that calls to the system are not included in the default build.

        // Due to the use of a different C preprocessor, warnings should not be
        // ignored, as they may correspond to a different behavior in the OpenCL
        // C preprocessor
        auto o = get_defines(options) + get_defines(cl_options);
        std::string preprocess_cmd
                = std::string() + "cpp -P " + o + " | clang-format";
        execute_command(preprocess_cmd, source);
        std::cout << "OCL_ARCH_OPTIONS: " << cl_options << std::endl;
    }
#endif
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
