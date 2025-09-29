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

#include "utils/summary.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>

summary_t summary {};

// Prints the statistics summary over implementations used in the run in a
// table.
void print_impl_names_summary() {
    if (!summary.impl_names) return;

    // If there is no content in the table, just exit.
    if (benchdnn_stat.impl_names.empty()) return;

    std::string footer_text(
            "= Implementation statistics (--summary=no-impl to disable) ");
    // +1 for closing `=`.
    const size_t footer_size = footer_text.size() + 1;

    const auto swap_pair = [](const std::pair<std::string, size_t> &p) {
        return std::pair<size_t, std::string>(p.second, p.first);
    };

    const auto swap_map = [swap_pair](const std::map<std::string, size_t> &m) {
        std::multimap<size_t, std::string, std::greater<size_t>> sm;
        std::transform(
                m.begin(), m.end(), std::inserter(sm, sm.begin()), swap_pair);
        return sm;
    };

    // Reverse original map's key-value pairs to sort by the number of hits.
    std::multimap<size_t, std::string, std::greater<size_t>> swapped_map
            = swap_map(benchdnn_stat.impl_names);

    // Collect the biggest sizes across entries to properly pad for a nice view.
    size_t longest_impl_length = 0;
    size_t longest_count_length = 0;
    size_t total_cases = 0;
    for (const auto &impl_entry : swapped_map) {
        longest_impl_length
                = std::max(impl_entry.second.size(), longest_impl_length);
        longest_count_length = std::max(
                std::to_string(impl_entry.first).size(), longest_count_length);
        total_cases += impl_entry.first;
    }

    // `extra_symbols` covers final string chars not covered by other variables,
    // e.g., between entry's key and value, and entry borders `| ` and ` |`
    constexpr size_t extra_symbols = 7;
    // The largest percent format is ` (xxx%)`.
    constexpr size_t largest_percent_length = 7;
    // Must match `entry_length` from the loop below.
    size_t longest_entry_length = std::max(footer_size,
            longest_impl_length + longest_count_length + largest_percent_length
                    + extra_symbols);

    // Print the footer. Adjusted if content strings are larger.
    std::string footer(longest_entry_length, '=');
    std::string footer_text_pad(
            std::max(longest_entry_length, footer_size) - footer_size, ' ');
    printf("%s\n", footer.c_str());
    printf("%s%s=\n", footer_text.c_str(), footer_text_pad.c_str());
    printf("%s\n", footer.c_str());

    // Print the table content.
    // TODO: short terminal windows have the output skewed. There's a way to get
    // the window width:
    // ```
    //     struct winsize w;
    //     if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) { return 80; }
    //     return static_cast<size_t>(w.ws_col);
    // ```
    // This size brings a hard limit on a printed line and the idea is to
    // shorten the impl name string and end it with `~` or `...` instead of
    // reporting the full name to fit a single line.
    for (const auto &impl_entry : swapped_map) {
        size_t percent = static_cast<size_t>(
                std::round(100.f * impl_entry.first / total_cases));
        std::string percent_str = " (" + std::to_string(percent) + "%)";

        // Pad to the left of impl based on the longest impl name.
        size_t left_pad_length = longest_impl_length - impl_entry.second.size();
        std::string left_pad(left_pad_length, ' ');

        // Get the entry length to properly pad it from left and right.
        // Must be computed with same members as `longest_entry_length`.
        size_t entry_length = /* impl_name = */ impl_entry.second.size() +
                /* count_length = */ std::to_string(impl_entry.first).size() +
                /* percent_length = */ percent_str.size() +
                /* extra_symbols = */ extra_symbols;

        // Pad to the right of numbers based on largest count.
        size_t right_pad_length
                = longest_entry_length - entry_length - left_pad_length;
        std::string right_pad(right_pad_length, ' ');

        printf("| %s%s : %zu%s%s |\n", left_pad.c_str(),
                impl_entry.second.c_str(), impl_entry.first,
                percent_str.c_str(), right_pad.c_str());
    }
    printf("%s\n", footer.c_str());
}

// Prints the statistics summary over implementations used in the run in CSV
// format.
void print_impl_names_csv_summary() {
    if (!summary.impl_names_csv) return;

    // If there is no content in the table, just exit.
    if (benchdnn_stat.impl_names.empty()) return;

    const auto swap_pair = [](const std::pair<std::string, size_t> &p) {
        return std::pair<size_t, std::string>(p.second, p.first);
    };

    const auto swap_map = [swap_pair](const std::map<std::string, size_t> &m) {
        std::multimap<size_t, std::string, std::greater<size_t>> sm;
        std::transform(
                m.begin(), m.end(), std::inserter(sm, sm.begin()), swap_pair);
        return sm;
    };

    // Reverse original map's key-value pairs to sort by the number of hits.
    std::multimap<size_t, std::string, std::greater<size_t>> swapped_map
            = swap_map(benchdnn_stat.impl_names);

    // Print the string content.
    printf("benchdnn_summary,impl_names");
    for (const auto &impl_entry : swapped_map) {
        printf(",%s:%zu", impl_entry.second.c_str(), impl_entry.first);
    }
    printf("\n");
}
