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
#include <iostream>
#include <map>
#include <string>
#include <unistd.h>
#include <sys/ioctl.h>

summary_t summary {};

size_t get_terminal_width() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) { return 80; }
    return static_cast<size_t>(w.ws_col);
}

void print_in_frame(const std::string &str, size_t terminal_width) {
    size_t content_width = terminal_width - 2;
    size_t pos = 0;

    while (pos < str.size()) {
        std::string part = str.substr(pos, content_width);
        if (part.size() < content_width) {
            part += std::string(content_width - part.size(), ' ');
        }
        std::cout << "|" << part << "|" << std::endl;
        pos += content_width;
    }
}

// Prints the statistics summary over implementations used in the run in a
// table.
void print_impl_names_summary() {
    if (!summary.impl_names) return;

    // If there is no content in the table, just exit.
    if (benchdnn_stat.impl_names.empty()) return;

    auto term_width = get_terminal_width();
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
    std::string footer(term_width, '=');
    std::string footer_text_pad(
            std::max(longest_entry_length, footer_size) - footer_size, ' ');
            
    std::cout << footer << std::endl
              << footer_text + footer_text_pad << std::endl
              << footer << std::endl;

    // Print the table content.
    for (const auto &impl_entry : swapped_map) {
        size_t percent = static_cast<size_t>(
                std::round(100.f * impl_entry.first / total_cases));

        std::string right_part = std::to_string(impl_entry.first) + " ("
                + std::to_string(percent) + "%)";
        size_t left_padding = (longest_impl_length > impl_entry.second.size())
                ? longest_impl_length - impl_entry.second.size()
                : 0;
        std::string left_padded
                = std::string(left_padding, ' ') + impl_entry.second;

        std::string line = left_padded + ":" + right_part;

        if (line.size() <= term_width - 2) {
            std::cout << "|" << line
                      << std::string(term_width - 2 - line.size(), ' ') << "|"
                      << std::endl;
        } else {
            std::string simple_line = impl_entry.second + ":" + right_part;
            print_in_frame(simple_line, term_width);
        }
    }
    std::cout << footer << std::endl;
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
