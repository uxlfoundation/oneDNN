/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef UTILS_STRINGSTREAM_HPP
#define UTILS_STRINGSTREAM_HPP

// This is a copy-paste from the library type.
// It's copied to untie from the library deendencies. Can't include it directly
// due to the responsible headers brings a lot of undesired code.
//
// TODO: consolidate when the library isolate this class not to have any extra
// dependencies.

#include <locale>
#include <sstream>

struct stringstream_t : public std::stringstream {
    template <typename... Args>
    stringstream_t(Args &&...args)
        : std::stringstream(std::forward<Args>(args)...) {
        this->imbue(std::locale::classic());
    }

    stringstream_t(const stringstream_t &) = delete;
    stringstream_t &operator=(const stringstream_t &) = delete;

    stringstream_t(stringstream_t &&) = delete;
    stringstream_t &operator=(stringstream_t &&) = delete;

private:
    using std::stringstream::imbue;
};

struct istringstream_t : public std::istringstream {
    template <typename... Args>
    istringstream_t(Args &&...args)
        : std::istringstream(std::forward<Args>(args)...) {
        this->imbue(std::locale::classic());
    }

    istringstream_t(const istringstream_t &) = delete;
    istringstream_t &operator=(const istringstream_t &) = delete;

    istringstream_t(istringstream_t &&) = delete;
    istringstream_t &operator=(istringstream_t &&) = delete;

private:
    using std::istringstream::imbue;
};

struct ostringstream_t : public std::ostringstream {
    template <typename... Args>
    ostringstream_t(Args &&...args)
        : std::ostringstream(std::forward<Args>(args)...) {
        this->imbue(std::locale::classic());
    }

    ostringstream_t(const ostringstream_t &) = delete;
    ostringstream_t &operator=(const ostringstream_t &) = delete;

    ostringstream_t(ostringstream_t &&) = delete;
    ostringstream_t &operator=(ostringstream_t &&) = delete;

private:
    using std::ostringstream::imbue;
};

#endif
