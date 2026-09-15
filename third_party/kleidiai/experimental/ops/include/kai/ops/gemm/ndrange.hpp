//
// SPDX-FileCopyrightText: Copyright 2019-2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <initializer_list>

namespace kai {
namespace ops {
template <unsigned int D>
class NDRange
{
private:
    std::array<unsigned int, D> m_sizes{};
    std::array<unsigned int, D> m_totalsizes{};

    class NDRangeIterator
    {
    private:
        const NDRange &m_parent;
        unsigned int   m_pos = 0;
        unsigned int   m_end = 0;

    public:
        NDRangeIterator(const NDRange &p, unsigned int s, unsigned int e)
            : m_parent(p), m_pos(s), m_end(e)
        {
        }

        bool done() const
        {
            return (m_pos >= m_end);
        }

        unsigned int dim(unsigned int d) const
        {
            unsigned int r = m_pos;

            if(d < (D - 1))
            {
                r %= m_parent.m_totalsizes[d];
            }

            if(d > 0)
            {
                r /= m_parent.m_totalsizes[d - 1];
            }

            return r;
        }

        bool next_dim0()
        {
            m_pos++;

            return !done();
        }

        bool next_dim1()
        {
            m_pos += m_parent.m_sizes[0] - dim(0);

            return !done();
        }

        unsigned int dim0_max() const
        {
            unsigned int offset = std::min(m_end - m_pos, m_parent.m_sizes[0] - dim(0));

            return dim(0) + offset;
        }
    };

    void set_totalsizes()
    {
        unsigned int t = 1;

        for(unsigned int i = 0; i < D; i++)
        {
            if(m_sizes[i] == 0)
            {
                m_sizes[i] = 1;
            }

            t *= m_sizes[i];

            m_totalsizes[i] = t;
        }
    }

public:
    NDRange &operator=(const NDRange &rhs) = default;
    NDRange(const NDRange &rhs)            = default;

    template <typename... T>
    NDRange(T... ts)
        : m_sizes{ ts... }
    {
        set_totalsizes();
    }

    NDRange(const std::array<unsigned int, D> &n)
        : m_sizes(n)
    {
        set_totalsizes();
    }

    NDRangeIterator iterator(unsigned int start, unsigned int end) const
    {
        return NDRangeIterator(*this, start, end);
    }

    unsigned int total_size() const
    {
        return m_totalsizes[D - 1];
    }

    unsigned int get_size(unsigned int v) const
    {
        return m_sizes[v];
    }
};

/** NDCoordinate builds upon a range, but specifies a starting position
 * in addition to a size which it inherits from NDRange
 */
template <unsigned int N>
class NDCoordinate : public NDRange<N>
{
    using int_t     = unsigned int;
    using ndrange_t = NDRange<N>;

    std::array<int_t, N> m_positions{};

public:
    NDCoordinate &operator=(const NDCoordinate &rhs) = default;
    NDCoordinate(const NDCoordinate &rhs)            = default;
    NDCoordinate(const std::initializer_list<std::pair<int_t, int_t>> &list)
    {
        std::array<int_t, N> sizes{};

        std::size_t i = 0;
        for(auto &p : list)
        {
            m_positions[i] = p.first;
            sizes[i++]     = p.second;
        }

        //update the parents sizes
        static_cast<ndrange_t &>(*this) = ndrange_t(sizes);
    }

    int_t get_position(int_t d) const
    {
        assert(d < N);

        return m_positions[d];
    }

    void set_position(int_t d, int_t v)
    {
        assert(d < N);

        m_positions[d] = v;
    }

    int_t get_position_end(int_t d) const
    {
        return get_position(d) + ndrange_t::get_size(d);
    }
}; //class NDCoordinate

using ndrange_t = NDRange<6>;
using ndcoord_t = NDCoordinate<6>;

}  // namespace ops
}  // namespace kai
