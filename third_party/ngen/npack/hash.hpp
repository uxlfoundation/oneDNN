#ifndef NGEN_NPACK_HASH_H
#define NGEN_NPACK_HASH_H

namespace NGEN_NAMESPACE {
namespace npack {

/*********************************************/
/* A Jenkins hash function, as found in NEO: */
/*    runtime/helpers/hash.h                 */
/*********************************************/

static inline void hash_jenkins_mix(uint32_t &a, uint32_t &b, uint32_t &c)
{
    // clang-format off
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);
    // clang-format on
}

static inline uint32_t neo_hash(const unsigned char *buf, size_t len)
{
    auto ubuf = (const uint32_t *)buf;

    uint32_t a = 0x428a2f98;
    uint32_t hi = 0x71374491;
    uint32_t lo = 0xb5c0fbcf;

    for (; len >= 4; len -= 4) {
        a ^= *ubuf++;
        hash_jenkins_mix(a, hi, lo);
    }

    if (len > 0) {
        auto rbuf = (const uint8_t *)ubuf;
        uint32_t rem = 0;
        for (; len > 0; len--)
            rem = (rem | *rbuf++) << 8;
        hash_jenkins_mix(rem, hi, lo);
    }

    return lo;
}

} /* namespace npack */
} /* namespace NGEN_NAMESPACE */

#endif /* header guard */
