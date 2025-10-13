#include "slang-math.h"

#include <limits>
#include <math.h>
#include <stdint.h>
#include <type_traits>
#include <utility>

namespace Slang
{
namespace Mx
{
// Implement OCP Microscaling Formats type conversions.
// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

// Floating point type configuration
struct FloatingConfig
{
    unsigned exponentWidth;
    unsigned mantissaWidth;
    unsigned hasInf : 1;
    unsigned hasNaN : 1;
};

// Floating point type storage
using FloatingStorage = uint32_t;

// Floating point type attribute
struct FloatingAttribute
{
    FloatingStorage max;
    FloatingStorage min;
    FloatingStorage lowest;
    FloatingStorage infinity;
    FloatingStorage signalingNaN;
    FloatingStorage epsilon;
    FloatingStorage denormMin;
    FloatingStorage exponentMask;
    FloatingStorage mantissaMask;
    uint32_t exponentBias;
};

// toggle the n-th bit of storage
static FloatingStorage toggleBit(FloatingStorage storage, FloatingStorage bitIdx)
{
    return storage ^ (FloatingStorage(1u) << bitIdx);
}

// clear the n-th bit of storage
static FloatingStorage clearBit(FloatingStorage storage, FloatingStorage bitIdx)
{
    return storage & ~(FloatingStorage(1u) << bitIdx);
}

// get the attributes of floating point type from type configuration.
static FloatingAttribute getAttribute(const FloatingConfig config)
{
    FloatingAttribute attr;
    attr.exponentMask = ((FloatingStorage(1u) << config.exponentWidth) - FloatingStorage(1u))
                        << config.mantissaWidth;
    attr.mantissaMask = (FloatingStorage(1u) << config.mantissaWidth) - FloatingStorage(1u);
    attr.exponentBias = (1u << (config.exponentWidth - 1)) - 1u;
    attr.max =
        (config.hasInf ? toggleBit(attr.exponentMask, config.mantissaWidth) : attr.exponentMask) |
        (!config.hasInf && config.hasNaN ? toggleBit(attr.mantissaMask, 0u) : attr.mantissaMask);
    attr.min = FloatingStorage(1u) << config.mantissaWidth;
    attr.lowest = toggleBit(attr.max, config.exponentWidth + config.mantissaWidth);
    attr.infinity = config.hasInf ? attr.exponentMask : 0u;
    attr.signalingNaN = config.hasNaN ? attr.exponentMask | attr.mantissaMask : 0u;
    attr.epsilon = (attr.exponentBias - config.mantissaWidth) << config.mantissaWidth;
    attr.denormMin = 1u;
    return attr;
}

// check current floating point value is infinity
static bool isinf(FloatingStorage storage, const FloatingConfig config)
{
    if (!config.hasInf)
    {
        return false;
    }
    const FloatingAttribute attr = getAttribute(config);
    return attr.infinity == clearBit(storage, config.exponentWidth + config.mantissaWidth);
}

// check current floating point value is nan
static bool isnan(FloatingStorage storage, const FloatingConfig config)
{
    const FloatingAttribute attr = getAttribute(config);
    if (!config.hasInf && config.hasNaN)
    {
        // Only one singalingNaN
        return attr.signalingNaN == clearBit(storage, config.exponentWidth + config.mantissaWidth);
    }
    if (config.hasInf && config.hasNaN)
    {
        // Both quiet NaN and signaling NaN. Assume exponent is all 1, and mantissa is any 1.
        return (attr.exponentMask & storage) == attr.exponentMask &&
               (attr.mantissaMask & storage) != 0u;
    }
    return false;
}

// get the fpclassify enum value of current floating point value
static int fpclassify(FloatingStorage storage, const FloatingConfig config)
{
    if (isinf(storage, config))
    {
        return FP_INFINITE;
    }
    if (isnan(storage, config))
    {
        return FP_NAN;
    }
    if (clearBit(storage, config.exponentWidth + config.mantissaWidth) == FloatingStorage(0u))
    {
        return FP_ZERO;
    }
    const FloatingAttribute attr = getAttribute(config);
    if ((attr.exponentMask & storage) == 0u && (attr.mantissaMask & storage) != 0u)
    {
        return FP_SUBNORMAL;
    }
    return FP_NORMAL;
}

static std::
    pair<typename std::make_signed<FloatingStorage>::type /* exp */, FloatingStorage /* mant */>
    normalizeExpMant(
        FloatingStorage storage,
        const FloatingConfig srcConfig,
        const FloatingConfig dstConfig,
        std::float_round_style rounding)
{
    using UnsignedT = FloatingStorage;
    using SignedT = typename std::make_signed<UnsignedT>::type;
    const FloatingAttribute srcAttr = getAttribute(srcConfig);
    SignedT exponent = SignedT((storage & srcAttr.exponentMask) >> srcConfig.mantissaWidth) -
                       SignedT(srcAttr.exponentBias);
    UnsignedT mantissa = storage & srcAttr.mantissaMask;

    // Normalize exponent and mantissa for subnormal
    if (exponent == -SignedT(srcAttr.exponentBias))
    {
        const UnsignedT denormShift = Math::CountlZero32(mantissa) -
                                      (31u - srcConfig.exponentWidth - srcConfig.mantissaWidth) -
                                      1u - srcConfig.exponentWidth;
        exponent -= SignedT(denormShift);
        mantissa = (mantissa << (denormShift + 1u)) & srcAttr.mantissaMask;
    }

    // Shift value for up-casting
    const SignedT numThrowBits =
        SignedT(srcConfig.mantissaWidth) - SignedT(dstConfig.mantissaWidth);
    if (numThrowBits <= 0)
    {
        return std::make_pair(exponent, UnsignedT(mantissa) << -numThrowBits);
    }

    // Normalize exponent from source to target
    const FloatingAttribute dstAttr = getAttribute(dstConfig);
    UnsignedT guardMantissaBit;
    if (exponent <= -SignedT(dstAttr.exponentBias))
    {
        if (exponent >= -SignedT(dstAttr.exponentBias) - SignedT(dstConfig.mantissaWidth))
        {
            mantissa |= UnsignedT(1u) << std::max(srcConfig.mantissaWidth, dstConfig.mantissaWidth);
            const SignedT numGuardMantissaBits =
                numThrowBits - (exponent + SignedT(dstAttr.exponentBias)) + 1u;
            guardMantissaBit = UnsignedT(1u)
                               << (numGuardMantissaBits > 0
                                       ? numGuardMantissaBits
                                       : (numGuardMantissaBits - numThrowBits));
        }
        else
        {
            guardMantissaBit = 0u;
            mantissa = 0u;
        }
        exponent = -SignedT(dstAttr.exponentBias);
    }
    else
    {
        guardMantissaBit = UnsignedT(1u) << numThrowBits;
    }

    // Rounding floating point
    const UnsignedT roundMantissaBit = guardMantissaBit >> 1u;
    const UnsignedT stickyMantissaMask = roundMantissaBit > 0u ? roundMantissaBit - 1u : 0u;

    // Whether rounding to zero for down-casting
    bool roundTowardZero = true;
    const bool isNegative =
        storage & (FloatingStorage(1u) << (srcConfig.exponentWidth + srcConfig.mantissaWidth));
    if (rounding == std::round_toward_infinity)
    {
        roundTowardZero = isNegative || !(mantissa & (roundMantissaBit | stickyMantissaMask));
    }
    else if (rounding == std::round_toward_neg_infinity)
    {
        roundTowardZero = !(isNegative && (mantissa & (roundMantissaBit | stickyMantissaMask)));
    }
    else if (rounding == std::round_to_nearest)
    {
        if ((roundMantissaBit & mantissa) != 0u &&
            (mantissa & (guardMantissaBit | stickyMantissaMask)) != 0u)
            roundTowardZero = false;
    }

    // If not rounding to zero, that might carry out for exponent
    if (!roundTowardZero)
    {
        mantissa += roundMantissaBit | stickyMantissaMask;
        if ((UnsignedT(mantissa) & (UnsignedT(1u) << srcConfig.mantissaWidth)) != 0u &&
            exponent != -SignedT(dstAttr.exponentBias))
        {
            mantissa = 0u;
            exponent++;
        }
    }
    return std::make_pair(
        exponent,
        guardMantissaBit > 0u ? mantissa >> Math::CountrZero32(guardMantissaBit) : 0u);
}

// Implement floating point conversion with rounding mode and saturating.
static FloatingStorage cast(
    FloatingStorage storage,
    const FloatingConfig srcConfig,
    const FloatingConfig dstConfig,
    std::float_round_style rounding,
    bool saturating)
{
    if (srcConfig.exponentWidth == dstConfig.exponentWidth &&
        srcConfig.mantissaWidth <= dstConfig.mantissaWidth &&
        srcConfig.hasInf == dstConfig.hasInf && srcConfig.hasNaN == dstConfig.hasNaN)
    {
        // Convert Floating point type with different mantissa from low precision to high.
        return storage << (dstConfig.mantissaWidth - srcConfig.mantissaWidth);
    }

    const int classify = fpclassify(storage, srcConfig);
    const bool isNegative =
        storage & (FloatingStorage(1u) << (srcConfig.exponentWidth + srcConfig.mantissaWidth));
    const FloatingAttribute dstAttr = getAttribute(dstConfig);

    // Handle zero.
    if (classify == FP_ZERO)
    {
        return isNegative ? toggleBit(0u, dstConfig.exponentWidth + dstConfig.mantissaWidth) : 0u;
    }

    // Handle NaN
    if (srcConfig.hasNaN && dstConfig.hasNaN && classify == FP_NAN)
    {
        return dstAttr.signalingNaN;
    }

    // Get normalized exponent and rounded mantissa
    using UnsignedT = FloatingStorage;
    using SignedT = typename std::make_signed<UnsignedT>::type;
    auto normalized = normalizeExpMant(storage, srcConfig, dstConfig, rounding);
    SignedT exponent = normalized.first;
    UnsignedT mantissa = normalized.second;

    // Handle infinity
    const UnsignedT signBit =
        isNegative ? (UnsignedT(1u) << (dstConfig.exponentWidth + dstConfig.mantissaWidth)) : 0u;
    const bool isInf = classify != FP_NAN &&
                       (classify == FP_INFINITE ||
                        (dstConfig.hasInf && exponent >= SignedT(dstAttr.exponentBias + 1u)) ||
                        (!dstConfig.hasInf && exponent > SignedT(dstAttr.exponentBias + 1u)));
    if (isInf)
    {
        if (!saturating && dstConfig.hasInf)
        {
            return dstAttr.infinity | signBit;
        }
        if (!saturating && dstConfig.hasNaN)
        {
            return dstAttr.signalingNaN | signBit;
        }
        return dstAttr.max | signBit;
    }

    const UnsignedT targetExp =
        UnsignedT((exponent + SignedT(dstAttr.exponentBias)) << dstConfig.mantissaWidth) &
        dstAttr.exponentMask;
    return FloatingStorage(signBit | targetExp | mantissa);
}

static constexpr FloatingConfig floatConfig = FloatingConfig{8, 23, true, true};     // float
static constexpr FloatingConfig bfloat16Config = FloatingConfig{8, 7, true, true};   // bfloat16
static constexpr FloatingConfig floate4m3Config = FloatingConfig{4, 3, false, true}; // float8e4m3
static constexpr FloatingConfig floate5m2Config = FloatingConfig{5, 2, true, true};  // float8e5m2
} // namespace Mx

unsigned short FloatToBfloat16(float val)
{
    const auto x = FloatAsInt(val);
    return Mx::cast(x, Mx::floatConfig, Mx::bfloat16Config, std::round_to_nearest, false);
}

float Bfloat16ToFloat(unsigned short val)
{
    Mx::FloatingStorage casted =
        Mx::cast(val, Mx::bfloat16Config, Mx::floatConfig, std::round_to_nearest, false);
    return IntAsFloat(casted);
}

unsigned char FloatToFloate4m3(float val)
{
    const auto x = FloatAsInt(val);
    return Mx::cast(x, Mx::floatConfig, Mx::floate4m3Config, std::round_to_nearest, false);
}

float Floate4m3ToFloat(unsigned char val)
{
    Mx::FloatingStorage casted =
        Mx::cast(val, Mx::floate4m3Config, Mx::floatConfig, std::round_to_nearest, false);
    return IntAsFloat(casted);
}

unsigned char FloatToFloate5m2(float val)
{
    const auto x = FloatAsInt(val);
    return Mx::cast(x, Mx::floatConfig, Mx::floate5m2Config, std::round_to_nearest, false);
}

float Floate5m2ToFloat(unsigned char val)
{
    Mx::FloatingStorage casted =
        Mx::cast(val, Mx::floate5m2Config, Mx::floatConfig, std::round_to_nearest, false);
    return IntAsFloat(casted);
}

} // namespace Slang
