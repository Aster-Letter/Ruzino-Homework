#pragma once
namespace Ruzino {

inline int div_ceil(int dividend, int divisor)
{
    return (dividend + (divisor - 1)) / divisor;
}

}  // namespace Ruzino