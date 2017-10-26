#ifndef EIGEN_AVX_SCALAR_H
#define EIGEN_AVX_SCALAR_H

#define USE_SSE2
#include "sse_mathfun.h"

//typedef float float4 __attribute__ ((vector_size (sizeof(float)*4)));
typedef float float4 __attribute__ ((ext_vector_type(4)));
typedef uint64_t uint64_2 __attribute__ ((ext_vector_type(2)));
typedef uint32_t uint32_4 __attribute__ ((ext_vector_type(4)));

#include <adolc/adouble.h>
#include <Eigen/Core>
namespace Eigen {
template<> struct NumTraits<adtl::adouble>
 : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef adtl::adouble Real;
  typedef adtl::adouble NonInteger;
  typedef adtl::adouble Nested;
  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};
}
namespace avx {
inline const adouble& conj(const adouble& x)  { return x; }
inline const adouble& real(const adouble& x)  { return x; }
inline adouble imag(const adouble&)    { return 0.; }
inline adouble abs(const adouble&  x)  { return fabs(x); }
inline adouble abs2(const adouble& x)  { return x*x; }
}
#endif
