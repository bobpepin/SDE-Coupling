// clang++ -std=c++14 -g -O3 -ffast-math -march=native -dynamiclib -o sde.dylib sde.cpp -I eigen

#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <random>
#include <Eigen/Dense>

#include "sde.h"

using namespace Eigen;

typedef float float2 __attribute__ ((ext_vector_type(2)));
typedef uint32_t uint32_2 __attribute__ ((ext_vector_type(2)));
typedef uint64_t uint64_1 __attribute__ ((ext_vector_type(1)));

// typedef struct { float kappa_X, kappa_Y, sigma_X, sigma_Y, invepsilon; } params_t;

// xoroshiro128+ PRNG

class Xoroshiro {
public:
    uint64_t s[2];
    uint64_t next() {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s[1] = rotl(s1, 36); // c

	return result;
    }
    float2 next_float2() {
	uint64_1 x = next();
	x &= 0x007fffff007fffff;
	x |= 0x3f8000003f800000;
	float2 xf = reinterpret_cast<float2>(x);
	xf -= 1.0f;
	/*
	uint32_2 x2 = reinterpret_cast<uint32_2>(x);
	x2 >>= 8;
	float2 xf = { static_cast<float>(x2[0]), static_cast<float>(x2[1]) };
	return xf * (1. / (UINT64_C(1) << 24));
	*/
	return xf;
    }
    float2 boxmuller2() { // returns float2 of standard gaussians
	const float twopi = 2 * M_PI;
	const float mtwo = -2.0;
	float2 u = next_float2();
	u[0] += FLT_EPSILON;
	float a = sqrt(mtwo * log(u[0]));
	float2 r;
	r[0] = sin(twopi*u[1]);
	r[1] = cos(twopi*u[1]);
	return a*r;
    }

    VectorXf gaussVector(Index n) {
	VectorXf r(n);
	Index k;
	for(k=0; k < n-1; k+=2) {
	    float2 g = boxmuller2();
	    r(k) = g[0];
	    r(k+1) = g[1];
	}
	if(k == n-1) {
	    float2 g = boxmuller2();
	    r(k) = g[0];
	}
	return r;
    }
private:
    static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
    }
};

namespace dynamics {

template<size_t _dim, typename _V, typename _M> struct OU {
    typedef _M M;
    typedef _V V;
    static const size_t dim = _dim;
    typedef struct params_ou params;
    static V b(params params, float t, V x) {
        return -x;
    }

    static M sigma(params params, float t, V x) {
        return params.sigma * M::Identity(); //(x.size(), x.size())/sqrt(params.epsilon);
        // return params.sigma_X;
    }
};

template<size_t _dim, typename _V, typename _M> struct OU_reflected {
    typedef _M M;
    typedef _V V;
    static const size_t dim = _dim;
    typedef struct params_ou params;
    static V b(params params, float t, V x) {
        return -x;
    }

    static M sigma(params params, float t, V x) {
        constexpr size_t dimX = dim/2;
        auto X = x.head(dimX);
        auto Y = x.tail(dimX);
        auto e = (X-Y).normalized();
        M sigma = M::Identity();
        sigma.block(dimX, dimX, dimX, dimX) =
            Matrix<float, dimX, dimX>::Identity() - 2 * e * e.transpose();
        return params.sigma * sigma;
    }
};


template<size_t _dimX, size_t _dimY, typename _V, typename _M> struct GD_reflected {
    typedef _M M;
    typedef _V V;
    typedef Matrix<float, _dimX, 1> VX;
    typedef Matrix<float, _dimX, _dimX> MX;
    typedef Matrix<float, _dimY, 1> VY;
    typedef Matrix<float, _dimY, _dimY> MY;
    static const size_t dim = 2*(_dimX + _dimY);
    static const size_t dimX = _dimX;
    static const size_t dimY = _dimY;
    typedef struct params_gd_r params;

    static VX b_X(VX x, VY y) {
        VX r;
        r.setZero();
        r.head(dimY) = y - x.head(dimY);
        return r;
        // return -x;
    }
    static VY b_Y(VX x, VY y) {
        return x.head(dimY)-y;
    }
    static V b(params params, float t, V xi) {
        VX X = xi.head(dimX);
        VY Y = xi.segment(dimX, dimY);
        VX Xt = xi.segment(dimX+dimY, dimX);
        VY Yt = xi.tail(dimY);
        V r;
        r.head(dimX) = params.sqrtalpha * params.sqrtalpha * b_X(X, Y);
        r.segment(dimX, dimY) = b_Y(X, Y);
        r.segment(dimX+dimY, dimX) = params.sqrtalpha * params.sqrtalpha * b_X(Xt, Yt);
        r.tail(dimY) = b_Y(Xt, Yt);
        return r;
    }

    static M sigma(params params, float t, V xi) {
        auto X = xi.head(dimX);
        auto Y = xi.segment(dimX, dimY);
        auto Xt = xi.segment(dimX+dimY, dimX);
        auto Yt = xi.tail(dimX);

        auto sigma_X = params.sqrtalpha * params.sigma_X * MX::Identity();
        
        auto e = (X-Xt).normalized();
        auto sigma_Xt = params.sqrtalpha * params.sigma_X * (MX::Identity() - 2 * e * e.transpose());

        auto sigma_Y = params.sigma_Y * MY::Identity();

        M sigma = M::Identity();
        sigma.block(0, 0, dimX, dimX) = sigma_X;
        sigma.block(dimX, dimX, dimY, dimY) = sigma_Y;
        sigma.block(dimX+dimY, dimX+dimY, dimX, dimX) = sigma_Xt;
        sigma.block(dimX+dimY+dimX, dimX+dimY+dimX, dimY, dimY) = sigma_Y;
        return sigma;
    }
};


}

template<typename Dynamics>
void sde(typename Dynamics::params params,
                  struct sde_input input,
                  struct sde_output output)
{
    constexpr size_t dim = Dynamics::dim;
    typedef typename Dynamics::V V;
    Map<Array<float, Dynamic, Dynamic, ColMajor> > X(output.X, dim, input.N);

    std::mt19937_64 mtgen;
    Xoroshiro Xgen[dim];
    for(size_t i = 0; i < dim; i++) {
        mtgen.seed(input.omega[i]);
        Xgen[i].s[0] = mtgen();
        Xgen[i].s[1] = mtgen();
    }

    float h = input.h;
    float sqh = sqrt(h);
    
    Map<VectorXf> x0(input.x0, dim);
    V x = x0;
    X.col(0) = x0;
    
    for(size_t k=0; k < input.N-1; k++) {
	const float t = k*h;
	auto dx = Dynamics::b(params, t, x);
        auto sx = Dynamics::sigma(params, t, x);
        V dB;
        for(unsigned int i=0; i < dim; i++) {
            float2 dB_X = Xgen[i].boxmuller2();
            dB(i) = dB_X[0];
        }
	x += dx * h + sx * dB * sqh;
	// X.col(k+1) = x;
	// Y.col(k+1) = y;
	X.col(k+1) = x;
    }
}

typedef dynamics::OU<2, Vector2f, Matrix2f> dynou2;
extern "C" void sde_ou_2(dynou2::params params,
                         sde_input input,
                         sde_output output) {
    sde<dynou2>(params, input, output);
}

typedef dynamics::OU_reflected<2, Vector2f, Matrix2f> dynou_r2;
extern "C" void sde_ou_r_2(dynou_r2::params params,
                           sde_input input,
                           sde_output output) {
    sde<dynou_r2>(params, input, output);
}

typedef dynamics::GD_reflected<2, 1, Matrix<float, 6, 1>, Matrix<float, 6, 6> > dyngd_r6;
extern "C" void sde_gd_r_6(dyngd_r6::params params,
                           sde_input input,
                           sde_output output) {
    sde<dyngd_r6>(params, input, output);
}

typedef dynamics::GD_reflected<3, 2, Matrix<float, 10, 1>, Matrix<float, 10, 10> > dyngd_r_3_2;
extern "C" void sde_gd_r_3_2_10(dyngd_r_3_2::params params,
                                sde_input input,
                                sde_output output) {
    sde<dyngd_r_3_2>(params, input, output);
}
