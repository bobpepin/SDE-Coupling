// clang++ -std=c++14 -g -O3 -ffast-math -dynamiclib -o sde_eigen.dylib sde_eigen.cpp -I eigen
// clang++ -std=c++14 -O3 -ffast-math -o sde_eigen sde_eigen.cpp -I eigen


#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

typedef float float2 __attribute__ ((ext_vector_type(2)));
typedef uint32_t uint32_2 __attribute__ ((ext_vector_type(2)));
typedef uint64_t uint64_1 __attribute__ ((ext_vector_type(1)));

typedef struct { float kappa_X, kappa_Y, invepsilon; } params_t;

extern "C"
void sde_eigen(params_t *params,
	       unsigned long omega_X, unsigned long omega_Y,
	       float h, unsigned long N, float t0,
	       unsigned long dimX, unsigned long dimY,
	       float *x0_p, float *y0_p,
	       float *X_p, float *Y_p);

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

template<Index n> using Vecf = Matrix<float, n, 1>;

template<typename V>
V b_X(params_t params, float t, V x, V y) {
    return -params.kappa_X*(x - y);
}

template<typename V>
V b_Y(params_t params, float t, V x, V y) {
    return -params.kappa_Y*(y - x);
}

template<typename V>
float sigma_X(params_t params, float t, V x, V y) {
//    return M::Identity; //(x.size(), x.size())/sqrt(params.epsilon);
    return 1;
}

template<typename V>
float sigma_Y(params_t params, float t, V x, V y) {
//    return M::Identity; //(y.size(), y.size());
    return 1;
}

using Matf1 = Matrix<float, 1, 1>;

void sde_eigen(params_t *params_p,
	       unsigned long omega_X, unsigned long omega_Y,
	       float h, unsigned long N, float t0,
	       unsigned long dimX, unsigned long dimY,
	       float *x0_p, float *y0_p,
	       float *X_p, float *Y_p)
{
    params_t params = *params_p;
    Map<Matrix<float, Dynamic, Dynamic, ColMajor> > X(X_p, dimX, N);
    Map<Matrix<float, Dynamic, Dynamic, ColMajor> > Y(Y_p, dimY, N);
    Map<VectorXf> x0(x0_p, dimX);
    Map<VectorXf> y0(y0_p, dimY);
    std::mt19937_64 mtgen;
    Xoroshiro Xgen, Ygen;
    mtgen.seed(omega_X);
    Xgen.s[0] = mtgen();
    Xgen.s[1] = mtgen();
    mtgen.seed(omega_Y);
    Ygen.s[0] = mtgen();
    Ygen.s[1] = mtgen();
    float sqh = sqrt(h);
    float sqinveps = sqrt(params.invepsilon);
    float inveps = params.invepsilon;
    float x = x0(0);
    float y = y0(0);
    X.col(0) = x0;
    Y.col(0) = y0;
    for(size_t k=0; k < N-1; k++) {
	const float t = k*h;
	auto dx = b_X(params, t, x, y);
	auto dy = b_Y(params, t, x, y);
	float sx = sigma_X(params, t, x, y);
	float sy = sigma_Y(params, t, x, y);
	float2 dB_X = Xgen.boxmuller2();
	float2 dB_Y = Ygen.boxmuller2();
	// x += dx * h + sx * Xgen.gaussVector(dimX) * sqh;
	// y += dy * h + sy * Ygen.gaussVector(dimY) * sqh;
	x += inveps * dx * h + sqinveps * sx * dB_X[0] * sqh;
	y += dy * h + sy * dB_Y[0] * sqh;
	// X.col(k+1) = x;
	// Y.col(k+1) = y;
	X(k+1) = x;
	Y(k+1) = y;
    }
}

int main(int argc, char **argv)
{
    Xoroshiro rng;
    uint64_t s0 = std::strtoull(argv[1], 0, 10);
    uint64_t s1 = std::strtoull(argv[2], 0, 10);
    size_t N = std::strtoull(argv[3], 0, 10);
    rng.s[0] = s0;
    rng.s[1] = s1;
    for(size_t i=0; i < N; i++) {
	float2 g = rng.boxmuller2();
	std::cout << g[0] << " " << g[1] << " ";
//	uint64_t u = rng.next();
//	std::cout << u << " ";
    }
    std::cout << "\n";
}
