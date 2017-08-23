#include <cmath>
#include <vector>
#include <random>
#include <iostream>
//#include <x86intrin.h>
#define USE_SSE2
#include "sse_mathfun.h"


//typedef float float4 __attribute__ ((vector_size (sizeof(float)*4)));
typedef float float4 __attribute__ ((ext_vector_type(4)));
typedef uint64_t uint64_2 __attribute__ ((ext_vector_type(2)));
typedef uint32_t uint32_4 __attribute__ ((ext_vector_type(4)));


const int TOTAL = 100000000;
const int N = 100;
const int S = TOTAL / N;
const float h = 1e-2;

extern "C" float b(float, float);
extern "C" void sde(float *x0, float t0, float h, unsigned int N, unsigned int S, float *X);
extern "C" void sde_vec(float *x0, float t0, float h, unsigned int N, unsigned int S, float4 *X);

// xoroshiro128+ PRNG
uint64_t s[2];

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t next(void) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s[1] = rotl(s1, 36); // c

	return result;
}
float next_float(void) {
    uint64_t x = next();
    return (x >> 8) * (1. / (UINT64_C(1) << 24));
}


void boxmuller(float r[2])
{
    float u1, u2, a;
    float twopi = 2 * M_PI;
    float mtwo = -2.0;
    u1 = next_float();
    u2 = next_float();
    a = sqrt(mtwo * log(u1));
    r[0] = sin(twopi*u2);
    r[1] = cos(twopi*u2);
}

// xoroshiro128+ PRNG
thread_local uint64_2 s2[2];

static inline uint64_2 rotl2(const uint64_2 x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_2 next2(void) {
	const uint64_2 s0 = s2[0];
	uint64_2 s1 = s2[1];
	const uint64_2 result = s0 + s1;

	s1 ^= s0;
	s2[0] = rotl2(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s2[1] = rotl2(s1, 36); // c

	return result;
}

float4 next_float4(void) {
    uint32_4 x = (uint32_4)next2();
    float4 xf = (float4)(x >> 8); //{ x[1] >> 8, x[2] >> 8, x[3] >> 8, x[4] >> 8};
    return xf * (1. / (UINT64_C(1) << 24));
}

void boxmuller4(float4 r[2])
{
    float4 u1, u2, a;
    float4 twopi = 2 * M_PI;
    float4 mtwo = -2.0;
    u1 = next_float4();
    u2 = next_float4();
    a = _mm_sqrt_ps(mtwo * log_ps(u1));
    sincos_ps(twopi*u2, r, r+1);
}



float b(float t, float x)
{
    return 2*x*x*x + 2*x*(-1 + x*x);
//    return -x;
}

float4 b4(float t, float4 x)
{
    return 2*x*x*x + 2*x*(-1 + x*x);
//    return -x;
}

#if 0
    for(unsigned int i=0; i < S; i++) {
	for(unsigned int j=0; j < N; j++) {
	    std::cout << X[N*i + j] << ' ';
	}
	std::cout << '\n';
    }
#endif

// (x >> 11) * (1. / (UINT64_C(1) << 53))


void sde_vec(float *x0, float t0, float h, unsigned int N, unsigned int S, float4 *X)
{
    /*
    std::random_device rd;
    std::mt19937 gen1(rd()), gen2(rd()), gen3(rd()), gen4(rd());
    std::normal_distribution<float> G(0, sqrt(h));
    */
    S /= 4;
    for(unsigned int i=0; i < S; i++) {
	X[i] = x0[0];
    }
    float4 hh = {h, h, h, h};
    float4 x = 0;
    for(unsigned int i=0; i < S; i++) {
	for(unsigned int j=0; j < N-1; j+=2) {
//	    float4 dB = { G(gen1), G(gen2), G(gen3), G(gen4) };
	    float4 dB[2]; // = next_float4();
	    boxmuller4(dB);
//	    float4 x = { X[j*S + i], X[j*S + i + 1], X[j*S + i + 2], X[j*S + i + 3] };
//	    x = X[i*N+j];
	    float4 dx = b4(j*h, x);
	    x += dx*hh + dB[0];
	    /*
	    for(unsigned int k=0; k < 4; k++) {
		X[(j+1)*S + i + k] = x[k];
	    }
	    */
	    X[i*N + j + 1] = x;
	    dx = b4((j+1)*h, x);
	    x += dx*hh + dB[1];
	    X[i*N + j + 2] = x;
	}
    }
}
    
void sde(float *x0, float t0, float h, unsigned int N, unsigned int S, float *X)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dB(0, sqrt(h));
    float x;
    for(unsigned int i=0; i < S; i++) {
	x = x0[0];
	X[N*i] = x;
	for(unsigned int j=0; j < N-1; j++) {
	    float dx = b(j*h, x);
//	    x += dx*h + dB(gen);
	    float dB[2];
	    boxmuller(dB);
	    x += dx*h + dB[0];
	    X[N*i + j + 1] = x;
	}
    }
}

int main(int argc, char **argv)
{
    std::vector<float> X(TOTAL);
    std::vector<float> B(TOTAL);
    float x = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dB(0, sqrt(h));
    while(1)
    for(unsigned int i=0; i < S; i++) {
	x = 0;
	X[N*i] = 0;
	for(unsigned int j=0; j < N-1; j++) {
	    x += b(j*h, x)*h + dB(gen);
	    X[N*i + j + 1] = x;
	}
    }
}
