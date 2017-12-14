/* #ifndef SDE_COUPLING_H */
/* #define SDE_COUPLING_H */

struct params_ou {
    float sigma;
};

struct params_ou_r {
    float sigma;
};

struct params_gd_r {
    float sqrtalpha; // Acceleration
    float sigma_X, sigma_Y;
};

struct params_gd_r_3_2 {
    float sqrtalpha; // Acceleration
    float sigma_X, sigma_Y;
};

struct sde_input {
    float h;
    float t0;
    float *x0;
    unsigned long *omega;
    unsigned long N;
};

struct sde_output {
    float *X;
};

/* #endif */
