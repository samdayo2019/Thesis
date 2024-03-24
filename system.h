#ifndef SYSTEM_H
#define SYSTEM_H

// #include <cmath>
/// #include <vector>
// #include "math.h"
// #include <corecrt_math_defines.h>
#include "HLS\extendedmath.h"


// FastSLAM constants
#define TICK 0.1
#define SIM_TICK 50.0
#define MAX_RANGE 20.0
#define DIST_MAL 2.0
#define STATE_SIZE 3
#define LM_SIZE 2
#define NUM_PARTICLES TOTAL_NUM_PARTICLES/2
#define TOTAL_NUM_PARTICLES 100
#define N_RESAMPLE NUM_PARTICLES/2
#define OFFSET_YAW_RATE_NOISE 0.01
#define ROWS 3
#define COLS 3
#define pi 3.14159265

#define num_landmarks 8

// Q diagonal matrix 3.0 and 10pi/180 in the diagonals
float Q_matrix[2][2] = {{9.0, 0.0}, {0.0, (float)pow(10.0*pi/180, 2)}};
// R diagonal matrix
float R_matrix[2][2] = {{1.0, 0.0}, {0.0, (float)pow(20.0*pi/180, 2)}};

float Q_sim[2][2] = {{(float)pow(0.3, 2), 0.0}, {0.0, (float)pow(2.0*pi/180, 2)}};
float R_sim[2][2] = {{(float)pow(0.5, 2), 0.0}, {0.0, (float)pow(10.0*pi/180, 2)}};

bool shown_animation = true; 

// int num_landmarks = 0;

float check = 0; 

float time_step = 0;

class Particle
{
public:
    float w; 
    float x; 
    float y; 
    float yaw; 
    // initializing 2 multidimensional vectors. 
    float** lm;
    float** lm_cov;
    float** P;
    Particle(int number_landmarks);
    Particle() = default;
};

void fast_slam1(Particle* particles, float control[2], float z[3][num_landmarks], int num_cols);
void motion_model(float states[STATE_SIZE], float control[2]);
void update_landmark(Particle& particle, float z[3], float (&Q_mat)[2][2]);
void update_with_observation(Particle* particles, float z[STATE_SIZE][num_landmarks], int num_cols);
void add_new_landmark(Particle& particle, float z[3], float (&Q_mat)[2][2]);
void proposal_sampling(Particle& particle, float z[3], float (&Q_mat)[2][2]);



// float* motion_model(float* states, float* control);
float pi_2_pi(float value);

float get_rand_gaussian(float min, float max);
float get_rand_uniform(float min, float max);

float compute_weight(Particle particle, float z[STATE_SIZE], float (&Q_mat)[2][2]); 
void compute_jacobians(Particle particle, float (&xf)[2], float (&pf)[2][2], float (&Q_mat)[2][2], float (Hf)[2][2], float (&Hv)[2][3], float (&Sf)[2][2], float (&zp)[2]);


void update_kf_with_cholesky(float (&xf)[2], float (&pf)[2][2], float (&dz)[2], float (&Q_mat)[2][2], float (&Hf)[2][2]);
void predict_particles(Particle* particles, float control[2]); 
void resampling(Particle* particles); 
void normalize_weight(Particle* particles);

void observation(float xTrue[3], float xd[3], float u[2], float rfid[num_landmarks][2], int num_id, float ud[2], int& num_cols, float z[3][num_landmarks]);
void calc_input(float time, float u[2]);
void calc_final_state(Particle* particles, float xEst[3]);




float dot_product(float (&row_vec)[2], float (&col_vec)[2]);
float det(float (&matrix)[2][2]);
void mult_mat(float matrix1[2][2], float matrix2[2][2]);
bool cholesky_decomp(float S[2][2], float vector_b[2], int n);
bool cholesky_decomp(float S[3][3], float vector_b[3], int n);
void inverse(float matrix[2][4], int n);
void inverse(float matrix[3][6], int n);
void matrix_vector_3(float (&matrix)[3][2], float (&vector)[2], float (&result)[3]);
void matrix_vector_33(float matrix[3][3], float (&vector)[3], float (&result)[3]);
void matrix_vector(float matrix[2][2], float (&vector)[2], float (&result)[2]);
float clamp(float val, float low, float high);
float vector_vector(float row_vec[NUM_PARTICLES], float col_vec[NUM_PARTICLES]); 
void cumulative_sum(float array[NUM_PARTICLES], float sum[NUM_PARTICLES]); 
void transpose_mat(float matrix[2][2]);

#endif