#ifndef SYSTEM_H
#define SYSTEM_H

#include <cmath>
#include <stdint.h>
#include <vector>
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

// Q diagonal matrix 3.0 and 10pi/180 in the diagonals
float Q_matrix[2][2] = {{9.0, 0.0}, {0.0, (float)pow(10.0*M_PI/180, 2)}};
// R diagonal matrix
float R_matrix[2][2] = {{1.0, 0.0}, {0.0, (float)pow(20.0*M_PI/180, 2)}};

float Q_sim[2][2] = {{(float)pow(0.3, 2), 0.0}, {0.0, (float)pow(2.0*M_PI/180, 2)}};
float R_sim[2][2] = {{(float)pow(0.5, 2), 0.0}, {0.0, (float)pow(10.0*M_PI/180, 2)}};

bool shown_animation = true; 

int num_landmarks = 0;

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
    Particle(int num_landmarks);
    Particle() = default;
};


void motion_model(float* states, float* control);
Particle update_landmark(Particle particle, float *z, float (&Q_mat)[2][2]);
Particle* update_with_observation(Particle* particles, float** z, int num_cols);
Particle add_new_landmark(Particle particle, float *z, float (&Q_mat)[2][2]);

void transpose_mat(float** matrix);

// float* motion_model(float* states, float* control);
float pi_2_pi(float value);

void mult_mat(float** matrix1, float** matrix2, int n);
float compute_weight(Particle particle, float* z, float (&Q_mat)[2][2]);
float dot_product(float (&row_vec)[2], float (&col_vec)[2]);
float det(float (&matrix)[2][2]);
void compute_jacobians(Particle particle, float (&xf)[2], float (&pf)[2][2], float (&Q_mat)[2][2], float (Hf)[2][2], float (&Hv)[2][3], float (&Sf)[2][2], float (&zp)[2]);

// bool cholesky_decomp(float** S, int n);

void update_kf_with_cholesky(float (&xf)[2], float (&pf)[2][2], float (&dz)[2], float (&Q_mat)[2][2], float (&Hf)[2][2]);
void matrix_vector(float (&matrix)[2][2], float (&vector)[2], float (&result)[2]);
Particle* predict_particles(Particle* particles, float* control);
Particle* resampling(Particle* particles);
Particle* normalize_weight(Particle* particles);
void cumulative_sum(float* array, float* sum);
float** observation(float* xTrue, float* xd, float* u, float** rfid, uint8_t num_id, float* ud, int& num_cols);
void calc_input(float time, float* u);
void calc_final_state(Particle* particles, float* xEst);
Particle* fast_slam1(Particle* particles, float* control, float** z, int num_cols);
float vector_vector(float* row_vec, float* col_vec, int num_rows);
Particle proposal_sampling(Particle particle, float*z, float (&Q_mat)[2][2]);
void matrix_vector_3(float (&matrix)[3][2], float (&vector)[2], float (&result)[3]);

void matrix_vector_33(float** matrix, float (&vector)[3], float (&result)[3]);
float invert_3x3(float(&matrix)[3][3], float (&inverse)[3][3], int n);
float determinant(float A[3][3], int n);
void getCofactor(float A[3][3], float temp[3][3], int p, int q, int n);
void adjoint(float A[3][3], float Adj[3][3]); 
void forward_elim(float** lower, int n);

void inverse(float** matrix, int n);
void matrix_vector(float** matrix, float (&vector)[2], float (&result)[2]);

bool cholesky_decomp(float** S, float* vector_b, int n);
float clamp(float val, float low, float high);



#endif