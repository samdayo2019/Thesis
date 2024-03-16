#include "system.h"
#include <string>
#include <fstream>
#include <iomanip>
// #include <sys/time.h>
#include <queue>
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>
#include <math.h>

// #include <corecrt_math_defines.h>
// #include <bits/stdc++.h>

//Includes for HLS
#include "HLS\math.h"
#include "HLS\extendedmath.h"
#include "HLS\hls.h"
#include "HLS\stdio.h" // to be able to use printf properly


using namespace std;

Particle::Particle(int number_landmarks)
{
    uint8_t i;
    uint16_t j;
    w = 1.0/number_landmarks; // each particle get uniform weight
    //each particle starts at point 0,0
    x = 0.0; 
    y = 0.0; 
    yaw = 0.0; 
    lm = new float*[number_landmarks];
    lm_cov = new float*[number_landmarks*LM_SIZE];
    P = new float*[3];

    //init lm and lm_cov to be all zeroes
    for (i = 0; i < number_landmarks; i++){
        lm[i] = new float[LM_SIZE];
    }

    for (j = 0; j < number_landmarks*LM_SIZE; j++){
        lm_cov[j] = new float[LM_SIZE];
    }
    for (i = 0; i < 3; i++){
        P[i] = new float[3];
    }

    for(i = 0; i < number_landmarks; i ++){
        for (j = 0; j < LM_SIZE; j++){
            lm[i][j] = 0.0; 
        }
    }

    for(i = 0; i < number_landmarks*LM_SIZE; i ++){
        for (j = 0; j < LM_SIZE; j++){
            lm_cov[i][j] = 0.0; 
        }
    }

    P[0][0] = 1; 
    P[1][1] = 1; 
    P[2][2] = 1; 
}

void fast_slam1(Particle* particles, float control[2], float z[3][num_landmarks], int num_cols){
    predict_particles(particles, control);

    // printf("Particle 0 x coordinate: %f; y coordinate: %f \n", particles[0].x, particles[0].y);

    update_with_observation(particles, z, num_cols);
  
    resampling(particles);
}

// control is a 2d vector (v, w)
void predict_particles(Particle* particles, float control[2]){
    // printf("Pre Particle 0 x coordinate: %f; y coordinate: %f \n", particles[0].x, particles[0].y);

    float px[STATE_SIZE]  = {};
    // float state[STATE_SIZE];
    // px = state;
    float noise[2], prod[2]; //we store noise as a column vector
    // float r_mat[2][2]; 
    random_device rd; 
    mt19937 gen(rd()); 
    // uniform_real_distribution<float> distribution(-1.0, 1.0); 
    normal_distribution<float> distribution1(0, 0.35); 
    normal_distribution<float> distribution2(0, 0.35);
    uint8_t i, j, k;

    for (i = 0; i < NUM_PARTICLES; i ++){ 
        // init coln with zeroes
        for (j = 0; j < STATE_SIZE; j++){
            px[j] = 0;
        }

        // load the particle pose into our state vector
        px[0] = particles[i].x;
        px[1] = particles[i].y;
        px[2] = particles[i].yaw;

        noise[0] = distribution1(gen); 
        noise[1] = distribution2(gen);
        // printf("Noise 1: %f; Noise 2: %f \n", noise[0], noise[1]);

        for (j = 0; j < 2; j++){
            float sum = 0.0;
            for (k = 0; k < 2; k++){
                sum += noise[k]*pow(R_matrix[j][k], 0.5);
            }
            //prod is kept as a column vector --> conduct a transpose on the noise, r matrix product
            prod[j] = sum;
        }

        //we keep control as a column vector, but it should be a row vector
        // This add noise to the robot's command using the process noise matrix and the gaussian noise distribution.
        prod[0] += control[0];
        prod[1] += control[1];

        // if(i < 2) printf("Pre Px 1: %f; Pre Px 2: %f; Pre Px 3: %f \n", px[0], px[1], px[2]);

        //Returns the particle's estimated position using the noise motion command and it's initial known state vector
        printf("Time: %f; px: %f, %f, %f\n", time_step, px[0], px[1], px[2]);
        motion_model(px, prod); // pxrod= u + noise*r matrix --> add gaussian noise to the command accounting for our coveriance
        
        particles[i].x = px[0];
        particles[i].y = px[1];
        particles[i].yaw = px[2];

        // if(i < 2) printf("Post Px 1: %f; Post Px 2: %f; Post Px 3: %f \n", px[0], px[1], px[2]);
    }
    // printf("Post Particle 0 x coordinate: %f; y coordinate: %f \n", particles[0].x, particles[0].y);

    // return particles; 
}

void update_with_observation(Particle* particles, float z[STATE_SIZE][num_landmarks], int num_cols){     
    uint8_t landmark_id = 0;
    uint8_t i;
    uint8_t j;
    float obs[3] = {};
    float weight = 0;
    for (i = 0; i < num_cols; i++){
        //cout << (int)i << endl; 
        landmark_id = (uint8_t)z[2][i]; // this assumes that the data association problem is known
        //cout << "Landmark ID" << (int)landmark_id << endl; 
        obs[0] = z[0][i]; 
        obs[1] = z[1][i];
        obs[2] = z[2][i];        
        for (j = 0; j < NUM_PARTICLES; j++){
            if (abs(particles[j].lm[landmark_id][0]) <= 0.01){
                add_new_landmark(particles[j], obs, Q_matrix);  
            }
            else{             
                weight = compute_weight(particles[j], obs, Q_matrix);               
                particles[j].w *= weight; 
                //cout << "Weight: " << weight << endl;
                update_landmark(particles[j], obs, Q_matrix);
                proposal_sampling(particles[j], obs, Q_matrix);
            }
        }
    }
}  

void proposal_sampling(Particle& particle, float z[3], float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float HvT[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    float mult[2][3] = {{0, 0, 0}, {0, 0, 0}};
    float mult_3[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    float Sf[2][2] =  {{0,0}, {0,0}};
    float sInv[2][4] = { 0 };
    float P[3][6] = { 0 }; 

    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            P[i][j] = 0; 
            P[i][j + 3] = 0; 
        }
    }
    float zp[2]; //treat zp as a column vector
    float dz[2]; //treat dx as a column vector
    float res[2] = {0.0, 0.0}; //treat dx as a column vector
    float res_2[3] = {0.0, 0.0, 0.0}; //treat dx as a column vector
    float result[3] = {0.0, 0.0, 0.0}; //treat dx as a column vector


    xf[0] = particle.lm[lm_id][0];
    xf[1] = particle.lm[lm_id][1];


    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] = particle.lm_cov[2*lm_id + i][j];
        }
    }

    compute_jacobians(particle, xf, pf, Q_mat, Hf, Hv, Sf, zp);

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            sInv[i][j] = Sf[i][j];
        }
    }

    inverse(sInv, 2);

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            Sf[i][j] = sInv[i][j + 2]; 
        }
    }


    dz[0] = z[0] - zp[0];
    dz[1] = pi_2_pi(z[1] - zp[1]); 

    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            P[i][j] = particle.P[i][j];
        }
    }

    inverse(P, 3);

    HvT[0][0] = Hv[0][0]; 
    HvT[1][0] = Hv[0][1];
    HvT[2][0] = Hv[0][2]; 
    HvT[0][1] = Hv[1][0]; 
    HvT[1][1] = Hv[1][1]; 
    HvT[2][1] = Hv[1][2];

    //S^(-1)*HV

    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 3; j++){
            mult[i][j] = 0;
            for (int k = 0; k < 2; k++){
                mult[i][j] += sInv[i][k + 2]*Hv[k][j];
            }
        }
    }

    // H^T * (S^-1)* H
    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            mult_3[i][j] = 0;
            for (int k = 0; k < 2; k++){
                mult_3[i][j] += HvT[i][k]*mult[k][j];
            }
        }
    }

    // P + H^T*S^-1*H
    for (int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            P[i][j] = P[i][j + 3] + mult_3[i][j];
            P[i][j + 3] = 0; 
        }
    }

    inverse(P, 3);

    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            particle.P[i][j] = P[i][j + 3];
            P[i][j + 3] = 0;
        }
    }

    matrix_vector(Sf, dz, res);
    matrix_vector_3(HvT, res, res_2);

    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            mult_3[i][j] = particle.P[i][j];
        }
    }

    matrix_vector_33(mult_3, res_2, result);

    particle.x += result[0]; 
    particle.y += result[1]; 
    particle.yaw += result[2];
}

void matrix_vector_3(float (&matrix)[3][2], float (&vector)[2], float (&result)[3]){
    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 2; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void matrix_vector_33(float matrix[3][3], float (&vector)[3], float (&result)[3]){
    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void matrix_vector(float matrix[2][2], float (&vector)[2], float (&result)[2]){
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void inverse(float matrix[2][4], int n){
    float temp_val; 
    float temp[4] = {};

    for (int i = 0; i < n; i++){
        for (int j = 0; j <2 * n; j++){
            // if(j < n){
            //     matrix[i][j] = clamp(matrix[i][j], -512, 511.984375);
            // }
            if(j == (i + n))
                matrix[i][j] = 1; 
        }
    }

    // perform row swapping
    for (int i = n - 1; i > 0; i--){
        if(matrix[i - 1][0] < matrix[i][0]){
            for(int j = 0; j < 2*n; j++){
                temp[j] = matrix[i][j];
                matrix[i][j] = matrix[i-1][j];
                matrix[i - 1][j] = temp[j];
            } 
            
        }
    }

    //replace row by sum of itself and constant multiple of another row
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(j != i){
                // temp_val = clamp(matrix[j][i] / matrix[i][i], -2097152, 2097151.9990234375);
                temp_val = matrix[j][i] / matrix[i][i];
                // min_value = (temp_val < min_value) ? temp_val : min_value;
                // max_value = (temp_val > max_value) ? temp_val : max_value;
                for (int k = 0; k < 2 * n; k++){
                    // min_value = (matrix[j][k] < min_value) ? matrix[j][k] : min_value;
                    // max_value = (matrix[j][k] > max_value) ? matrix[j][k] : max_value;
                    // matrix[j][k] = clamp(matrix[j][k] - matrix[i][k] * temp_val, -2097152, 2097151.9990234375);
                    matrix[j][k] = matrix[j][k] - matrix[i][k] * temp_val;

                }
            }
        }
    }

    for (int i = 0; i < n; i++){
        temp_val = matrix[i][i]; 
        for (int j = 0; j < 2 * n; j++){
            // matrix[i][j] = clamp(matrix[i][j] / temp_val, -2097152, 2097151.9990234375);
            matrix[i][j] = matrix[i][j] / temp_val;
        }
    }
}

void inverse(float matrix[3][6], int n){
    float temp_val; 
    float temp[6] = {};

    for (int i = 0; i < n; i++){
        for (int j = 0; j <2 * n; j++){
            // if(j < n){
            //     matrix[i][j] = clamp(matrix[i][j], -512, 511.984375);
            // }
            if(j == (i + n))
                matrix[i][j] = 1; 
        }
    }

    // perform row swapping
    for (int i = n - 1; i > 0; i--){
        if(matrix[i - 1][0] < matrix[i][0]){
            for(int j = 0; j < 2*n; j++){
                temp[j] = matrix[i][j];
                matrix[i][j] = matrix[i-1][j];
                matrix[i - 1][j] = temp[j];
            } 
            
        }
    }

    //replace row by sum of itself and constant multiple of another row
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(j != i){
                // temp_val = clamp(matrix[j][i] / matrix[i][i], -2097152, 2097151.9990234375);
                temp_val = matrix[j][i] / matrix[i][i];
                // min_value = (temp_val < min_value) ? temp_val : min_value;
                // max_value = (temp_val > max_value) ? temp_val : max_value;
                for (int k = 0; k < 2 * n; k++){
                    // min_value = (matrix[j][k] < min_value) ? matrix[j][k] : min_value;
                    // max_value = (matrix[j][k] > max_value) ? matrix[j][k] : max_value;
                    // matrix[j][k] = clamp(matrix[j][k] - matrix[i][k] * temp_val, -2097152, 2097151.9990234375);
                    matrix[j][k] = matrix[j][k] - matrix[i][k] * temp_val;

                }
            }
        }
    }

    for (int i = 0; i < n; i++){
        temp_val = matrix[i][i]; 
        for (int j = 0; j < 2 * n; j++){
            // matrix[i][j] = clamp(matrix[i][j] / temp_val, -2097152, 2097151.9990234375);
            matrix[i][j] = matrix[i][j] / temp_val;
        }
    }
}

float clamp(float val, float low, float high){
    if(val < low) return low; 
    else if(val > high) return high; 
    else return val; 
}

void update_landmark(Particle& particle, float z[3], float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float Sf[2][2] =  {{0,0}, {0,0}};
    float zp[2]; //treat zp as a column vector
    float dz[2]; //treat dx as a column vector

    xf[0] = particle.lm[lm_id][0];
    xf[1] = particle.lm[lm_id][1];

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] = particle.lm_cov[2*lm_id + i][j];
        }
    }
    
    /*
    Sf = HCovH^T + Qt
    Hf and Hv are two parts of the jacobian of the observation function
    Zp = opservation function
    pf holds the landmark covariance matrix
    xf holds the landmark location used to calculate the measurement prediction 
    */
    compute_jacobians(particle, xf, pf, Q_mat, Hf, Hv, Sf, zp);
    // dz holds the innovation
    dz[0] = z[0] - zp[0];
    dz[1] = pi_2_pi(z[1] - zp[1]); 

    update_kf_with_cholesky(xf, pf, dz, Q_matrix,  Hf); 

    particle.lm[lm_id][0] = xf[0];
    particle.lm[lm_id][1] = xf[1];

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            particle.lm_cov[2*lm_id + i][j] = pf[i][j];
        }
    }
}


/*
xf holds the landmark locations to compute the measurement prediction 
dz holds the measurement innovation 
pf holds the landmark covariance
q_mat is the user defined Q matrix (measure of how much we believe the measurements)
Hf is the jacobian 
*/
void update_kf_with_cholesky(float (&xf)[2], float (&pf)[2][2], float (&dz)[2], float (&Q_mat)[2][2], float (&Hf)[2][2]){
    float HfT[2][2] = { 0 };
    float H[2][2] = { 0 };   
    float PHt[2][2] = { 0 }; 
    float S[2][4] = { 0 };  
    float sChol[2][2] = { 0 };  
    float ST[2][2]  { 0 }; 

    float L_matrix[2][2] = { 0 }; 
    float L_trans[2][2] = { 0 }; 
    float x[2] = {0,0};
    float vect[2] = {0,0};

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            HfT[i][j] = Hf[i][j]; 
            H[i][j] = Hf[i][j]; 
            PHt[i][j] = pf[i][j];
            ST[i][j] = 0.0;
        }
    }

   // computer Q = H * cov * H^T + Qt
    transpose_mat(HfT);
    mult_mat(PHt, HfT);
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            PHt[i][j] = HfT[i][j];
        }
    }
    
    mult_mat(H, HfT);
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            ST[i][j] = HfT[i][j] + Q_mat[i][j];
            sChol[i][j] = ST[i][j];
        }
    }

    //S = H*cov*H^T + Q
    // compute Q^T
    transpose_mat(ST);


    // make s symmetric
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            sChol[i][j] += ST[i][j];
            sChol[i][j] *= 0.5;
        }
    }

    cholesky_decomp(sChol, vect, 2);

    transpose_mat(sChol);

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            S[i][j] = sChol[i][j]; 
        }
    }

    inverse(S, 2);

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            L_trans[i][j] = S[i][j + 2];

        }
    }

    transpose_mat(L_trans);


    mult_mat(PHt, L_matrix); //W1
    mult_mat(L_matrix, L_trans); //W


    // x = W*V
    matrix_vector(L_trans, dz, x);

    // copy W1 into L_mattrans
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            L_trans[i][j] = L_matrix[i][j];
        }
    }

    transpose_mat(L_trans); //W1.T

    mult_mat(L_matrix, L_trans); // W1 * W1.T

    // x = W*v + Xf --> W*V represents K*i where is the innovation
    xf[0] += x[0]; 
    xf[1] += x[1]; 

    // P = pf - W1*W1.T
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] -= L_trans[i][j];
        }
    }
}

bool cholesky_decomp(float S[2][2], float vector_b[2], int n){
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            S[i][j] = 0; 
        }
    }

    // forward substitution solving Ly = b 
    for(int j = 0; j < n; j ++){ // j = 0; j 1
        if(S[j][j] <= 0){
            // cout << "Matrix is not positive definite" << endl; 
            return false;
        }
        S[j][j] = (float)sqrt(S[j][j]); // costly sqrt operation 
        // S[j][j] = clamp((float)sqrt(S[j][j]), -512, 511.984375); // costly sqrt operation 
        vector_b[j] = vector_b[j] / S[j][j]; 
        // vector_b[j] = clamp(vector_b[j] / S[j][j], -512, 511.984375); 
        for(int i = j + 1; i < n; i++){
            if(i < n){
                S[i][j] = S[i][j] / S[j][j]; 
                // S[i][j] = clamp(S[i][j] / S[j][j], -512, 511.984375); 
                vector_b[i] = vector_b[i] - S[i][j]*vector_b[j];
                // vector_b[i] = clamp(vector_b[i] - S[i][j]*vector_b[j], -512, 511.984375);
                for(int k = j + 1; k < i + 1; k++){
                    S[i][k] = S[i][k] - (S[i][j] * S[k][j]);
                    // S[i][k] = clamp(S[i][k] - (S[i][j] * S[k][j]), -512, 511.984375);
                }
            }
            else break;
        }
    }
    
    return true;    
}


bool cholesky_decomp(float S[3][3], float vector_b[3], int n){
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            S[i][j] = 0; 
        }
    }

    // forward substitution solving Ly = b 
    for(int j = 0; j < n; j ++){ // j = 0; j 1
        if(S[j][j] <= 0){
            // cout << "Matrix is not positive definite" << endl; 
            return false;
        }
        S[j][j] = (float)sqrt(S[j][j]); // costly sqrt operation 
        // S[j][j] = clamp((float)sqrt(S[j][j]), -512, 511.984375); // costly sqrt operation 
        vector_b[j] = vector_b[j] / S[j][j]; 
        // vector_b[j] = clamp(vector_b[j] / S[j][j], -512, 511.984375); 
        for(int i = j + 1; i < n; i++){
            if(i < n){
                S[i][j] = S[i][j] / S[j][j]; 
                // S[i][j] = clamp(S[i][j] / S[j][j], -512, 511.984375); 
                vector_b[i] = vector_b[i] - S[i][j]*vector_b[j];
                // vector_b[i] = clamp(vector_b[i] - S[i][j]*vector_b[j], -512, 511.984375);
                for(int k = j + 1; k < i + 1; k++){
                    S[i][k] = S[i][k] - (S[i][j] * S[k][j]);
                    // S[i][k] = clamp(S[i][k] - (S[i][j] * S[k][j]), -512, 511.984375);
                }
            }
            else break;
        }
    }
    
    return true;    
}

void add_new_landmark(Particle& particle, float z[3], float (&Q_mat)[2][2]){
    float r = z[0]; 
    float b = z[1]; 
    uint8_t lm_id = (uint8_t)z[2]; 
    float gz[2][2] = { 0 }; 
    float sChol[2][2] = { 0 }; 
    float sChol_trans[2][2] = { 0 };
    float Q[2][2] = { 0 }; 
    float s = sin(pi_2_pi(particle.yaw + b));
    float c = cos(pi_2_pi(particle.yaw + b));
    float vect_b[2] = {}; 
    float I[2][2] = {{1, 0}, {0, 1}};

    particle.lm[lm_id][0] = particle.x + r*c; 
    particle.lm[lm_id][1] = particle.y + r*s; 
    
    // adding the landmark covariance

    float dx = r * c; 
    float dy = r * s; 
    float d2 = (float)(pow(dx, 2) + pow(dy, 2));
    float d = (float)(sqrt(d2));

    //jacobian of kinematic motion model
    gz[0][0] = dx/d;
    gz[0][1] = dy/d; 
    gz[1][0] = -dy/d2; 
    gz[1][1] = dx/d2; 
    
    //calculate for Z in G = LZ so that we can invert it
    for(int j = 0; j < 2; j++){
        for(int k = 0; k < 2; k++){
            vect_b[k] = gz[k][j]; // fill in b with column i of the identity matrix
            for (int i = 0; i < 2; i++){
                Q[k][i] = Q_mat[k][i];
            }
        }  
        cholesky_decomp(Q, vect_b, 2);// compute the L matrix and solve for x (ith col of inverse) using foward elemination and backwards substitution
        for(int k = 0; k < 2; k++){
            sChol[k][j] = vect_b[k]; 
            sChol_trans[k][j] = vect_b[k]; // store the ith column in the ith column of the inverse
        }     
    }

    //transpose Z ==> Z^T
    transpose_mat(sChol_trans);

    // Z^T *Z -> sChol
    mult_mat(sChol_trans, sChol);

    // compute the inverse of Z^T*Z using forward elimination 
    for(int j = 0; j < 2; j++){
        for(int k = 0; k < 2; k++){
            vect_b[k] = I[k][j]; // fill in b with column i of the identity matrix
            for (int i = 0; i < 2; i++){
                Q[k][i] = sChol[k][i]; // copy the sChol into Q
            }
        }  
        cholesky_decomp(Q, vect_b, 2);// compute the L matrix and solve for x (ith col of inverse) using foward elemination 
        for(int k = 0; k < 2; k++){
            sChol_trans[k][j] = vect_b[k]; // store the ith column in the ith column of the inverse
        }     
    }

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            // float val = gv_trans[i][j];
            particle.lm_cov[2*lm_id + i][j] = sChol_trans[i][j];
        }
    }
}

void resampling(Particle* particles){
    float pw[NUM_PARTICLES] = {};    
    float n_eff = 0.0;
    float w_cumulative[NUM_PARTICLES] = {};
    float base[NUM_PARTICLES] = {};
    float resample_id[NUM_PARTICLES] = {};
    vector<float> weights;
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<float> distribution(-1.0, 1.0); 
    uint8_t indices[NUM_PARTICLES] = {}; 
    Particle tmp_particles[NUM_PARTICLES] = {};

    normalize_weight(particles);
    

    uint16_t i = 0;
    for (i = 0; i < NUM_PARTICLES; i++){
        pw[i] = particles[i].w;
    }

    float dp = vector_vector(pw, pw);
    n_eff = 1.0/dp;

    if (n_eff < N_RESAMPLE)
    {
        cumulative_sum(pw, w_cumulative); // get the cummulative sum of all the weights into w_cummulative 
        
        // normalize the commulative sun
        for (i = 0; i < NUM_PARTICLES; i++){
            pw[i] = (float)1.0/NUM_PARTICLES;
        }
        uint8_t index = 0;
        cumulative_sum(pw, base); // calculate the new normalized cummulative sum
        for (i = 0; i < NUM_PARTICLES; i++){
            base[i] -= 1/NUM_PARTICLES;
            resample_id[i] = distribution(gen);
            resample_id[i] /= NUM_PARTICLES;
            resample_id[i] += base[i];

            while (index < NUM_PARTICLES-1 && resample_id[i] > w_cumulative[index]){
                index++;
            }
            indices[i] = index; 
        }

        for (i = 0; i < NUM_PARTICLES; i++)
        {
           tmp_particles[i] = particles[i];
        }

        uint8_t j, k;

        for(i = 0; i < NUM_PARTICLES; i++){
            particles[i].x = tmp_particles[indices[i]].x;
            particles[i].y = tmp_particles[indices[i]].y;
            particles[i].yaw = tmp_particles[indices[i]].yaw;
            for(j = 0; j < num_landmarks; j++){
                for (k = 0; k < LM_SIZE; k++){
                    particles[i].lm[j][k] = tmp_particles[indices[i]].lm[j][k]; 
                }
            }    

            for(j = 0; j < num_landmarks*LM_SIZE; j++){
                for (k = 0; k < LM_SIZE; k++){
                    particles[i].lm_cov[j][k] = tmp_particles[indices[i]].lm_cov[j][k]; 
                }
            } 
            particles[i].w = 1.0/NUM_PARTICLES; 
        }
    }
}

void cumulative_sum(float array[NUM_PARTICLES], float sum[NUM_PARTICLES]){
    sum[0] = array[0];
    for (uint16_t i = 1; i < NUM_PARTICLES; i++){
        sum[i] = sum[i-1] + array[i];
    }

}

float vector_vector(float row_vec[NUM_PARTICLES], float col_vec[NUM_PARTICLES]){
    float sum = 0.0;
    for (int i = 0; i < NUM_PARTICLES; i++){
        sum += row_vec[i]*col_vec[i];
    }
    return sum; 
}

void normalize_weight(Particle* particles){
    float sum_weights = 0.0; 
    for (int i = 0; i < NUM_PARTICLES; i++){
        sum_weights+=particles[i].w;
    }

    if (sum_weights != 0.0){
        for (int i = 0; i < NUM_PARTICLES; i++){
            particles[i].w /= sum_weights;
        }
    }
    else{
        for (int i = 0; i < NUM_PARTICLES; i++){
            particles[i].w = 1.0/NUM_PARTICLES;
        }        
    }
}

float compute_weight(Particle particle, float z[STATE_SIZE], float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float Sf[2][2] =  {{0,0}, {0,0}};
    float zp[2]; //treat zp as a column vector
    float dx[2]; //treat dx as a column vector

    float sChol[2][2] = { 0 };  
    float result[2] = {0.0 ,0.0};
    float dotproduct = 0.0; 
    float num = 0.0; 
    float den = 0.0; 
    float w = 0.0;

    // landmark location 
    xf[0] = particle.lm[lm_id][0];
    xf[1] = particle.lm[lm_id][1];

    //pF holds the landmark covariance
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            pf[i][j] = particle.lm_cov[2*lm_id + i][j];
        }
    }
    // Q = H covariance matrix H^-1 + Q_patrix --> H is covariance of our innovation function h = zactual - zpredicts
    compute_jacobians(particle, xf, pf, Q_mat, Hf, Hv, Sf, zp);

    // zp holds the predicted distancfe from the particle to the landmark
    dx[0] = z[0] - zp[0];
    dx[1] = pi_2_pi(z[1] - zp[1]); 

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            sChol[i][j] = Sf[i][j];
        }
        result[i] = dx[i];
    }   
    
    // solve Ly = b where y = z 
    cholesky_decomp(sChol, result, 2);

    // matrix_vector(sChol, dx, result);
    // this dot product can now be computed by taking result and squaring it
    // dotproduct = dot_product(dx, result);
    dotproduct = result[0]*result[0] + result[1]*result[1];
    num = (float)exp(-0.5 * dotproduct);
    den = (float)sqrt(2.0 * pi * det(Sf));
    
    return (num/den);
}

float dot_product(float (&row_vec)[2], float (&col_vec)[2]){
    float sum = 0.0; 
    for (uint8_t i = 0; i < 2; i++){
        sum += row_vec[i]*col_vec[i];
    }
    return sum; 
}

float det(float (&matrix)[2][2]){
    return (matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]);
}

void compute_jacobians(Particle particle, float (&xf)[2], float (&pf)[2][2], float (&Q_mat)[2][2], float (Hf)[2][2], float (&Hv)[2][3], float (&Sf)[2][2], float (&zp)[2]){
    float mult_vals[2][2] = { 0 }; 
    float vals[2][2] = { 0 };
    float pf_mat[2][2] = { 0 };

    float dx = xf[0] - particle.x; 
    float dy = xf[1] - particle.y;
    float d2 = (float)(pow(dx, 2) + pow(dy, 2));
    float d = (float)sqrt(d2);
    // zp represents our observation function
    zp[0] = d;
    zp[1] = pi_2_pi(atan2(dy, dx) - particle.yaw);

    // Jacobian of our observation function
    Hv[0][0] = -dx/d; 
    Hv[0][1] = -dy/d; 
    Hv[0][2] = 0.0;
    Hv[1][0] = dy/d2; 
    Hv[1][1] = -dx/d2;
    Hv[1][2] = -1.0; 

    Hf[0][0] = dx/d; 
    Hf[0][1] = dy/d;
    Hf[1][0] = -dy/d2; 
    Hf[1][1] = dx/d2; 

    // shallow copy Hv into mult_vals
    // pf holds the landmark's EKF covariance 
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            mult_vals[i][j] = Hf[i][j];
            vals[i][j] = Hf[i][j];
            pf_mat[i][j] = pf[i][j];
        }
    }

    // put hf^T into mult_vals
    transpose_mat(mult_vals);
    // cov*H^T
    mult_mat(pf_mat, mult_vals);
    // H*cov*H^T
    mult_mat(vals, mult_vals);
    
    // Sf holds Q =H*cov*H^T + Qt
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            Sf[i][j] = mult_vals[i][j] + Q_mat[i][j];
        }
    }
}

component void mult_mat(float matrix1[2][2], float matrix2[2][2]){
    float inter[2][2]; 
    uint8_t i = 0;
    uint8_t j = 0;
    uint8_t k = 0;
    for (i = 0; i < 2; i++){
        for (j = 0; j < 2; j++){
            inter[i][j] = matrix2[i][j];
        }
    }

    for (i = 0; i < 2; i++){
        for (j = 0; j < 2; j++){
            matrix2[i][j] = 0;
            for (k = 0; k < 2; k++){
                matrix2[i][j] += matrix1[i][k]*inter[k][j];
            }
        }
    }
}

void transpose_mat(float matrix[2][2]){
    float b = matrix[0][1];
    float c = matrix[1][0];
    matrix[0][1] = c; 
    matrix[1][0] = b;  
}

void motion_model(float states[STATE_SIZE], float control[2]){
    float f[3][3] = {{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}}; // --> identity matrix to simply extract each part of the state vector
    float B[3][2] = { 0 };
    B[0][0] = (float)TICK*cos(states[2]);
    B[0][1] = 0;
    B[1][0] = (float)TICK*sin(states[2]); 
    B[1][1] = 0; 
    B[2][0] = 0; 
    B[2][1] =  (float)TICK;
    float int_1[3] = {0.0, 0.0, 0.0};
    float int_2[3] = {0.0, 0.0, 0.0};
    float sum = 0.0;
    // compute F*X
    for (int i = 0; i < 3; i++){
        sum = 0.0; 
        for (int j = 0; j < 3; j++){
            sum+= f[i][j]*states[j];
        }
        // printf("i: %d, sum = %f\n", i, sum);
        int_1[i] = sum;
    }

    // printf("States: %f, %f, %f \n", states[0], states[1], states[2]);

    // compute B*U
    for (int i = 0; i < 3; i++){
        sum = 0.0; 
        for (int j = 0; j < 2; j++){
            sum+= B[i][j]*control[j];
        }
        int_2[i] = sum;
    }

    for (int i = 0; i < 3; i ++){
        states[i] = int_1[i] + int_2[i];
    }

    // here x = x + 

    states[2] = pi_2_pi(states[2]); 
    // return states;
}

float pi_2_pi(float value){
    // if(inside == 1){
    //     printf("pi: %f; value: %f, sum: %f\n", pi, value, value + pi);
    // } 
    float new_value = fmod(value + pi, 2*pi) - pi;
    // if(inside == 1){
    //     printf("Mapped value: %f\n", new_value);
    // }
    // if (new_value < -pi) {
    //     new_value += 2 * pi;
    // } else if (new_value > pi) {
    //     new_value -= 2 * pi;
    // }
    // if(inside == 1){
    //     printf("Mapped value 2: %f\n", new_value);
    // }
    return new_value; 
}

// takes a 2 input float as input
void calc_input(float time, float u[2]){
    if(time <= 2.98){
        u[0] = 0.0; 
        u[1] = 0.0;
    }
    else{
        u[0] = 1.0; 
        u[1] = 0.1; 
    }
    // else if(time > 2.98 && time <= 15.0){
    //     u[0] = 1.0; 
    //     u[1] = 0.1; 
    // }
    // else if(time > 15.0 && time <= 30.0){
    //     u[0] = 0.7; 
    //     u[1] = 0.1;
    // }
    // else{
    //     u[0] = 0.5; 
    //     u[1] = 0.05;
    // }
}

void calc_final_state(Particle* particles, float xEst[3]){
    xEst[0] = 0;
    xEst[1] = 0; 
    xEst[2] = 0; 

    normalize_weight(particles);

    int i; 

    for(i = 0; i  < NUM_PARTICLES; i++){
        xEst[0] += particles[i].w*particles[i].x; 
        xEst[1] += particles[i].w*particles[i].y; 
        xEst[2] += particles[i].w*particles[i].yaw; 
    }
    xEst[2] = pi_2_pi(xEst[2]);
    printf("Xest is: %f, %f, %f\n", xEst[0], xEst[1], xEst[2]);
}

//outside the scope of the main fasst SLAM --> mainly used to advance the simulation
void observation(float xTrue[3], float xd[3], float u[2], float rfid[num_landmarks][2], uint8_t num_id, float ud[2], int& num_cols, float z[3][num_landmarks]){
    random_device rd; 
    mt19937 gen(rd()); 
    // uniform_real_distribution<float> distribution(0, 1.0); 
    printf("time: %f, xTrue (states obs) is: %f, %f, %f \n", time_step, xTrue[0], xTrue[1], xTrue[2]);

    motion_model(xTrue, u); // return the true trajectory from the motion model (no noise)
    // printf("time: %f, xTrue is: %f, %f, %f \n", time_step, xTrue[0], xTrue[1], xTrue[2]);
    vector<vector<float>> z_new = {{0}, {0}, {0}}; 
    float zi[3] = {};
    float dx, dy, d, angle, dn, angle_noisy; 
    uint8_t i, j; 
    int position = 0; 

    for(i = 0; i < num_id; i++){
        dx = rfid[i][0] - xTrue[0];
        // printf("i is: %d, RFID[0]: %f\n", i, rfid[i][0]);
        dy = rfid[i][1] - xTrue[1];
        // printf("i is: %d, RFID[1]: %f\n", i, rfid[i][1]);
        d = (float)hypot(dx, dy); 
        // printf("Time is: %f; i is: %d, d is: %f\n", time_step, i, d);
        float val = atan2(dy, dx) - xTrue[2];
        angle = (float)pi_2_pi(val);
        // printf("Time is: %f; i is: %d, Angle is: %f\n", time_step, i, angle);
        // add only the landmarks we can see (ie., within a given range of)
        if (d <= MAX_RANGE){
            // add noise to the measured distance to the landmark
            normal_distribution<float> distribution(0, 0.1);
            dn = d + distribution(gen)*pow(Q_sim[0][0], 0.5);  //gaussian noise added to our measurement
            // printf("Time is: %f; i is: %d, dn is: %f\n", time_step, i, dn);
            normal_distribution<float> distribution2(0, 0.1); 
            angle_noisy = angle + distribution2(gen)*pow(Q_sim[1][1], 0.5); // gaussian noise added to our measurement 
            // angle_noisy = angle + distribution(gen)*pow(Q_sim[1][1], 0.5);
            // printf("Time is: %f; i is: %d, Angle noisy is: %f\n", time_step, i, angle_noisy);

            // store the measurement and perform the known data association by including the landmark ID
            zi[0] = dn; 
            // inside = 1;
            zi[1] = pi_2_pi(angle_noisy);
            // inside = 0; 
            // printf("Time is: %f; i is: %d, Pi 2 Pi is: %f\n", time_step, i, zi[1]);
            zi[2] = i;

            for(int j = 0; j < 3; j++){
                z[j][position] = zi[j];
            }
            position++;
        }
    }

    num_cols = position; 
    // Add noise to the control input to advance the simulation 
    normal_distribution<float> distribution_cntr(0, 0.25);
    ud[0] = u[0] + distribution_cntr(gen)*pow(R_sim[0][0], 0.5);
    normal_distribution<float> distribution_cntr2(0, 0.25);
    ud[1] = u[1] + distribution_cntr2(gen)*pow(R_sim[1][1], 0.5) + OFFSET_YAW_RATE_NOISE; 

    printf("Time is: %f; Xd (states obs2 is: %f, %f, %f\n", time_step, xd[0], xd[1], xd[2]);

    motion_model(xd, ud);

    // printf("time: %f, xD is: %f, %f, %f, ud is: %f, %f \n", time_step, xd[0], xd[1], xd[2], ud[0], ud[1]);
}

int main(){
    
    printf("We starting FastSLAM execution now!\n");

    float RFID[num_landmarks][2] = {{10, -2}, {15, 10}, {15, 15}, {0, 20}, {3, 15}, {-5, 20}, {-5, 5}, {-10, 15}};

    float z[3][num_landmarks] = { 0 };
    
    float xEst[3] = {};
    float x_state[3] = {};
    float xTrue[3] = {};
    float xDR[3] = {};

    float u[2] =  {}; 
    float ud[2] = {}; 
        int num_columns; 


    Particle particles[TOTAL_NUM_PARTICLES] = {}; // working right
    for (int i = 0; i < TOTAL_NUM_PARTICLES; i++){
        particles[i] = Particle(num_landmarks);
    }


    // create file to store the particle data 
    ofstream outputFile("particleData.csv");
    ofstream outputFile2("historyData.csv");
    ofstream outputFile3("Landmark_coords.csv");

    // making the format for the file
    outputFile << "Time" << "," << "Particle" << "," << "Particle x" << "," << "Particle y" << "," << "Landmark 1 x" << "," << "Landmark 1 y" << "," << "Landmark 2 x" << "," << "Landmark 2 y" << "," << "Landmark 3 x" << "," << "Landmark 3 y" << "," << "Landmark 4 x" << "," << "Landmark 4 y"<< "," << "Landmark 5 x" << "," << "Landmark 5 y" << "," << "Landmark 6 x" << "," << "Landmark 6 y" << "," << "Landmark 7 x" << "," << "Landmark 7 y" << "," << "Landmark 8 x" << "," << "Landmark 8 y" << endl;   
    outputFile2 << "Time" << "," << "hxTrue x" << "," << "hxTrue y" << "," << "hxDr x" << "," << "hxDR y" << "," << "hxEst x" << "," << "hxEst y" << endl;
    outputFile3 << "Landmark x" << "," << "Landmark y" << endl;
    
    for(int i = 0; i < num_landmarks; i++){
        for (int j = 0; j < 2; j++){
            // 0, 0 =0; 0, 1 = 1; 1, 0= 2; 1, 1 = 3; 2, 0 = 4
            if(j < 1) outputFile3 << RFID[i][j] << ",";
            else outputFile3 << RFID[i][j];
        }
        outputFile3 << endl;
    }

    while(SIM_TICK>= time_step){
        time_step += TICK; 
        check = time_step; 
        
        calc_input(time_step, u);
        observation(xTrue, xDR, u, RFID, num_landmarks, ud, num_columns, z); 

        fast_slam1(particles, ud, z, num_columns); 
        // printf("Particle 0 x coordinate: %f; y coordinate: %f \n", particles[0].x, particles[0].y);
        calc_final_state(particles, xEst);  
        
        // populates the text file holding all the particles values through all the time steps 
        if(outputFile.is_open()){
            int i = 0;
            for(i = 0; i < TOTAL_NUM_PARTICLES; i++){
                if(i < NUM_PARTICLES){
                    outputFile << time_step << "," << i << "," << particles[i].x << "," << particles[i].y; 
                    for (int j = 0; j < 8; j++){
                        for (int k = 0; k < 2; k++){
                            outputFile << "," << particles[i].lm[j][k]; 
                        }
                    }
                    outputFile << endl; 
                }
            }
        }
        if(outputFile2.is_open()){
            outputFile2 << time_step << "," << xTrue[0] << "," << xTrue[1] << "," << xDR[0] << "," << xDR[1] << "," << xEst[0] << "," << xEst[1] << endl; 
        }


    }
    outputFile.close();
    outputFile2.close();
    outputFile3.close(); 

    printf("made that shit\n");  
    return 1; 
}