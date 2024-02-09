#include "system.h"
#include <string>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <queue>
#include <vector>
#include <cstdlib>
#include <random>
#include <iostream>

//stuff 
using namespace std;

Particle::Particle(int num_landmarks)
{
    uint8_t i;
    uint16_t j;
    w = 1.0/num_landmarks; // each particle get uniform weight
    //each particle starts at point 0,0
    x = 0.0; 
    y = 0.0; 
    yaw = 0.0; 
    lm = new float*[num_landmarks];
    lm_cov = new float*[num_landmarks*LM_SIZE];
    P = new float*[3];

    //init lm and lm_cov to be all zeroes
    for (i = 0; i < num_landmarks; i++){
        lm[i] = new float[LM_SIZE];
    }

    for (j = 0; j < num_landmarks*LM_SIZE; j++){
        lm_cov[j] = new float[LM_SIZE];
    }
    for (i = 0; i < 3; i++){
        P[i] = new float[3];
    }

    for(i = 0; i < num_landmarks; i ++){
        for (j = 0; j < LM_SIZE; j++){
            lm[i][j] = 0.0; 
        }
    }

    for(i = 0; i < num_landmarks*LM_SIZE; i ++){
        for (j = 0; j < LM_SIZE; j++){
            lm_cov[i][j] = 0.0; 
        }
    }

    P[0][0] = 1; 
    P[1][1] = 1; 
    P[2][2] = 1; 
}

Particle* fast_slam1(Particle* particles, float* control, float** z, int num_cols){
    particles = predict_particles(particles, control);

    particles = update_with_observation(particles, z, num_cols);
  
    particles = resampling(particles);
   
    return particles; 
}

// control is a 2d vector (v, w)
Particle* predict_particles(Particle* particles, float* control){
    float* px;
    float state[STATE_SIZE];
    px = state;
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

        //Returns the particle's estimated position using the noise motion command and it's initial known state vector
        px = motion_model(px, prod); // pxrod= u + noise*r matrix --> add gaussian noise to the command accounting for our coveriance
        
        particles[i].x = px[0];
        particles[i].y = px[1];
        particles[i].yaw = px[2];

    }
    return particles; 
}

Particle* update_with_observation(Particle* particles, float** z, int num_cols){     
    uint8_t landmark_id = 0;
    uint8_t i;
    uint8_t j;
    float* obs = new float[3];
    float weight;
    for (i = 0; i < num_cols; i++){
        //cout << (int)i << endl; 
        landmark_id = (uint8_t)z[2][i]; // this assumes that the data association problem is known
        //cout << "Landmark ID" << (int)landmark_id << endl; 
        obs[0] = z[0][i]; 
        obs[1] = z[1][i];
        obs[2] = z[2][i];        
        for (j = 0; j < NUM_PARTICLES; j++){
            if (abs(particles[j].lm[landmark_id][0]) <= 0.01){
                particles[j] = add_new_landmark(particles[j], obs, Q_matrix);  
            }
            else{             
                weight = compute_weight(particles[j], obs, Q_matrix);               
                particles[j].w *= weight; 
                //cout << "Weight: " << weight << endl;
                particles[j] = update_landmark(particles[j], obs, Q_matrix);
                particles[j] = proposal_sampling(particles[j], obs, Q_matrix);
            }
        }
    }

    return particles;
}  

Particle proposal_sampling(Particle particle, float*z, float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float HvT[3][2] = {{0, 0}, {0, 0}, {0, 0}};
    float mult[2][3] = {{0, 0, 0}, {0, 0, 0}};
    float mult_3[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    float Sf[2][2] =  {{0,0}, {0,0}};
    float** sInv = new float*[2];
    float** P = new float*[3]; 
    for (int i = 0; i < 3; i++){
        if(i < 2) sInv[i] = new float[4];
        P[i] = new float[6]; 
    }

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

    //S*HV

    for (int i = 0; i < 2; i++){
        for (int j = 0; j < 3; j++){
            mult[i][j] = 0;
            for (int k = 0; k < 2; k++){
                mult[i][j] += sInv[i][k + 2]*Hv[k][j];
            }
        }
    }

    for (int i = 0; i < 3; i++){
        for (int j = 0; j < 3; j++){
            mult_3[i][j] = 0;
            for (int k = 0; k < 2; k++){
                mult_3[i][j] += HvT[i][k]*mult[k][j];
            }
        }
    }

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

    matrix_vector(sInv, dz, res);
    matrix_vector_3(HvT, res, res_2);

    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            P[i][j] = particle.P[i][j];
        }
    }

    matrix_vector_33(P, res_2, result);

    particle.x += result[0]; 
    particle.y += result[1]; 
    particle.yaw += result[2];

    return particle;
}

void matrix_vector_3(float (&matrix)[3][2], float (&vector)[2], float (&result)[3]){
    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 2; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void matrix_vector_33(float** matrix, float (&vector)[3], float (&result)[3]){
    for (uint8_t i = 0; i < 3; i++){
        for (uint8_t j = 0; j < 3; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void matrix_vector(float** matrix, float (&vector)[2], float (&result)[2]){
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void inverse(float** matrix, int n){
    float temp_val; 

    for (int i = 0; i < n; i++){
        for (int j = 0; j <2 * n; j++){
            if(j == (i + n))
                matrix[i][j] = 1; 
        }
    }

    // perform row swapping
    for (int i = n - 1; i > 0; i--){
        if(matrix[i - 1][0] < matrix[i][0]){
            float* temp_val = matrix[i];
            matrix[i] = matrix[i - 1];
            matrix[i - 1] = temp_val;
        }
    }

    //replace row by sum of itself and constant multiple of another row
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(j != i){
                temp_val = matrix[j][i] / matrix[i][i];
                for (int k = 0; k < 2 * n; k++){
                    matrix[j][k] = matrix[j][k] - matrix[i][k] * temp_val;
                }
            }
        }
    }

    for (int i = 0; i < n; i++){
        temp_val = matrix[i][i]; 
        for (int j = 0; j < 2 * n; j++){
            matrix[i][j] = matrix[i][j] / temp_val;
        }
    }

    return;
}

Particle update_landmark(Particle particle, float *z, float (&Q_mat)[2][2]){
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

    return particle;
}


/*
xf holds the landmark locations to compute the measurement prediction 
dz holds the measurement innovation 
pf holds the landmark covariance
q_mat is the user defined Q matrix (measure of how much we believe the measurements)
Hf is the jacobian 
*/
void update_kf_with_cholesky(float (&xf)[2], float (&pf)[2][2], float (&dz)[2], float (&Q_mat)[2][2], float (&Hf)[2][2]){
    float** HfT = new float*[2];
    float** H = new float*[2];  
    float** PHt = new float*[2]; 
    float** S = new float*[2]; 
    float** ST = new float*[2];

    float** L_matrix = new float*[2];
    float** L_trans = new float*[2];
    for (int i = 0; i < 2; i++){
        L_matrix[i] = new float[2];
        L_trans[i] = new float[2];
        HfT[i] = new float[2]; 
        PHt[i] = new float[2]; 
        S[i] = new float[4]; 
        ST[i] = new float[2];
        H[i] = new float[2]; 
    }

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
    mult_mat(PHt, HfT,2);
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            PHt[i][j] = HfT[i][j];
        }
    }
    
    mult_mat(H, HfT,2);
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            ST[i][j] = HfT[i][j] + Q_mat[i][j];
            S[i][j] = ST[i][j];
        }
    }

    //S = H*cov*H^T + Q
    // compute Q^T
    transpose_mat(ST);


    // make s symmetric
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            S[i][j] += ST[i][j];
            S[i][j] *= 0.5;
        }
    }

    cholesky_decomp(S, vect, 2);

    transpose_mat(S);


    inverse(S, 2);

    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            L_trans[i][j] = S[i][j + 2];

        }
    }

    transpose_mat(L_trans);


    mult_mat(PHt, L_matrix,2); //W1
    mult_mat(L_matrix, L_trans,2); //W


    // x = W*V
    matrix_vector(L_trans, dz, x);

    // copy W1 into L_mattrans
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            L_trans[i][j] = L_matrix[i][j];
        }
    }

    transpose_mat(L_trans); //W1.T

    mult_mat(L_matrix, L_trans,2); // W1 * W1.T

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

bool cholesky_decomp(float** S, float* vector_b, int n){
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            S[i][j] = 0; 
        }
    }

    // forward substitution solving Ly = b 
    for(int j = 0; j < n; j ++){
        if(S[j][j] <= 0){
            cout << "Matrix is not positive definite" << endl; 
            return false;
        }
        S[j][j] = (float)sqrt(S[j][j]);
        vector_b[j] = vector_b[j] / S[j][j]; 
        for(int i = j + 1; i < n; i++){
            if(i < n){
                S[i][j] = S[i][j] / S[j][j]; 
                vector_b[i] = vector_b[i] - S[i][j]*vector_b[j];
                for(int k = j + 1; k < i + 1; k++){
                    S[i][k] = S[i][k] - (S[i][j] * S[k][j]);
                }
            }
            else break;
        }
    }
    
    return true;
}

Particle add_new_landmark(Particle particle, float *z, float (&Q_mat)[2][2]){
    float r = z[0]; 
    float b = z[1]; 
    uint8_t lm_id = (uint8_t)z[2]; 
    float** gz; 
    float** sChol; 
    float** sChol_trans;
    float ** Q; 
    float s = sin(pi_2_pi(particle.yaw + b));
    float c = cos(pi_2_pi(particle.yaw + b));
    float* vect_b = new float[2]; 
    float I[2][2] = {{1, 0}, {0, 1}};

    particle.lm[lm_id][0] = particle.x + r*c; 
    particle.lm[lm_id][1] = particle.y + r*s; 
    
    // adding the landmark covariance

    float dx = r * c; 
    float dy = r * s; 
    float d2 = (float)(pow(dx, 2) + pow(dy, 2));
    float d = (float)(sqrt(d2));


    Q = new float*[2];
    gz = new float*[2]; 
    sChol = new float*[2]; 
    sChol_trans = new float*[2];

    for (int i = 0; i < 2; i++){
        gz[i] = new float[2]; 
        sChol[i] = new float[2]; 
        sChol_trans[i] = new float[2];
        Q[i] = new float[2];
    }

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
    mult_mat(sChol_trans, sChol, 2);

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
    return particle; 
}
Particle* resampling(Particle* particles){
    float pw[NUM_PARTICLES];
    float n_eff = 0.0;
    float* w_cumulative = new float[NUM_PARTICLES];
    float* base = new float[NUM_PARTICLES];
    float* resample_id= new float[NUM_PARTICLES];
    vector<float> weights;
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<float> distribution(-1.0, 1.0); 
    uint8_t indices[NUM_PARTICLES]; 
    Particle tmp_particles[NUM_PARTICLES];

    particles = normalize_weight(particles);
    

    uint16_t i = 0;
    for (i = 0; i < NUM_PARTICLES; i++){
        pw[i] = particles[i].w;
    }

    float dp = vector_vector(pw, pw, NUM_PARTICLES);
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
    return particles;
}

void cumulative_sum(float* array, float* sum){
    sum[0] = array[0];

    for (uint16_t i = 1; i < NUM_PARTICLES; i++){
        sum[i] = sum[i-1] + array[i];
    }

}

float vector_vector(float* row_vec, float* col_vec, int num_rows){
    float sum = 0.0;
    for (int i = 0; i < num_rows; i++){
        sum += row_vec[i]*col_vec[i];
    }
    return sum; 
}

Particle* normalize_weight(Particle* particles){
    float sum_weights = 0.0; 
    for (uint16_t i = 0; i < NUM_PARTICLES; i++){
        sum_weights+=particles[i].w;
    }

    if (sum_weights != 0.0){
        for (uint16_t i = 0; i < NUM_PARTICLES; i++){
            particles[i].w /= sum_weights;
        }
    }
    else{
        for (uint16_t i = 0; i < NUM_PARTICLES; i++){
            particles[i].w = 1.0/NUM_PARTICLES;
        }        
    }

    return particles;
}

float compute_weight(Particle particle, float* z, float (&Q_mat)[2][2]){
    uint8_t lm_id = (uint8_t)z[2];
    float xf[2] = {0.0, 0.0}; // column vector
    float pf[2][2] =  {{0,0}, {0,0}};
    float Hv[2][3] =  {{0, 0, 0}, {0, 0, 0}};
    float Hf[2][2] =  {{0,0}, {0,0}};
    float Sf[2][2] =  {{0,0}, {0,0}};
    float zp[2]; //treat zp as a column vector
    float dx[2]; //treat dx as a column vector

    float** sChol = new float*[2];  
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
        sChol[i] = new float[2]; 
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
    den = (float)sqrt(2.0 * M_PI * det(Sf));
    
    return (num/den);
}

void matrix_vector(float (&matrix)[2][2], float (&vector)[2], float (&result)[2]){
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            result[i] += matrix[i][j] * vector[j];
        }
    }
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
    float** mult_vals = new float*[2]; 
    float** vals = new float*[2];
    float** pf_mat = new float*[2];

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

    for (uint8_t i = 0; i < 2; i++){
        mult_vals[i] = new float[2]; 
        vals[i] = new float[2]; 
        pf_mat[i] = new float[2];
    }

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
    mult_mat(pf_mat, mult_vals, 2);
    // H*cov*H^T
    mult_mat(vals, mult_vals, 2);
    
    // Sf holds Q =H*cov*H^T + Qt
    for (uint8_t i = 0; i < 2; i++){
        for (uint8_t j = 0; j < 2; j++){
            Sf[i][j] = mult_vals[i][j] + Q_mat[i][j];
        }
    }
}

void mult_mat(float** matrix1, float** matrix2, int n){
    float inter[n][n]; 
    uint8_t i = 0;
    uint8_t j = 0;
    uint8_t k = 0;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            inter[i][j] = matrix2[i][j];
        }
    }

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            matrix2[i][j] = 0;
            for (k = 0; k < n; k++){
                matrix2[i][j] += matrix1[i][k]*inter[k][j];
            }
        }
    }
}

void transpose_mat(float** matrix){
    float b = matrix[0][1];
    float c = matrix[1][0];
    matrix[0][1] = c; 
    matrix[1][0] = b;  
}

float* motion_model(float* states, float* control){
    float f[3][3] = {{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}}; // --> identity matrix to simply extract each part of the state vector
    float B[3][2] = {{TICK*cos(states[2]), 0}, {TICK*sin(states[2]), 0}, {0.0, TICK}}; //
    float int_1[3] = {0.0, 0.0, 0.0};
    float int_2[3] = {0.0, 0.0, 0.0};
    float sum = 0.0;

    // compute F*X
    for (int i = 0; i < 3; i++){
        sum = 0.0; 
        for (int j = 0; j < 3; j++){
            sum+= f[i][j]*states[j];
        }
        int_1[i] = sum;
    }
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
    return states;
}

float pi_2_pi(float value){ 
    return fmod(value + M_PI, 2*M_PI) - M_PI;
}

void calc_input(float time, float* u){
    if(time <= 2.98){
        u[0] = 0.0; 
        u[1] = 0.0;
    }
    // else{
    //     u[0] = 1.0; 
    //     u[1] = 0.1; 
    // }
    else if(time > 2.98 && time <= 15.0){
        u[0] = 1.0; 
        u[1] = 0.1; 
    }
    else if(time > 15.0 && time <= 30.0){
        u[0] = 0.7; 
        u[1] = 0.1;
    }
    else{
        u[0] = 0.5; 
        u[1] = 0.05;
    }
}

void calc_final_state(Particle* particles, float* xEst){
    xEst[0] = 0;
    xEst[1] = 0; 
    xEst[2] = 0; 

    particles = normalize_weight(particles);

    uint16_t i; 

    for(i = 0; i  < NUM_PARTICLES; i++){
        xEst[0] += particles[i].w*particles[i].x; 
        xEst[1] += particles[i].w*particles[i].y; 
        xEst[2] += particles[i].w*particles[i].yaw; 
    }
    xEst[2] = pi_2_pi(xEst[2]);
}

//outside the scope of the main fasst SLAM --> mainly used to advance the simulation
float** observation(float* xTrue, float* xd, float* u, float** rfid, uint8_t num_id, float* ud, int& num_cols){
    random_device rd; 
    mt19937 gen(rd()); 
    // uniform_real_distribution<float> distribution(0, 1.0); 

    xTrue = motion_model(xTrue, u); // return the true trajectory from the motion model (no noise)
    vector<vector<float>> z_new = {{0}, {0}, {0}}; 
    float* zi = new float[3];
    float dx, dy, d, angle, dn, angle_noisy;
    float ** z = new float*[3]; 
    uint8_t i, j; 
    int position = 0; 

    for(i = 0; i < num_id; i++){
        dx = rfid[i][0] - xTrue[0];
        dy = rfid[i][1] - xTrue[1];
        d = (float)hypot(dx, dy); 
        float val = atan2(dy, dx) - xTrue[2];
        angle = (float)pi_2_pi(val);
        // add only the landmarks we can see (ie., within a given range of)
        if (d <= MAX_RANGE){
            // add noise to the measured distance to the landmark
            normal_distribution<float> distribution(0, 0.1);
            dn = d + distribution(gen)*pow(Q_sim[0][0], 0.5);  //gaussian noise added to our measurement
            normal_distribution<float> distribution2(0, 0.1); 
            angle_noisy = angle + distribution2(gen)*pow(Q_sim[1][1], 0.5); // gaussian noise added to our measurement 
            // angle_noisy = angle + distribution(gen)*pow(Q_sim[1][1], 0.5);

            // store the measurement and perform the known data association by including the landmark ID
            zi[0] = dn; 
            zi[1] = pi_2_pi(angle_noisy);
            zi[2] = i;
            if (position == 0){
                z_new[0][0] = zi[0];
                z_new[1][0] = zi[1];    
                z_new[2][0] = zi[2];
            }
            else{
                for(j = 0; j < 3; j++){
                    z_new[j].insert(z_new[j].begin() + position, zi[j]);
                }
            }
            position++;
        }
    }
    num_cols = z_new[0].size();

    for (i = 0; i < 3; i++){
        z[i] = new float[num_cols];
    }

    for(i = 0; i < 3; i++){
        for(j = 0; j < num_cols; j++){
            z[i][j] = z_new[i][j];
        }
    }

    // Add noise to the control input to advance the simulation 
    normal_distribution<float> distribution_cntr(0, 0.25);
    ud[0] = u[0] + distribution_cntr(gen)*pow(R_sim[0][0], 0.5);
    normal_distribution<float> distribution_cntr2(0, 0.25);
    ud[1] = u[1] + distribution_cntr2(gen)*pow(R_sim[1][1], 0.5) + OFFSET_YAW_RATE_NOISE; 


    xd = motion_model(xd, ud);

    return z; 
}

int main(){
    float ave_min = 0; 
    float ave_max = 0; 

    
    std::cout << "We starting FastSLAM execution now!" << endl; 

    float** RFID = new float*[8]; 
    for (int i = 0; i < 8; i++){
        RFID[i] = new float[2]; 
    }
    
    RFID[0][0] = 10.0; 
    RFID[0][1] = -2.0; 
    RFID[1][0] = 15.0; 
    RFID[1][1] = 10.0; 
    RFID[2][0] = 15.0; 
    RFID[2][1] = 15.0; 
    RFID[3][0] = 0.0; 
    RFID[3][1] = 20.0; 
    RFID[4][0] = 3.0; 
    RFID[4][1] = 15.0; 
    RFID[5][0] = -5.0; 
    RFID[5][1] = 20.0; 
    RFID[6][0] = -5.0; 
    RFID[6][1] = 5.0; 
    RFID[7][0] = -10.0; 
    RFID[7][1] = 15.0; 

    num_landmarks = 8; 
    ave_min = 0; 
    ave_max = 0; 

    for(int l = 0 ; l < 1; l++){

        float* xEst = new float[3];
        float* x_state = new float[3];
        float* xTrue = new float[3];
        float* xDR = new float[3];

        float* hxEst = xEst; 
        float* hxTrue = xTrue; 
        float* hxDR = xDR; 
        float time = 0;
        float* u = new float[2]; 
        float* ud = new float[2]; 
        float** z; 
        int num_columns; 


        Particle* particles = new Particle[TOTAL_NUM_PARTICLES]; // working right
        for (int i = 0; i < TOTAL_NUM_PARTICLES; i++){
            particles[i] = Particle(num_landmarks);
        }

        hxEst = xEst; 
        hxDR = xDR; 
        hxTrue = xTrue;

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
                if(j < 1) outputFile3 << RFID[i][j] << ",";
                else outputFile3 << RFID[i][j];
            }
            outputFile3 << endl;
        }

        while(SIM_TICK>= time){
            time += TICK; 
            check = time; 
            
            calc_input(time, u);
            z = observation(xTrue, xDR, u, RFID, num_landmarks, ud, num_columns); 

            particles = fast_slam1(particles, ud, z, num_columns); 

            calc_final_state(particles, xEst);  
            
            calc_final_state(particles, xEst);


            // populates the text file holding all the particles values through all the time steps 
            if(outputFile.is_open()){
                int i = 0;
                for(i = 0; i < TOTAL_NUM_PARTICLES; i++){
                    if(i < NUM_PARTICLES){
                        outputFile << time << "," << i << "," << particles[i].x << "," << particles[i].y; 
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
                outputFile2 << time << "," << xTrue[0] << "," << xTrue[1] << "," << xDR[0] << "," << xDR[1] << "," << xEst[0] << "," << xEst[1] << endl; 
            }


        }
        outputFile.close();
        outputFile2.close();
        outputFile3.close(); 
    // ave_max += max_value; 
    // ave_min += min_value;

    }

    //cout << particles[24].x << " " << particles[24].y << endl;
    // outputFile.close();
    // outputFile2.close();
    // outputFile3.close(); 
    // min_value = ave_min / 25; 
    // max_value = ave_max / 25; 

    // std::cout << "Min value: " << " " << min_value << endl; 
    // std::cout << "Max value: " << " " << max_value << endl; 
    std::cout << "made that shit" << endl;  
    return 1; 

    // int n = 2; 

    // float** matrix = new float*[n]; 

    // for (int i = 0; i < n; i++){
    //     matrix[i] = new float[n];
    // }

    // matrix[0][0] = 74.20;
    // matrix[0][1] = 88.01;
    // // matrix[0][2] =66.93;
    // matrix[1][0] = 68.30;
    // matrix[1][1] = 26.39;
    // // matrix[1][2] = 63.48;
    // // matrix[2][0] = 10.25;
    // // matrix[2][1] = 40.93;
    // // matrix[2][2] = 18.08;

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         cout << matrix[i][j] << "  ";
    //     }
    //     cout << endl;
    // }

    // inverse(matrix, n);

    // cout << std::fixed << std::setprecision(3);

    // for (int i = 0; i < n; i++) {
    //     for (int j = n; j < 2 * n; j++) {
    //         cout << matrix[i][j] << "  ";
    //     }
    //     cout << endl; 
    // }


    // float** test_matrix = new float*[2];
    // float* vector_b = new float[2]; 

    // float** test_matrix_3 = new float*[3]; 

    // for (int i = 0; i < 3; i++){
    //     if(i < 3) test_matrix[i] = new float[2];
    //     test_matrix_3[i] = new float[3]; 
    // }

    // test_matrix_3[0][0] = 1; 
    // test_matrix_3[0][1] = 1.2; 
    // test_matrix_3[0][2] = 4; 
    // test_matrix_3[1][0] = -3.15; 
    // test_matrix_3[1][1] = 1; 
    // test_matrix_3[1][2] = 2;  
    // test_matrix_3[2][0] = 4.25; 
    // test_matrix_3[2][1] = 2.3; 
    // test_matrix_3[2][2] = 35;  

    // test_matrix[0][0] = 17.62545; 
    // test_matrix[0][1] = 0.17492; 
    // test_matrix[1][0] = 0.17492; 
    // test_matrix[1][1] = 0.03431;   

    // vector_b[0] = 1; 
    // vector_b[1] = 7;     

    // inverse(test_matrix_3, 3); 

    // cholesky_decomp(test_matrix, vector_b, 2);

}