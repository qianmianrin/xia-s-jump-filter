#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M_PI 3.14159265358979323846

// ================== 网格与物理参数 ==================
#define Nx 400        
#define xL 0.0      
#define xR 1.0        
#define gamma_gas 1.4 
#define T_END 0.13
#define CFL 0.1
#define dx ((xR - xL) / (double)Nx)

// ================== 1D P^2 DG 宏定义 ==================
#define NUM_VARS 3       
#define N_MODE 3         
#define N_QUAD 3         
#define N_DERIV 3        
#define N_ZS_PTS 3       

typedef struct {
    double U[NUM_VARS][N_MODE]; 
    double face_left[NUM_VARS];
    double face_right[NUM_VARS];
} Element;

Element Mesh_h[Nx];

// 3点高斯积分节点与权重
static const double nodes_G_h[N_QUAD]   = {-0.7745966692414834, 0.0, 0.7745966692414834};
static const double weights_G_h[N_QUAD] = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};

// 3点高斯-洛巴托积分节点
static const double nodes_GL_h[N_ZS_PTS] = {-1.0, 0.0, 1.0};

double M_diag_inv_h[N_MODE];          
double phi_vol_h[N_MODE][N_QUAD];      
double dphi_vol_h[N_MODE][N_QUAD];  
double phi_ZS_h[N_MODE][N_ZS_PTS]; 

__constant__ double d_nodes_G[N_QUAD];
__constant__ double d_weights_G[N_QUAD];
__constant__ double d_M_diag_inv[N_MODE];
__constant__ double d_phi_vol[N_MODE][N_QUAD];
__constant__ double d_dphi_vol[N_MODE][N_QUAD];
__constant__ double d_phi_ZS[N_MODE][N_ZS_PTS];

// ================== 物理逻辑与基础函数 ==================
__device__ static inline double safe_sqrt(double x) { return sqrt(x); }

__device__ static inline double calc_pressure(double rho, double rhou, double E) {
    return (gamma_gas - 1.0) * (E - 0.5 * (rhou * rhou) / rho);
}

__device__ static inline void euler_flux_1d(double U[NUM_VARS], double F[NUM_VARS]) {
    double rho = U[0], rhou = U[1], E = U[2];
    double u = rhou / rho;
    double p = calc_pressure(rho, rhou, E);
    F[0] = rhou;
    F[1] = rhou * u + p;
    F[2] = u * (E + p);
}

__device__ static inline void llf_flux_1d(double UL[NUM_VARS], double UR[NUM_VARS], double flux_res[NUM_VARS]) {
    double FL[NUM_VARS], FR[NUM_VARS];
    euler_flux_1d(UL, FL);
    euler_flux_1d(UR, FR);
    
    double rhoL = UL[0], uL = UL[1]/rhoL, pL = calc_pressure(rhoL, UL[1], UL[2]);
    double cL = safe_sqrt(gamma_gas * pL / rhoL);
    
    double rhoR = UR[0], uR = UR[1]/rhoR, pR = calc_pressure(rhoR, UR[1], UR[2]);
    double cR = safe_sqrt(gamma_gas * pR / rhoR);
    
    double alpha = fmax(fabs(uL) + cL, fabs(uR) + cR);
    for(int v = 0; v < NUM_VARS; v++) {
        flux_res[v] = 0.5 * (FL[v] + FR[v] - alpha * (UR[v] - UL[v]));
    }
}

// 降级至 P^2 空间，仅保留 0, 1, 2 阶多项式及其导数
__device__ static void eval_legendre_basis_1d(double x, double P[N_MODE], double dP[N_MODE], double ddP[N_MODE]) {
    P[0] = 1.0; dP[0] = 0.0; ddP[0] = 0.0;
    P[1] = x;   dP[1] = 1.0; ddP[1] = 0.0;
    P[2] = 0.5 * (3.0 * x * x - 1.0); dP[2] = 3.0 * x; ddP[2] = 3.0;
}

__device__ static void eval_element_derivatives_1d(Element *cell, double xi, double deriv_out[N_DERIV][NUM_VARS]) {
    double P[N_MODE], dP[N_MODE], ddP[N_MODE];
    eval_legendre_basis_1d(xi, P, dP, ddP);
    double d_dx = 2.0 / dx;

    for (int v = 0; v < NUM_VARS; v++) {
        deriv_out[0][v] = 0.0; deriv_out[1][v] = 0.0; deriv_out[2][v] = 0.0;
        for (int k = 0; k < N_MODE; k++) {
            double u_k = cell->U[v][k]; 
            if (k == 0) deriv_out[0][v] += u_k * P[0];
            if (k == 1) { 
                deriv_out[0][v] += u_k * P[1]; 
                deriv_out[1][v] += u_k * dP[1] * d_dx; 
            }
            if (k == 2) { 
                deriv_out[0][v] += u_k * P[2]; 
                deriv_out[1][v] += u_k * dP[2] * d_dx; 
                deriv_out[2][v] += u_k * ddP[2] * (d_dx * d_dx); 
            }
        }
    }
}

__device__ void compute_face_values_device_1d(Element *cell) {
    for (int var = 0; var < NUM_VARS; var++) {
        // P^2: L_0(-1)=1, L_1(-1)=-1, L_2(-1)=1
        // P^2: L_0(1)=1, L_1(1)=1, L_2(1)=1
        double vl = cell->U[var][0] - cell->U[var][1] + cell->U[var][2];
        double vr = cell->U[var][0] + cell->U[var][1] + cell->U[var][2];
        cell->face_left[var] = vl;
        cell->face_right[var] = vr;
    }
}

__global__ void compute_boundary_faces_kernel_1d(Element *d_Mesh) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    if (ii < Nx) compute_face_values_device_1d(&d_Mesh[ii]);
}

__global__ void precompute_vertex_derivs_kernel_1d(Element *d_Mesh, double d_cell_vertex_derivs[Nx][2][N_DERIV][NUM_VARS]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nx) {
        eval_element_derivatives_1d(&d_Mesh[idx], -1.0, d_cell_vertex_derivs[idx][0]);
        eval_element_derivatives_1d(&d_Mesh[idx],  1.0, d_cell_vertex_derivs[idx][1]);
    }
}
__global__ void compute_damp_coeffs_kernel_1d(Element *d_Mesh, double d_cell_vertex_derivs[Nx][2][N_DERIV][NUM_VARS], double d_damp_local[Nx][N_MODE], double *d_eta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx) return;

    Element *C_cell = &d_Mesh[idx];
    double deriv_curr_L[N_DERIV][NUM_VARS], deriv_curr_R[N_DERIV][NUM_VARS];

    for(int d=0; d<N_DERIV; d++) for(int v=0; v<NUM_VARS; v++) {
        deriv_curr_L[d][v] = d_cell_vertex_derivs[idx][0][d][v];
        deriv_curr_R[d][v] = d_cell_vertex_derivs[idx][1][d][v];
    }

    double sum_X[N_DERIV][NUM_VARS] = {0};
    double max_jump_X[N_DERIV] = {0};

    for(int d=0; d<N_DERIV; d++) {
        for(int v=0; v<NUM_VARS; v++) {
            double jump_L = (idx == 0) ? 0.0 : fabs(deriv_curr_L[d][v] - d_cell_vertex_derivs[idx - 1][1][d][v]);
            double jump_R = (idx == Nx - 1) ? 0.0 : fabs(deriv_curr_R[d][v] - d_cell_vertex_derivs[idx + 1][0][d][v]);
            sum_X[d][v] = jump_L + jump_R;
            
            if (sum_X[d][v] > max_jump_X[d]) {
                max_jump_X[d] = sum_X[d][v];
            }
        }
    }

    // [新增逻辑] 计算当前单元的跳跃指示器 eta_j (文献中基于 0 阶导数，即变量值本身的绝对跳跃)
    double local_eta = 0.0;
    for(int v=0; v<NUM_VARS; v++) {
        double eta_v = sum_X[0][v] / (2.0 * dx); // ||[u_h]|| / (2*h_j)
        if (eta_v > local_eta) local_eta = eta_v;
    }
    d_eta[idx] = local_eta;

    double rho = C_cell->U[0][0], rhou = C_cell->U[1][0], E = C_cell->U[2][0];
    double p = calc_pressure(rho, rhou, E);
    double c = safe_sqrt(gamma_gas * p / rho);
    double H = (E + p) / rho;
    double beta_x = fabs(rhou / rho) + c;

    d_damp_local[idx][0] = 0.0;
    for(int l=1; l<=2; l++) {
        double S_X = dx * max_jump_X[0];
        if (l >= 1) S_X += 2.0 * dx * dx * max_jump_X[1];
        if (l >= 2) S_X += 6.0 * dx * dx * dx * max_jump_X[2];
        
        double val = 0.0;
        if (H > 1e-13) {
            val = (1.0 / H) * (beta_x / (dx * dx)) * S_X;
        }
        d_damp_local[idx][l] = val;
    }
}

// 移除混合架构，直接实施纯粹的跳跃阻尼滤波

    __global__ void apply_hybrid_filter_kernel_1d(Element *d_Mesh, double d_damp_local[Nx][N_MODE], double *d_eta, double eta_max, double dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx) return;
    
    double eta_j = d_eta[id];

    // 如果局部指示器 > 1，说明单元包含激波或强梯度，执行混合策略
    if (eta_j > 1.0 && eta_max > 1.0) {
        // 计算混合权重因子
        printf("trigger!");
        double omega = (eta_max - eta_j) / (eta_max - 1.0);
        // 数值防越界保护
        if (omega < 0.0) omega = 0.0;
        if (omega > 1.0) omega = 1.0;

        for (int v = 0; v < NUM_VARS; v++) {
            for (int k = 1; k < N_MODE; k++) { 
                // k=0 是单元均值，根据物理守恒率不参与任何缩放与修改
                double U_DG = d_Mesh[id].U[v][k];
                double sigma = d_damp_local[id][k];
                
                // 计算滤波限制解
                double U_lim = U_DG * exp(-sigma * dt);
                
                // 进行平滑的非线性凸组合混合
                d_Mesh[id].U[v][k] = omega * U_DG + (1.0 - omega) * U_lim;
            }
        }
    }
    else{
         printf("noooootrigger!");
    }
    // else { 
    //    eta_j <= 1.0 区域为光滑区，无需任何限制策略，保留原高阶 U_DG 数据 
    // }
}


__global__ void compute_rhs_kernel_1d(Element *d_Mesh, double d_RHS[Nx][NUM_VARS][N_MODE]) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nx) return;

    Element *cell = &d_Mesh[idx];
    double Vol_Int[NUM_VARS][N_MODE] = {{0}};
    double Surf_Int[NUM_VARS][N_MODE] = {{0}};

    for (int q = 0; q < N_QUAD; q++) {
        double U_phys[NUM_VARS] = {0};
        for (int k = 0; k < N_MODE; k++)
            for(int v=0; v<NUM_VARS; v++) U_phys[v] += cell->U[v][k] * d_phi_vol[k][q];
        
        double F_val[NUM_VARS];
        euler_flux_1d(U_phys, F_val);

        for (int k = 0; k < N_MODE; k++) {
            for(int v=0; v<NUM_VARS; v++) {
                Vol_Int[v][k] += d_weights_G[q] * F_val[v] * d_dphi_vol[k][q];
            }
        }
    }

    double U_L_inside[NUM_VARS], U_R_inside[NUM_VARS];
    double U_L_outside[NUM_VARS], U_R_outside[NUM_VARS];
    for(int v=0; v<NUM_VARS; v++) {
        U_L_inside[v] = cell->face_left[v];
        U_R_inside[v] = cell->face_right[v];
        U_L_outside[v] = (idx == 0) ? U_L_inside[v] : d_Mesh[idx - 1].face_right[v];
        U_R_outside[v] = (idx == Nx - 1) ? U_R_inside[v] : d_Mesh[idx + 1].face_left[v];
    }

    double num_f_L[NUM_VARS], num_f_R[NUM_VARS];
    llf_flux_1d(U_L_outside, U_L_inside, num_f_L);
    llf_flux_1d(U_R_inside, U_R_outside, num_f_R);

    for (int k = 0; k < N_MODE; k++) {
        double phi_1 = 1.0;
        double phi_m1 = (k % 2 == 0) ? 1.0 : -1.0; 
        for(int v=0; v<NUM_VARS; v++) {
            Surf_Int[v][k] = num_f_R[v] * phi_1 - num_f_L[v] * phi_m1;
        }
    }

    for(int v=0; v<NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            d_RHS[idx][v][k] = (Vol_Int[v][k] - Surf_Int[v][k]) * d_M_diag_inv[k] / (dx / 2.0);
        }
    }
}

__global__ void apply_zhang_shu_limiter_kernel_1d(Element *d_Mesh) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx) return;

    Element *cell = &d_Mesh[id];
    const double eps = 1e-13;
    double Ubar[NUM_VARS];
    // 单元均值被严格守恒
    for (int v = 0; v < NUM_VARS; v++) Ubar[v] = cell->U[v][0];

    double U_test[N_ZS_PTS][NUM_VARS];
    for (int pt = 0; pt < N_ZS_PTS; pt++) {
        for (int v = 0; v < NUM_VARS; v++) U_test[pt][v] = 0.0;
        for (int k = 0; k < N_MODE; k++) {
            for (int v = 0; v < NUM_VARS; v++) U_test[pt][v] += cell->U[v][k] * d_phi_ZS[k][pt];
        }
    }

    // 保密度正定：通过标量系数 theta1 全局缩放避免破坏物理状态，拒绝简单截断
    double rho_min = U_test[0][0];
    for (int i = 1; i < N_ZS_PTS; i++) if (U_test[i][0] < rho_min) rho_min = U_test[i][0];

    double theta1 = 1.0;
    if (rho_min < eps) {
        if (Ubar[0] - rho_min > 1e-14) theta1 = (Ubar[0] - eps) / (Ubar[0] - rho_min);
        else theta1 = 0.0;
        if (theta1 < 0.0) theta1 = 0.0; if (theta1 > 1.0) theta1 = 1.0;
        for (int v = 0; v < NUM_VARS; v++) for (int k = 1; k < N_MODE; k++) cell->U[v][k] *= theta1;
        for (int i = 0; i < N_ZS_PTS; i++) for (int v = 0; v < NUM_VARS; v++) U_test[i][v] = Ubar[v] + theta1 * (U_test[i][v] - Ubar[v]);
    }

    // 保压力正定：使用缩放系数 theta2 以避免强制截断破坏能量守恒
    double theta2 = 1.0;
    for (int i = 0; i < N_ZS_PTS; i++) {
        double p_i = calc_pressure(U_test[i][0], U_test[i][1], U_test[i][2]);
        if (p_i < eps) {
            double t_L = 0.0, t_R = 1.0;
            for (int iter = 0; iter < 50; iter++) {
                double t_mid = 0.5 * (t_L + t_R);
                double Ut[NUM_VARS];
                for (int v = 0; v < NUM_VARS; v++) Ut[v] = Ubar[v] + t_mid * (U_test[i][v] - Ubar[v]);
                if (calc_pressure(Ut[0], Ut[1], Ut[2]) < eps) t_R = t_mid; else t_L = t_mid;
            }
            if (t_L < theta2) theta2 = t_L;
        }
    }
    if (theta2 < 1.0)
        for (int v = 0; v < NUM_VARS; v++) for (int k = 1; k < N_MODE; k++) cell->U[v][k] *= theta2;
}

__global__ void update_rk3_stage_kernel_1d(Element *d_Mesh, double d_U0[Nx][NUM_VARS][N_MODE], 
                                           double d_U_prev[Nx][NUM_VARS][N_MODE], 
                                           double d_RHS[Nx][NUM_VARS][N_MODE], int stage, double dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx) return;
    for (int v = 0; v < NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            if (stage == 1) d_Mesh[id].U[v][k] = d_U0[id][v][k] + dt * d_RHS[id][v][k];
            else if (stage == 2) d_Mesh[id].U[v][k] = 0.75 * d_U0[id][v][k] + 0.25 * (d_U_prev[id][v][k] + dt * d_RHS[id][v][k]);
            else if (stage == 3) d_Mesh[id].U[v][k] = (1.0 / 3.0) * d_U0[id][v][k] + (2.0 / 3.0) * (d_U_prev[id][v][k] + dt * d_RHS[id][v][k]);
        }
    }
}

__global__ void copy_mesh_to_tmp_1d(Element *d_Mesh, double d_U_tmp[Nx][NUM_VARS][N_MODE]) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < Nx) {
        for(int v=0; v<NUM_VARS; v++) for(int k=0; k<N_MODE; k++) d_U_tmp[id][v][k] = d_Mesh[id].U[v][k];
    }
}

// 并行规约算子：动态严格探测全场最大特征波速
struct WaveSpeedFunctor {
    Element* mesh;
    WaveSpeedFunctor(Element* _mesh) : mesh(_mesh) {}
    __device__ double operator()(const int& idx) const {
        double rho = mesh[idx].U[0][0]; 
        double rhou = mesh[idx].U[1][0], E = mesh[idx].U[2][0];
        if (rho < 1e-12) return 0.0; 
        double u = rhou / rho;
        double p = (gamma_gas - 1.0) * (E - 0.5 * rhou * rhou / rho);
        if (p < 1e-12) return fabs(u); 
        double c = sqrt(gamma_gas * p / rho);
        return fabs(u) + c;
    }
};

static inline double legendre_1d(int i, double x) {
    if (i == 0) return 1.0; 
    if (i == 1) return x; 
    if (i == 2) return 0.5 * (3.0 * x * x - 1.0); 
    return 0.0;
}
static inline double d_legendre_1d(int i, double x) {
    if (i == 0) return 0.0; 
    if (i == 1) return 1.0; 
    if (i == 2) return 3.0 * x; 
    return 0.0;
}

void init_quadrature_and_matrices_1d(void) {
    for (int k = 0; k < N_MODE; k++) {
        M_diag_inv_h[k] = (2.0 * k + 1.0) / 2.0; 
        for (int q = 0; q < N_QUAD; q++) {
            double r = nodes_G_h[q];
            phi_vol_h[k][q]  = legendre_1d(k, r);
            dphi_vol_h[k][q] = d_legendre_1d(k, r);
        }
        for (int pt = 0; pt < N_ZS_PTS; pt++) {
            phi_ZS_h[k][pt] = legendre_1d(k, nodes_GL_h[pt]);
        }
    }
}



// 采用亚网格输出分辨率以体现 P2 精确度
void output_results_1d(double t) {
    FILE *fp = fopen("result.dat", "w");
    if (fp == NULL) return;
    fprintf(fp, "VARIABLES = \"X\", \"Rho\", \"U\", \"P\"\n");

    int N_sub = 3; 
    for (int ii = 0; ii < Nx; ii++) {
        Element *cell = &Mesh_h[ii]; 
        double xc = xL + (ii + 0.5) * dx;
        
        for (int s = 0; s < N_sub; s++) {
            double xi = -1.0 + 2.0 * s / (double)(N_sub - 1);
            double x_phys = xc + xi * (dx / 2.0);
            
            double U_phys[NUM_VARS] = {0};
            for (int k = 0; k < N_MODE; k++) {
                double phi_val = legendre_1d(k, xi);
                for (int v = 0; v < NUM_VARS; v++) U_phys[v] += cell->U[v][k] * phi_val;
            }
            double rho = U_phys[0];
            double u = U_phys[1] / rho;
            double p = (gamma_gas - 1.0) * (U_phys[2] - 0.5 * rho * u * u);
            
            fprintf(fp, "%lf %lf %lf %lf\n", x_phys, rho, u, p);
        }
    }
    fclose(fp);
    printf("Results strictly written to result.dat\n");
}


// ================== 完备的精确黎曼求解器 ==================

// 计算波函数 f_K 及其导数 df_K
static double eval_f_K(double p, double rho_K, double p_K, double c_K, double *df_K) {
    double f_val;
    if (p <= p_K) { // 稀疏波
        double pratio = p / p_K;
        double a = (gamma_gas - 1.0) / (2.0 * gamma_gas);
        f_val = (2.0 * c_K / (gamma_gas - 1.0)) * (pow(pratio, a) - 1.0);
        *df_K = (1.0 / (rho_K * c_K)) * pow(pratio, -(gamma_gas + 1.0) / (2.0 * gamma_gas));
    } else {        // 激波
        double A = 2.0 / ((gamma_gas + 1.0) * rho_K);
        double B = (gamma_gas - 1.0) / (gamma_gas + 1.0) * p_K;
        double sqrt_term = sqrt(A / (p + B));
        f_val = (p - p_K) * sqrt_term;
        *df_K = sqrt_term * (1.0 - (p - p_K) / (2.0 * (p + B)));
    }
    return f_val;
}

// 严格空间采样，返回精确物理量
static void exact_riemann_solution(double x, double t, double x_discontinuity, 
                                   double rho_L, double u_L, double p_L, 
                                   double rho_R, double u_R, double p_R,
                                   double *rho_out, double *u_out, double *p_out) {
    double c_L = sqrt(gamma_gas * p_L / rho_L);
    double c_R = sqrt(gamma_gas * p_R / rho_R);
    
    // TRRS (Two-Rarefaction Riemann Solver) 作为牛顿迭代的初值猜测，具有极高的鲁棒性
    double z = (gamma_gas - 1.0) / (2.0 * gamma_gas);
    double p_L_pow = pow(p_L, z);
    double p_R_pow = pow(p_R, z);
    double p_star = pow((c_L + c_R - 0.5 * (gamma_gas - 1.0) * (u_R - u_L)) / (c_L / p_L_pow + c_R / p_R_pow), 1.0 / z);
    if (p_star < 1e-6 || isnan(p_star)) p_star = 0.5 * (p_L + p_R);

    // Newton-Raphson 迭代求解代数方程 (非 PDE 截断)
    double tol = 1e-7;
    for (int iter = 0; iter < 50; iter++) {
        double df_L, df_R;
        double f_val = eval_f_K(p_star, rho_L, p_L, c_L, &df_L) + eval_f_K(p_star, rho_R, p_R, c_R, &df_R) + (u_R - u_L);
        double df_val = df_L + df_R;
        
        double dp = -f_val / df_val;
        double p_new = p_star + dp;
        
        // 纯代数求根防发散边界，不涉及网格物理量的非物理截断
        if (p_new < 1e-6) p_new = 1e-6; 
        
        double err = fabs(p_new - p_star) / (p_star + 1e-6);
        p_star = p_new;
        if (err < tol) break;
    }
    
    // 计算中间速度 u_star
    double df_dummy1, df_dummy2;
    double u_star = 0.5 * (u_L + u_R + eval_f_K(p_star, rho_R, p_R, c_R, &df_dummy1) - eval_f_K(p_star, rho_L, p_L, c_L, &df_dummy2));
    
    double S = (x - x_discontinuity) / t;
    
    // ================= 全波系严密判断 =================
    if (S <= u_star) { // 位于接触间断左侧
        if (p_star <= p_L) { // 左稀疏波
            double S_HL = u_L - c_L;
            if (S <= S_HL) { 
                *rho_out = rho_L; *u_out = u_L; *p_out = p_L; 
            } else {
                double c_star_L = c_L * pow(p_star / p_L, (gamma_gas - 1.0) / (2.0 * gamma_gas));
                double S_TL = u_star - c_star_L;
                if (S >= S_TL) {
                    *rho_out = rho_L * pow(p_star / p_L, 1.0 / gamma_gas);
                    *u_out = u_star; *p_out = p_star;
                } else { // 稀疏波扇区内
                    double u_fan = (2.0 / (gamma_gas + 1.0)) * (c_L + 0.5 * (gamma_gas - 1.0) * u_L + S);
                    double c_fan = (2.0 / (gamma_gas + 1.0)) * (c_L + 0.5 * (gamma_gas - 1.0) * (u_L - S));
                    *rho_out = rho_L * pow(c_fan / c_L, 2.0 / (gamma_gas - 1.0));
                    *u_out = u_fan;
                    *p_out = p_L * pow(c_fan / c_L, 2.0 * gamma_gas / (gamma_gas - 1.0));
                }
            }
        } else { // 左激波
            double S_L = u_L - c_L * sqrt(((gamma_gas + 1.0) / (2.0 * gamma_gas)) * (p_star / p_L) + ((gamma_gas - 1.0) / (2.0 * gamma_gas)));
            if (S <= S_L) {
                *rho_out = rho_L; *u_out = u_L; *p_out = p_L;
            } else {
                double pratio = p_star / p_L;
                *rho_out = rho_L * (pratio + (gamma_gas - 1.0) / (gamma_gas + 1.0)) / (pratio * (gamma_gas - 1.0) / (gamma_gas + 1.0) + 1.0);
                *u_out = u_star; *p_out = p_star;
            }
        }
    } else { // 位于接触间断右侧
        if (p_star <= p_R) { // 右稀疏波
            double S_HR = u_R + c_R;
            if (S >= S_HR) {
                *rho_out = rho_R; *u_out = u_R; *p_out = p_R;
            } else {
                double c_star_R = c_R * pow(p_star / p_R, (gamma_gas - 1.0) / (2.0 * gamma_gas));
                double S_TR = u_star + c_star_R;
                if (S <= S_TR) {
                    *rho_out = rho_R * pow(p_star / p_R, 1.0 / gamma_gas);
                    *u_out = u_star; *p_out = p_star;
                } else { // 稀疏波扇区内
                    double u_fan = (2.0 / (gamma_gas + 1.0)) * (-c_R + 0.5 * (gamma_gas - 1.0) * u_R + S);
                    double c_fan = (2.0 / (gamma_gas + 1.0)) * (c_R - 0.5 * (gamma_gas - 1.0) * (u_R - S));
                    *rho_out = rho_R * pow(c_fan / c_R, 2.0 / (gamma_gas - 1.0));
                    *u_out = u_fan;
                    *p_out = p_R * pow(c_fan / c_R, 2.0 * gamma_gas / (gamma_gas - 1.0));
                }
            }
        } else { // 右激波
            double S_R = u_R + c_R * sqrt(((gamma_gas + 1.0) / (2.0 * gamma_gas)) * (p_star / p_R) + ((gamma_gas - 1.0) / (2.0 * gamma_gas)));
            if (S >= S_R) {
                *rho_out = rho_R; *u_out = u_R; *p_out = p_R;
            } else {
                double pratio = p_star / p_R;
                *rho_out = rho_R * (pratio + (gamma_gas - 1.0) / (gamma_gas + 1.0)) / (pratio * (gamma_gas - 1.0) / (gamma_gas + 1.0) + 1.0);
                *u_out = u_star; *p_out = p_star;
            }
        }
    }
}

// L2 投影初始化函数
void init_condition_exact_riemann(double t_init) {
    const int N_GQ5 = 5;
    const double nodes_GQ5[5] = {-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640};
    const double weights_GQ5[5] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891};

    double x_discontinuity = 0.3;
    double rho_L = 10000.0, u_L = 0.0, p_L = 10000.0;
    double rho_R = 1.0,     u_R = 0.0, p_R = 1.0;

    for (int ii = 0; ii < Nx; ii++) {
        Element *cell = &Mesh_h[ii];
        for(int v = 0; v < NUM_VARS; v++) {
            for(int k = 0; k < N_MODE; k++) cell->U[v][k] = 0.0;
        }
        
        double xc = xL + (ii + 0.5) * dx;
        
        for (int q = 0; q < N_GQ5; q++) {
            double xi = nodes_GQ5[q];
            double x_phys = xc + xi * (dx / 2.0);
            double w = weights_GQ5[q];
            
            double rho, u, p;
            exact_riemann_solution(x_phys, t_init, x_discontinuity, rho_L, u_L, p_L, rho_R, u_R, p_R, &rho, &u, &p);
            
            double rhou = rho * u;
            double E = p / (gamma_gas - 1.0) + 0.5 * rho * u * u;
            
            double phys_vars[NUM_VARS] = {rho, rhou, E};
            
            for (int k = 0; k < N_MODE; k++) {
                double phi_k = legendre_1d(k, xi);
                for (int v = 0; v < NUM_VARS; v++) {
                    cell->U[v][k] += w * phys_vars[v] * phi_k * M_diag_inv_h[k];
                }
            }
        }
    }
}
int main(void) {
    init_quadrature_and_matrices_1d();
 

    cudaMemcpyToSymbol(d_nodes_G, nodes_G_h, sizeof(nodes_G_h));
    cudaMemcpyToSymbol(d_weights_G, weights_G_h, sizeof(weights_G_h));
    cudaMemcpyToSymbol(d_M_diag_inv, M_diag_inv_h, sizeof(M_diag_inv_h));
    cudaMemcpyToSymbol(d_phi_vol, phi_vol_h, sizeof(phi_vol_h));
    cudaMemcpyToSymbol(d_dphi_vol, dphi_vol_h, sizeof(dphi_vol_h));
    cudaMemcpyToSymbol(d_phi_ZS, phi_ZS_h, sizeof(phi_ZS_h));

    Element *d_Mesh;
    cudaMalloc(&d_Mesh, Nx * sizeof(Element));

    double (*d_cell_vertex_derivs)[2][N_DERIV][NUM_VARS], (*d_damp_local)[N_MODE];
    cudaMalloc(&d_cell_vertex_derivs, Nx * sizeof(*d_cell_vertex_derivs));
    cudaMalloc(&d_damp_local, Nx * sizeof(*d_damp_local));

    double (*d_U0)[NUM_VARS][N_MODE], (*d_U_tmp)[NUM_VARS][N_MODE], (*d_RHS)[NUM_VARS][N_MODE];
    cudaMalloc(&d_U0, Nx * sizeof(*d_U0)); 
    cudaMalloc(&d_U_tmp, Nx * sizeof(*d_U_tmp));
    cudaMalloc(&d_RHS, Nx * sizeof(*d_RHS));

    double *d_eta;
    cudaMalloc(&d_eta, Nx * sizeof(double));
    thrust::device_ptr<double> d_eta_ptr(d_eta); // 用于 thrust 并行规约

    int blockSize = 256;
    int gridSize = (Nx + blockSize - 1) / blockSize;

    thrust::counting_iterator<int> iter(0);

  
    int nit = 0;

  
    double current_time = 0; 
    init_condition_exact_riemann(current_time);

    // 【修正点 2】：将携带正确初始物理场数据的 Mesh_h 拷贝至显存中的 d_Mesh
    cudaMemcpy(d_Mesh, Mesh_h, Nx * sizeof(Element), cudaMemcpyHostToDevice);

    apply_zhang_shu_limiter_kernel_1d<<<gridSize, blockSize>>>(d_Mesh);

    printf("Starting 1D Lax Problem (P2 DG with Jump Filter)...\n");
    printf("%-10s  %-14s %-14s\n", "Step", "Time", "dt");
    printf("------------------------------------------\n");

while (current_time < T_END) {
        double max_wave = thrust::transform_reduce(thrust::device, iter, iter + Nx, WaveSpeedFunctor(d_Mesh), 0.0, thrust::maximum<double>());
        if (max_wave < 1e-6) max_wave = 1.0;
        
        double dt = CFL * dx / max_wave;
        if (current_time + dt > T_END) dt = T_END - current_time;

        copy_mesh_to_tmp_1d<<<gridSize, blockSize>>>(d_Mesh, d_U0);

        for (int stage = 1; stage <= 3; stage++) {
            if (stage > 1) copy_mesh_to_tmp_1d<<<gridSize, blockSize>>>(d_Mesh, d_U_tmp);
            
            compute_boundary_faces_kernel_1d<<<gridSize, blockSize>>>(d_Mesh);
            compute_rhs_kernel_1d<<<gridSize, blockSize>>>(d_Mesh, d_RHS);

            update_rk3_stage_kernel_1d<<<gridSize, blockSize>>>(d_Mesh, d_U0, d_U_tmp, d_RHS, stage, dt);

            // ================= 混合架构后处理 =================
            compute_boundary_faces_kernel_1d<<<gridSize, blockSize>>>(d_Mesh);
            precompute_vertex_derivs_kernel_1d<<<gridSize, blockSize>>>(d_Mesh, d_cell_vertex_derivs);
            
            // 步骤1：计算各个单元的阻尼系数以及局部跳跃指示器 d_eta
            compute_damp_coeffs_kernel_1d<<<gridSize, blockSize>>>(d_Mesh, d_cell_vertex_derivs, d_damp_local, d_eta);
            
            // 步骤2：利用 thrust 进行并行规约，求取全场的极值 eta_max
            double eta_max = thrust::reduce(thrust::device, d_eta_ptr, d_eta_ptr + Nx, 0.0, thrust::maximum<double>());
            
            // 步骤3：基于 eta_max 和局部 eta 应用非线性混合限制
            apply_hybrid_filter_kernel_1d<<<gridSize, blockSize>>>(d_Mesh, d_damp_local, d_eta, eta_max, dt);

            // ================= 张-舒 保正限制器 =================
            // 控制密度与压力的严格物理有效性 (拒绝使用简单的强制截断)
            apply_zhang_shu_limiter_kernel_1d<<<gridSize, blockSize>>>(d_Mesh);
        }

        current_time += dt; 
        nit++;
        if (nit % 100 == 0) printf("%-10d  %-14.6e %-14.6e \n", nit, current_time, dt);
    }
    cudaMemcpy(Mesh_h, d_Mesh, Nx * sizeof(Element), cudaMemcpyDeviceToHost);
    output_results_1d(T_END);
cudaFree(d_eta);
    cudaFree(d_Mesh); 
    cudaFree(d_cell_vertex_derivs); cudaFree(d_damp_local); 
    cudaFree(d_U0); cudaFree(d_U_tmp); cudaFree(d_RHS);
    return 0;
}
