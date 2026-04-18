#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define M_PI 3.14159265358979323846
#define Nx 960
#define Ny 240      // 根据双马赫计算域比例调整
#define xL 0.0
#define xR 4.0
#define yL 0.0
#define yR 1.0
#define dx ((xR - xL) / Nx)
#define dy ((yR - yL) / Ny)
#define gamma_gas 1.4      // 必须修改为 1.4

#define NUM_VARS 4 
// 移除 #define Mach 800.0

#define CFL   0.75
#define T_END 0.2          // 终止时间

// ================== P^3 升级核心宏定义 ==================
#define N_MODE 10        // P^3 基函数总数
#define N_QUAD 16        // 4x4 体积分点
#define N_FACE_PTS 4     // 面上的高斯积分点
#define N_ZS_PTS 32      // Zhang-Shu 限制器的测试点 (4x4x2)
#define N_DERIV 10       // 0 到 3 阶偏导数总数

typedef struct {
    double U[NUM_VARS][N_MODE]; 
    double face_top[NUM_VARS][N_FACE_PTS];
    double face_bottom[NUM_VARS][N_FACE_PTS];
    double face_left[NUM_VARS][N_FACE_PTS];
    double face_right[NUM_VARS][N_FACE_PTS];
} Element;

Element Mesh_h[Nx * Ny];

// ================== 积分点与映射定义 (4点精度) ==================
static const double nodes_G_h[N_FACE_PTS]   = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
static const double weights_G_h[N_FACE_PTS] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};

static const double nodes_GL_h[N_FACE_PTS]   = {-1.0, -0.4472135954999579, 0.4472135954999579, 1.0};
static const double weights_GL_h[N_FACE_PTS] = {0.1666666666666666, 0.8333333333333333, 0.8333333333333333, 0.1666666666666666};

static const int mk_map_h[N_MODE] = {0, 1, 0, 2, 1, 0, 3, 2, 1, 0};
static const int nk_map_h[N_MODE] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};

double r_quad_h[N_QUAD], s_quad_h[N_QUAD], w_quad_h[N_QUAD]; 
double M_diag_inv_h[N_MODE];          
double phi_vol_h[N_MODE][N_QUAD];      
double dphi_dr_vol_h[N_MODE][N_QUAD];  
double dphi_ds_vol_h[N_MODE][N_QUAD];  
double phi_face_T_h[N_MODE][N_FACE_PTS], phi_face_B_h[N_MODE][N_FACE_PTS];
double phi_face_L_h[N_MODE][N_FACE_PTS], phi_face_R_h[N_MODE][N_FACE_PTS];
double phi_ZS_h[N_MODE][N_ZS_PTS]; 

__constant__ double d_nodes_G[N_FACE_PTS];
__constant__ double d_weights_G[N_FACE_PTS];
__constant__ int d_mk_map[N_MODE];
__constant__ int d_nk_map[N_MODE];

__constant__ double d_r_quad[N_QUAD], d_s_quad[N_QUAD], d_w_quad[N_QUAD];
__constant__ double d_M_diag_inv[N_MODE];
__constant__ double d_phi_vol[N_MODE][N_QUAD];
__constant__ double d_dphi_dr_vol[N_MODE][N_QUAD];
__constant__ double d_dphi_ds_vol[N_MODE][N_QUAD];
__constant__ double d_phi_face_T[N_MODE][N_FACE_PTS], d_phi_face_B[N_MODE][N_FACE_PTS];
__constant__ double d_phi_face_L[N_MODE][N_FACE_PTS], d_phi_face_R[N_MODE][N_FACE_PTS];
__constant__ double d_phi_ZS[N_MODE][N_ZS_PTS];

// ================== 物理逻辑与基础函数 ==================
__device__ static inline double safe_sqrt(double x) {
    if(x<-1) return sqrt(x);
    else return sqrt(fmax(x, 0.0));
}


__device__ static inline double calc_pressure(double rho, double rhou, double rhov, double E) {
    return (gamma_gas - 1.0) * (E - 0.5 * (rhou * rhou + rhov * rhov) / rho);
}

__device__ static inline void euler_flux(double U[NUM_VARS], double F[NUM_VARS], double G[NUM_VARS]) {
    double rho = U[0], rhou = U[1], rhov = U[2], E = U[3];
    double u = rhou / rho, v = rhov / rho;
    double p = calc_pressure(rho, rhou, rhov, E);

    F[0] = rhou;       F[1] = rhou * u + p; F[2] = rhou * v;       F[3] = u * (E + p);
    G[0] = rhov;       G[1] = rhou * v;     G[2] = rhov * v + p;   G[3] = v * (E + p);
}

__device__ static inline void llf_flux_vector(double UL[NUM_VARS], double UR[NUM_VARS], double nx, double ny, double flux_res[NUM_VARS]) {
    double FL[NUM_VARS], GL[NUM_VARS], FR[NUM_VARS], GR[NUM_VARS];
    euler_flux(UL, FL, GL);
    euler_flux(UR, FR, GR);
    
    double rhoL = UL[0]; double uL = UL[1]/rhoL, vL = UL[2]/rhoL;
    double pL = calc_pressure(rhoL, UL[1], UL[2], UL[3]);
    double cL = safe_sqrt(gamma_gas * pL / rhoL);
    double unL = uL * nx + vL * ny;
    
    double rhoR = UR[0]; double uR = UR[1]/rhoR, vR = UR[2]/rhoR;
    double pR = calc_pressure(rhoR, UR[1], UR[2], UR[3]);
    double cR = safe_sqrt(gamma_gas * pR / rhoR);
    double unR = uR * nx + vR * ny;
    
    double alpha = fmax(fabs(unL) + cL, fabs(unR) + cR);
    
    for(int v = 0; v < NUM_VARS; v++) {
        double flux_n_L = FL[v] * nx + GL[v] * ny;
        double flux_n_R = FR[v] * nx + GR[v] * ny;
        flux_res[v] = 0.5 * (flux_n_L + flux_n_R - alpha * (UR[v] - UL[v]));
    }
}

__device__ static void eval_legendre_basis_1d(double x, double P[4], double dP[4], double ddP[4], double dddP[4]) {
    P[0] = 1.0; dP[0] = 0.0; ddP[0] = 0.0; dddP[0] = 0.0;
    P[1] = x;   dP[1] = 1.0; ddP[1] = 0.0; dddP[1] = 0.0;
    P[2] = 0.5 * (3.0 * x * x - 1.0); dP[2] = 3.0 * x; ddP[2] = 3.0; dddP[2] = 0.0;
    P[3] = 0.5 * (5.0 * x * x * x - 3.0 * x); dP[3] = 1.5 * (5.0 * x * x - 1.0); ddP[3] = 15.0 * x; dddP[3] = 15.0;
}

__device__ static void eval_element_derivatives(Element *cell, double xi, double eta, double deriv_out[N_DERIV][NUM_VARS]) {
    double P_xi[4], dP_xi[4], ddP_xi[4], dddP_xi[4];
    double P_eta[4], dP_eta[4], ddP_eta[4], dddP_eta[4];

    eval_legendre_basis_1d(xi, P_xi, dP_xi, ddP_xi, dddP_xi);
    eval_legendre_basis_1d(eta, P_eta, dP_eta, ddP_eta, dddP_eta);

    double d_dx = 2.0 / dx;
    double d_dy = 2.0 / dy;

    for (int v = 0; v < NUM_VARS; v++) {
        for (int d = 0; d < N_DERIV; d++) deriv_out[d][v] = 0.0;
        for (int k = 0; k < N_MODE; k++) {
            int mk = d_mk_map[k], nk = d_nk_map[k];
            double u_k = cell->U[v][k]; 
            deriv_out[0][v] += u_k * P_xi[mk] * P_eta[nk];
            deriv_out[1][v] += u_k * dP_xi[mk] * P_eta[nk] * d_dx;
            deriv_out[2][v] += u_k * P_xi[mk] * dP_eta[nk] * d_dy;
            deriv_out[3][v] += u_k * ddP_xi[mk] * P_eta[nk] * (d_dx * d_dx);
            deriv_out[4][v] += u_k * dP_xi[mk] * dP_eta[nk] * (d_dx * d_dy);
            deriv_out[5][v] += u_k * P_xi[mk] * ddP_eta[nk] * (d_dy * d_dy);
            deriv_out[6][v] += u_k * dddP_xi[mk] * P_eta[nk] * (d_dx * d_dx * d_dx);
            deriv_out[7][v] += u_k * ddP_xi[mk] * dP_eta[nk] * (d_dx * d_dx * d_dy);
            deriv_out[8][v] += u_k * dP_xi[mk] * ddP_eta[nk] * (d_dx * d_dy * d_dy);
            deriv_out[9][v] += u_k * P_xi[mk] * dddP_eta[nk] * (d_dy * d_dy * d_dy);
        }
    }
}

__device__ void compute_face_values_device(Element *cell) {
    for (int var = 0; var < NUM_VARS; var++) {
        for (int p = 0; p < N_FACE_PTS; p++) {
            double vt = 0.0, vb = 0.0, vl = 0.0, vr = 0.0;
            for (int k = 0; k < N_MODE; k++) {
                double mode_val = cell->U[var][k];
                vt += d_phi_face_T[k][p] * mode_val;
                vb += d_phi_face_B[k][p] * mode_val;
                vl += d_phi_face_L[k][p] * mode_val;
                vr += d_phi_face_R[k][p] * mode_val;
            }
            cell->face_top[var][p]    = vt;
            cell->face_bottom[var][p] = vb;
            cell->face_left[var][p]   = vl;
            cell->face_right[var][p]  = vr;
        }
    }
}

__global__ void apply_ghost_cells_kernel(Element *d_Mesh, Element *d_Ghost_bottom, Element *d_Ghost_top, 
                                         Element *d_Ghost_left, Element *d_Ghost_right, double current_time) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 激波后状态
    const double rho1 = 8.0;
    const double u1   = 4.125 * sqrt(3.0);
    const double v1   = -4.125;
    const double p1   = 116.5;
    const double E1   = p1 / (gamma_gas - 1.0) + 0.5 * rho1 * (u1 * u1 + v1 * v1);
    const double Q1[4] = {rho1, rho1 * u1, rho1 * v1, E1};

    // 激波前状态
    const double rho0 = 1.4;
    const double u0   = 0.0;
    const double v0   = 0.0;
    const double p0   = 1.0;
    const double E0   = p0 / (gamma_gas - 1.0);
    const double Q0[4] = {rho0, rho0 * u0, rho0 * v0, E0};

    if (id < Nx) {
        // --- 下边界 (Bottom, y = 0) ---
        double xc_bottom = xL + (id + 0.5) * dx;
        if (xc_bottom < 1.0 / 6.0) {
            // 激波流入区
            for (int v = 0; v < NUM_VARS; v++) {
                d_Ghost_bottom[id].U[v][0] = Q1[v];
                for (int k = 1; k < N_MODE; k++) d_Ghost_bottom[id].U[v][k] = 0.0;
            }
        } else {
            // 反射壁面：法向速度（Y动量）由于奇偶性映射翻转
            for (int k = 0; k < N_MODE; k++) {
                int nk = d_nk_map[k]; // 获取 y 方向基函数的阶数
                double sign = (nk % 2 == 0) ? 1.0 : -1.0; 
                
                d_Ghost_bottom[id].U[0][k] =  sign * d_Mesh[0 * Nx + id].U[0][k];
                d_Ghost_bottom[id].U[1][k] =  sign * d_Mesh[0 * Nx + id].U[1][k];
                d_Ghost_bottom[id].U[2][k] = -sign * d_Mesh[0 * Nx + id].U[2][k]; // Y 动量翻转
                d_Ghost_bottom[id].U[3][k] =  sign * d_Mesh[0 * Nx + id].U[3][k];
            }
        }
        compute_face_values_device(&d_Ghost_bottom[id]);

        // --- 上边界 (Top, y = 1) 动态激波位置 ---
        double sin60 = sqrt(3.0) / 2.0;
        double xs_top = (1.0 / 6.0) + (10.0 * current_time + 0.5) / sin60;
        double xc_top = xL + (id + 0.5) * dx;
        
        if (xc_top < xs_top) {
            for (int v = 0; v < NUM_VARS; v++) {
                d_Ghost_top[id].U[v][0] = Q1[v];
                for (int k = 1; k < N_MODE; k++) d_Ghost_top[id].U[v][k] = 0.0;
            }
        } else {
            for (int v = 0; v < NUM_VARS; v++) {
                d_Ghost_top[id].U[v][0] = Q0[v];
                for (int k = 1; k < N_MODE; k++) d_Ghost_top[id].U[v][k] = 0.0;
            }
        }
        compute_face_values_device(&d_Ghost_top[id]);
    }
    
    if (id < Ny) {
        // --- 左边界 (Left, x = 0) --- 
        for (int v = 0; v < NUM_VARS; v++) {
            d_Ghost_left[id].U[v][0] = Q1[v];
            for (int k = 1; k < N_MODE; k++) d_Ghost_left[id].U[v][k] = 0.0;
        }
        compute_face_values_device(&d_Ghost_left[id]);

        // --- 右边界 (Right, x = 4) --- 零梯度外推
        for (int v = 0; v < NUM_VARS; v++) {
            for (int k = 0; k < N_MODE; k++) {
                d_Ghost_right[id].U[v][k] = d_Mesh[id * Nx + (Nx - 1)].U[v][k];
            }
        }
        compute_face_values_device(&d_Ghost_right[id]);
    }
}

__global__ void compute_boundary_faces_kernel(Element *d_Mesh) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii < Nx && jj < Ny) compute_face_values_device(&d_Mesh[jj * Nx + ii]);
}

__device__ void get_vertex_deriv_device(int ii, int jj, int vert, double out[N_DERIV][NUM_VARS], 
                                        Element *d_Mesh, Element *d_Ghost_bottom, Element *d_Ghost_top, 
                                        Element *d_Ghost_left, Element *d_Ghost_right,
                                        double d_cell_vertex_derivs[Nx*Ny][4][N_DERIV][NUM_VARS]) {
    if (ii >= 0 && ii < Nx && jj >= 0 && jj < Ny) {
        for(int d=0; d<N_DERIV; d++) for(int v=0; v<NUM_VARS; v++)
            out[d][v] = d_cell_vertex_derivs[jj * Nx + ii][vert][d][v];
    } else {
        Element *ghost = NULL;
        if (jj < 0) ghost = &d_Ghost_bottom[ii];
        else if (jj >= Ny) ghost = &d_Ghost_top[ii];
        else if (ii < 0) ghost = &d_Ghost_left[jj];
        else if (ii >= Nx) ghost = &d_Ghost_right[jj];
        
        double xi = (vert == 1 || vert == 2) ? 1.0 : -1.0;
        double eta = (vert == 2 || vert == 3) ? 1.0 : -1.0;
        eval_element_derivatives(ghost, xi, eta, out);
    }
}

__global__ void precompute_vertex_derivs_kernel(Element *d_Mesh, double d_cell_vertex_derivs[Nx*Ny][4][N_DERIV][NUM_VARS]) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii < Nx && jj < Ny) {
        int idx = jj * Nx + ii;
        eval_element_derivatives(&d_Mesh[idx], -1.0, -1.0, d_cell_vertex_derivs[idx][0]);
        eval_element_derivatives(&d_Mesh[idx],  1.0, -1.0, d_cell_vertex_derivs[idx][1]);
        eval_element_derivatives(&d_Mesh[idx],  1.0,  1.0, d_cell_vertex_derivs[idx][2]);
        eval_element_derivatives(&d_Mesh[idx], -1.0,  1.0, d_cell_vertex_derivs[idx][3]);
    }
}

__global__ void compute_damp_coeffs_kernel(Element *d_Mesh, Element *d_Ghost_bottom, Element *d_Ghost_top, 
                                           Element *d_Ghost_left, Element *d_Ghost_right,
                                           double d_cell_vertex_derivs[Nx*Ny][4][N_DERIV][NUM_VARS],
                                           double d_damp_local[Nx*Ny][4]) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= Nx || jj >= Ny) return;
    int idx = jj * Nx + ii;

    Element *C_cell = &d_Mesh[idx];
    double deriv_curr[4][N_DERIV][NUM_VARS], deriv_B[2][N_DERIV][NUM_VARS], deriv_T[2][N_DERIV][NUM_VARS], deriv_L[2][N_DERIV][NUM_VARS], deriv_R[2][N_DERIV][NUM_VARS];    

    get_vertex_deriv_device(ii, jj, 0, deriv_curr[0], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii, jj, 1, deriv_curr[1], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii, jj, 2, deriv_curr[2], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii, jj, 3, deriv_curr[3], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 

    get_vertex_deriv_device(ii, jj - 1, 3, deriv_B[0], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii, jj - 1, 2, deriv_B[1], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii, jj + 1, 0, deriv_T[0], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii, jj + 1, 1, deriv_T[1], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii - 1, jj, 1, deriv_L[0], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii - 1, jj, 2, deriv_L[1], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii + 1, jj, 0, deriv_R[0], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 
    get_vertex_deriv_device(ii + 1, jj, 3, deriv_R[1], d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs); 

    double jump[4][2][N_DERIV][NUM_VARS];
    for(int d=0; d<N_DERIV; d++) {
        for(int v=0; v<NUM_VARS; v++) {
            jump[0][0][d][v] = fabs(deriv_curr[0][d][v] - deriv_L[0][d][v]); 
            jump[0][1][d][v] = fabs(deriv_curr[0][d][v] - deriv_B[0][d][v]);
            jump[1][0][d][v] = fabs(deriv_curr[1][d][v] - deriv_R[0][d][v]); 
            jump[1][1][d][v] = fabs(deriv_curr[1][d][v] - deriv_B[1][d][v]);
            jump[2][0][d][v] = fabs(deriv_curr[2][d][v] - deriv_R[1][d][v]); 
            jump[2][1][d][v] = fabs(deriv_curr[2][d][v] - deriv_T[1][d][v]);
            jump[3][0][d][v] = fabs(deriv_curr[3][d][v] - deriv_L[1][d][v]); 
            jump[3][1][d][v] = fabs(deriv_curr[3][d][v] - deriv_T[0][d][v]);
        }
    }

    double U_mean[NUM_VARS];
    for (int v = 0; v < NUM_VARS; v++) U_mean[v] = C_cell->U[v][0];
    double rho = U_mean[0];
    double rhou = U_mean[1], rhov = U_mean[2], E = U_mean[3];
    double p = calc_pressure(rho, rhou, rhov, E);
    
    double c = safe_sqrt(gamma_gas * p / rho);
    double H = (E + p) / rho;
    double beta_x = fabs(rhou / rho) + c;
    double beta_y = fabs(rhov / rho) + c;

    int order_map[N_DERIV] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3}; 
    double sum_X[4][NUM_VARS] = {0};
    double sum_Y[4][NUM_VARS] = {0};

    for(int d=0; d<N_DERIV; d++) {
        int m = order_map[d];
        for(int s=0; s<NUM_VARS; s++) {
            sum_X[m][s] += 0.5*(jump[0][0][d][s] + jump[3][0][d][s] + jump[1][0][d][s] + jump[2][0][d][s]);
            sum_Y[m][s] += 0.5*(jump[0][1][d][s] + jump[1][1][d][s] + jump[3][1][d][s] + jump[2][1][d][s]);
        }
    }


    double max_contrib[4] = {0.0, 0.0, 0.0, 0.0};

    if (H > 1e-13) {
        for (int m = 0; m <= 3; m++) {
            double current_max = 0.0;
            for (int s = 0; s < NUM_VARS; s++) {
                double term_s = 0.0;
                // 计算该阶次 m 对变量 s 的独立贡献
                if (m == 0) {
                    term_s = (beta_x * sum_X[0][s] + beta_y * sum_Y[0][s]);
                } else if (m == 1) {
                    term_s = (2.0 * dx * beta_x * sum_X[1][s] + 2.0 * dy * beta_y * sum_Y[1][s]);
                } else if (m == 2) {
                    term_s = (6.0 * dx * dx * beta_x * sum_X[2][s] + 6.0 * dy * dy * beta_y * sum_Y[2][s]);
                } else if (m == 3) {
                    // 注意：此处 pow(dx, 3) 对应原代码 (beta_x/dx) * pow(dx, 4) 的化简
                    term_s = (12.0 * dx * dx * dx * beta_x * sum_X[3][s] + 12.0 * dy * dy * dy * beta_y * sum_Y[3][s]);
                }
                
                term_s /= H;
                if (term_s > current_max) current_max = term_s;
            }
            max_contrib[m] = current_max;
        }
    }


    d_damp_local[idx][0] = max_contrib[0];
    double running_sum = max_contrib[0];
    for (int l = 0; l <= 3; l++) {
        running_sum += max_contrib[l];
        d_damp_local[idx][l] = running_sum;
    }

/*
    d_damp_local[idx][0] = 0.0;
    for(int l=1; l<=3; l++) {
        double max_val = 0.0;
        for(int s=0; s<NUM_VARS; s++) {
            double S_X = dx * sum_X[0][s];
            double S_Y = dy * sum_Y[0][s];
            if (l >= 1) { S_X += 2.0 * dx * dx * sum_X[1][s]; S_Y += 2.0 * dy * dy * sum_Y[1][s]; }
            if (l >= 2) { S_X += 6.0 * dx * dx * dx * sum_X[2][s]; S_Y += 6.0 * dy * dy * dy * sum_Y[2][s]; }
            if (l >= 3) { S_X += 12.0 * pow(dx, 4) * sum_X[3][s]; S_Y += 12.0 * pow(dy, 4) * sum_Y[3][s]; }
            double val = 0.0;
            if (H > 1e-13) val = (1.0 / H) * ( (beta_x / dx) * S_X + (beta_y / dy) * S_Y );
            if (val > max_val) max_val = val;
        }
        d_damp_local[idx][l] = max_val;
    }*/
}

__device__ void get_neighbor_face_device(int ii, int jj, int face, double u_plus[NUM_VARS][N_FACE_PTS],
                                         Element *d_Mesh, Element *d_Ghost_bottom, Element *d_Ghost_top, 
                                         Element *d_Ghost_left, Element *d_Ghost_right) {
    for(int var = 0; var < NUM_VARS; var++) {
        if (face == 0) { 
            if (jj == 0) for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Ghost_bottom[ii].face_top[var][k];
            else for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Mesh[(jj - 1) * Nx + ii].face_top[var][k];
        } else if (face == 1) { 
            if (jj == Ny - 1) for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Ghost_top[ii].face_bottom[var][k];
            else for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Mesh[(jj + 1) * Nx + ii].face_bottom[var][k];
        } else if (face == 2) { 
            if (ii == 0) for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Ghost_left[jj].face_right[var][k];
            else for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Mesh[jj * Nx + (ii - 1)].face_right[var][k];
        } else { 
            if (ii == Nx - 1) for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Ghost_right[jj].face_left[var][k];
            else for (int k = 0; k < N_FACE_PTS; k++) u_plus[var][k] = d_Mesh[jj * Nx + (ii + 1)].face_left[var][k];
        }
    }
}

__global__ void compute_rhs_kernel(Element *d_Mesh, Element *d_Ghost_bottom, Element *d_Ghost_top, 
                                   Element *d_Ghost_left, Element *d_Ghost_right,
                                   double d_RHS[Nx*Ny][NUM_VARS][N_MODE]) {
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (ii >= Nx || jj >= Ny) return;
    int idx = jj * Nx + ii;

    Element *cell = &d_Mesh[idx];
    double u_plus[NUM_VARS][N_FACE_PTS];
    double Vol_Int[NUM_VARS][N_MODE] = {{0}};
    double Surf_Int[NUM_VARS][N_MODE] = {{0}};

    for (int q = 0; q < N_QUAD; q++) {
        double U_phys[NUM_VARS] = {0};
        for (int k = 0; k < N_MODE; k++)
            for(int v=0; v<NUM_VARS; v++) U_phys[v] += cell->U[v][k] * d_phi_vol[k][q];
        
        double F_val[NUM_VARS], G_val[NUM_VARS];
        euler_flux(U_phys, F_val, G_val);

        for (int k = 0; k < N_MODE; k++) {
            for(int v=0; v<NUM_VARS; v++) {
                Vol_Int[v][k] += d_w_quad[q] * ( F_val[v] * d_dphi_dr_vol[k][q] * (dy / 2.0) + 
                                                 G_val[v] * d_dphi_ds_vol[k][q] * (dx / 2.0) );
            }
        }
    }

    double U_minus[NUM_VARS], U_p[NUM_VARS], num_f[NUM_VARS];

    get_neighbor_face_device(ii, jj, 0, u_plus, d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right);
    for (int p = 0; p < N_FACE_PTS; p++) {
        for(int v=0; v<NUM_VARS; v++) { U_minus[v] = cell->face_bottom[v][p]; U_p[v] = u_plus[v][p]; }
        llf_flux_vector(U_minus, U_p, 0, -1, num_f);
        for(int k=0; k<N_MODE; k++) for(int v=0; v<NUM_VARS; v++) 
            Surf_Int[v][k] += d_weights_G[p] * num_f[v] * d_phi_face_B[k][p] * (dx / 2.0);
    }
    
    get_neighbor_face_device(ii, jj, 1, u_plus, d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right);
    for (int p = 0; p < N_FACE_PTS; p++) {
        for(int v=0; v<NUM_VARS; v++) { U_minus[v] = cell->face_top[v][p]; U_p[v] = u_plus[v][p]; }
        llf_flux_vector(U_minus, U_p, 0, 1, num_f);
        for(int k=0; k<N_MODE; k++) for(int v=0; v<NUM_VARS; v++) 
            Surf_Int[v][k] += d_weights_G[p] * num_f[v] * d_phi_face_T[k][p] * (dx / 2.0);
    }
    
    get_neighbor_face_device(ii, jj, 2, u_plus, d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right);
    for (int p = 0; p < N_FACE_PTS; p++) {
        for(int v=0; v<NUM_VARS; v++) { U_minus[v] = cell->face_left[v][p]; U_p[v] = u_plus[v][p]; }
        llf_flux_vector(U_minus, U_p, -1, 0, num_f);
        for(int k=0; k<N_MODE; k++) for(int v=0; v<NUM_VARS; v++) 
            Surf_Int[v][k] += d_weights_G[p] * num_f[v] * d_phi_face_L[k][p] * (dy / 2.0);
    }
    
    get_neighbor_face_device(ii, jj, 3, u_plus, d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right);
    for (int p = 0; p < N_FACE_PTS; p++) {
        for(int v=0; v<NUM_VARS; v++) { U_minus[v] = cell->face_right[v][p]; U_p[v] = u_plus[v][p]; }
        llf_flux_vector(U_minus, U_p, 1, 0, num_f);
        for(int k=0; k<N_MODE; k++) for(int v=0; v<NUM_VARS; v++) 
            Surf_Int[v][k] += d_weights_G[p] * num_f[v] * d_phi_face_R[k][p] * (dy / 2.0);
    }

    double J = (dx * dy) / 4.0; 
    for(int v=0; v<NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            d_RHS[idx][v][k] = (Vol_Int[v][k] - Surf_Int[v][k]) * d_M_diag_inv[k] / J;
        }
    }
}

__global__ void update_rk3_stage_kernel(Element *d_Mesh, double d_U0[Nx*Ny][NUM_VARS][N_MODE], 
                                        double d_U_prev[Nx*Ny][NUM_VARS][N_MODE], 
                                        double d_RHS[Nx*Ny][NUM_VARS][N_MODE], int stage, double dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx * Ny) return;
    for (int v = 0; v < NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            if (stage == 1) d_Mesh[id].U[v][k] = d_U0[id][v][k] + dt * d_RHS[id][v][k];
            else if (stage == 2) d_Mesh[id].U[v][k] = 0.75 * d_U0[id][v][k] + 0.25 * (d_U_prev[id][v][k] + dt * d_RHS[id][v][k]);
            else if (stage == 3) d_Mesh[id].U[v][k] = (1.0 / 3.0) * d_U0[id][v][k] + (2.0 / 3.0) * (d_U_prev[id][v][k] + dt * d_RHS[id][v][k]);
        }
    }
}

__global__ void apply_jump_filter_kernel(Element *d_Mesh, double d_damp_local[Nx*Ny][4], double dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx * Ny) return;
    for (int k = 1; k < N_MODE; k++) {
        int l = d_mk_map[k] + d_nk_map[k];
        if (l > 0 && l <= 3) {
            double sigma = d_damp_local[id][l];
            for (int v = 0; v < NUM_VARS; v++) d_Mesh[id].U[v][k] *= exp(-sigma * dt);
        }
    }
}

__global__ void apply_zhang_shu_limiter_kernel(Element *d_Mesh) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx * Ny) return;

    Element *cell = &d_Mesh[id];
    const double eps = 1e-13;
    double Ubar[NUM_VARS];
    for (int v = 0; v < NUM_VARS; v++) Ubar[v] = cell->U[v][0];

    double p_bar = calc_pressure(Ubar[0], Ubar[1], Ubar[2], Ubar[3]);
    
   /*// 强制投影回容许集边界以抵抗极端的数学截断累积误差?这对吗？
    bool mean_modified = false;
    if (Ubar[0] < eps) {
        Ubar[0] = eps;
        cell->U[0][0] = Ubar[0];
        mean_modified = true;
    }
    if (p_bar < eps || mean_modified) {
        Ubar[3] = eps / (gamma_gas - 1.0) + 0.5 * (Ubar[1] * Ubar[1] + Ubar[2] * Ubar[2]) / Ubar[0];
        cell->U[3][0] = Ubar[3];
        p_bar = eps;
    }
*/ 
    // 采用 32 个测试点保证高阶多项式极值测试完整
    double U_test[N_ZS_PTS][NUM_VARS];
    for (int pt = 0; pt < N_ZS_PTS; pt++) {
        for (int v = 0; v < NUM_VARS; v++) U_test[pt][v] = 0.0;
        for (int k = 0; k < N_MODE; k++) {
            for (int v = 0; v < NUM_VARS; v++) {
                U_test[pt][v] += cell->U[v][k] * d_phi_ZS[k][pt];
            }
        }
    }

    double rho_min = U_test[0][0];
    for (int i = 1; i < N_ZS_PTS; i++) if (U_test[i][0] < rho_min) rho_min = U_test[i][0];

    double theta1 = 1.0;
    if (rho_min < eps) {
        theta1 = (Ubar[0] - eps) / (Ubar[0] - rho_min);
        if (theta1 < 0.0) theta1 = 0.0; if (theta1 > 1.0) theta1 = 1.0;
        for (int v = 0; v < NUM_VARS; v++) for (int k = 1; k < N_MODE; k++) cell->U[v][k] *= theta1;
        for (int i = 0; i < N_ZS_PTS; i++) for (int v = 0; v < NUM_VARS; v++) U_test[i][v] = Ubar[v] + theta1 * (U_test[i][v] - Ubar[v]);
    }

    double theta2 = 1.0;
    for (int i = 0; i < N_ZS_PTS; i++) {
        double p_i = calc_pressure(U_test[i][0], U_test[i][1], U_test[i][2], U_test[i][3]);
        if (p_i < eps) {
            double t_L = 0.0, t_R = 1.0;
            for (int iter = 0; iter < 50; iter++) {
                double t_mid = 0.5 * (t_L + t_R);
                double Ut[NUM_VARS];
                for (int v = 0; v < NUM_VARS; v++) Ut[v] = Ubar[v] + t_mid * (U_test[i][v] - Ubar[v]);
                double p_mid = calc_pressure(Ut[0], Ut[1], Ut[2], Ut[3]);
                if (p_mid < eps) t_R = t_mid; else t_L = t_mid;
            }
            if (t_L < theta2) theta2 = t_L;
        }
    }
    if (theta2 < 1.0)
        for (int v = 0; v < NUM_VARS; v++) for (int k = 1; k < N_MODE; k++) cell->U[v][k] *= theta2;
}

__global__ void copy_mesh_to_tmp(Element *d_Mesh, double d_U_tmp[Nx*Ny][NUM_VARS][N_MODE]) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < Nx * Ny) {
        for(int v=0; v<NUM_VARS; v++) for(int k=0; k<N_MODE; k++) d_U_tmp[id][v][k] = d_Mesh[id].U[v][k];
    }
}

__global__ void copy_mesh_to_U0(Element *d_Mesh, double d_U0[Nx*Ny][NUM_VARS][N_MODE]) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < Nx * Ny) {
        for(int v=0; v<NUM_VARS; v++) for(int k=0; k<N_MODE; k++) d_U0[id][v][k] = d_Mesh[id].U[v][k];
    }
}

struct WaveSpeedFunctor {
    Element* mesh;
    WaveSpeedFunctor(Element* _mesh) : mesh(_mesh) {}
    __device__ double operator()(const int& idx) const {
        double rho =mesh[idx].U[0][0]; 
        double rhou = mesh[idx].U[1][0], rhov = mesh[idx].U[2][0], E = mesh[idx].U[3][0];
        double u = rhou / rho, v = rhov / rho;
        double p = calc_pressure(rho, rhou, rhov, E);
        double c = safe_sqrt(gamma_gas * p / rho);
     
        return fmax(fmax(fabs(u) + c, 0.0), fabs(v) + c);
    }
};

static inline double legendre_1d(int i, double x) {
    if (i == 0) return 1.0; 
    if (i == 1) return x; 
    if (i == 2) return 0.5 * (3.0 * x * x - 1.0); 
    if (i == 3) return 0.5 * (5.0 * x * x * x - 3.0 * x);
    return 0.0;
}
static inline double d_legendre_1d(int i, double x) {
    if (i == 0) return 0.0; 
    if (i == 1) return 1.0; 
    if (i == 2) return 3.0 * x; 
    if (i == 3) return 1.5 * (5.0 * x * x - 1.0);
    return 0.0;
}

void init_quadrature_and_matrices(void) {
    for (int j = 0; j < N_FACE_PTS; j++) for (int i = 0; i < N_FACE_PTS; i++) {
        int idx = j * N_FACE_PTS + i;
        r_quad_h[idx] = nodes_G_h[i]; s_quad_h[idx] = nodes_G_h[j]; w_quad_h[idx] = weights_G_h[i] * weights_G_h[j];
    }
    
    for (int k = 0; k < N_MODE; k++) {
        int mk = mk_map_h[k], nk = nk_map_h[k]; 
        M_diag_inv_h[k] = 1.0 / ((2.0 / (2.0 * mk + 1.0)) * (2.0 / (2.0 * nk + 1.0)));
        for (int q = 0; q < N_QUAD; q++) {
            double r = r_quad_h[q], s = s_quad_h[q];
            phi_vol_h[k][q]     = legendre_1d(mk, r) * legendre_1d(nk, s);
            dphi_dr_vol_h[k][q] = d_legendre_1d(mk, r) * legendre_1d(nk, s);
            dphi_ds_vol_h[k][q] = legendre_1d(mk, r) * d_legendre_1d(nk, s);
        }
        for (int p = 0; p < N_FACE_PTS; p++) {
            double np = nodes_G_h[p];
            phi_face_T_h[k][p] = legendre_1d(mk, np) * legendre_1d(nk, 1.0);
            phi_face_B_h[k][p] = legendre_1d(mk, np) * legendre_1d(nk, -1.0);
            phi_face_L_h[k][p] = legendre_1d(mk, -1.0) * legendre_1d(nk, np);
            phi_face_R_h[k][p] = legendre_1d(mk, 1.0) * legendre_1d(nk, np);
        }
    }

    int pt = 0;
    for(int j = 0; j < N_FACE_PTS; j++) { 
        for(int i = 0; i < N_FACE_PTS; i++) {
            double x = nodes_GL_h[i], y = nodes_G_h[j];
            for(int k = 0; k < N_MODE; k++) phi_ZS_h[k][pt] = legendre_1d(mk_map_h[k], x) * legendre_1d(nk_map_h[k], y);
            pt++;
        }
    }
    for(int j = 0; j < N_FACE_PTS; j++) { 
        for(int i = 0; i < N_FACE_PTS; i++) {
            double x = nodes_G_h[i], y = nodes_GL_h[j];
            for(int k = 0; k < N_MODE; k++) phi_ZS_h[k][pt] = legendre_1d(mk_map_h[k], x) * legendre_1d(nk_map_h[k], y);
            pt++;
        }
    }
}


void init_condition(void) {
    for (int jj = 0; jj < Ny; jj++) {
        for (int ii = 0; ii < Nx; ii++) {
            Element *cell = &Mesh_h[jj * Nx + ii];
            double x_center = xL + (ii + 0.5) * dx;
            double y_center = yL + (jj + 0.5) * dy;

            for(int v=0; v<NUM_VARS; v++)
                for(int k=0; k<N_MODE; k++) cell->U[v][k] = 0.0;

            double theta = M_PI / 3.0;
            double shock_X = 1.0 / 6.0 + y_center / tan(theta);
            double rho, u, v, p;

            if (x_center < shock_X) {
                rho = 8.0; 
                u = 4.125 * sqrt(3.0); 
                v = -4.125; 
                p = 116.5;
            } else {
                rho = 1.4; 
                u = 0.0; 
                v = 0.0; 
                p = 1.0;
            }

            double Q[NUM_VARS];
            Q[0] = rho; 
            Q[1] = rho * u; 
            Q[2] = rho * v;
            Q[3] = p / (gamma_gas - 1.0) + 0.5 * rho * (u * u + v * v);

            for(int var = 0; var < NUM_VARS; var++) {
                cell->U[var][0] = Q[var];
            }
        }
    }
}
void output_results(double t) {
    FILE *fp = fopen("result2.dat", "w");
    if (fp == NULL) return;
    fprintf(fp, "VARIABLES = \"X\", \"Y\", \"Rho\", \"U\", \"V\", \"P\"\n");

    for (int jj = 0; jj < Ny; jj++) {
        for (int ii = 0; ii < Nx; ii++) {
            int idx = jj * Nx + ii;
            Element *cell = &Mesh_h[idx]; 
            double xc = (ii + 0.5) * dx, yc = (jj + 0.5) * dy;

            for (int i = 0; i < N_FACE_PTS; i++) {
                for (int j = 0; j < N_FACE_PTS; j++) {
                    double r = nodes_G_h[j], s = nodes_G_h[i]; 
                    double x_phys = xc + (dx / 2.0) * r, y_phys = yc + (dy / 2.0) * s;
                    
                    double U_phys[NUM_VARS] = {0};
                    for (int k = 0; k < N_MODE; k++) {
                        double phi_val = legendre_1d(mk_map_h[k], r) * legendre_1d(nk_map_h[k], s);
                        for (int v = 0; v < NUM_VARS; v++) U_phys[v] += cell->U[v][k] * phi_val;
                    }

                    double rho = U_phys[0];
                    double u = U_phys[1] / rho;
                    double v = U_phys[2] / rho;
                    double p = (gamma_gas - 1.0) * (U_phys[3] - 0.5 * (U_phys[1] * U_phys[1] + U_phys[2] * U_phys[2]) / rho);
                    
                    fprintf(fp, "%lf %lf %lf %lf %lf %lf\n", x_phys, y_phys, rho, u, v, p);
                }
            }
        }
    }
    fclose(fp);
}
// 用于处理简单的 u = u + (dt/6) * RHS 前向推进 (第 1-5, 6-9 步)
__global__ void update_rk4_stage_euler(Element *d_Mesh, double d_RHS[Nx*Ny][NUM_VARS][N_MODE], double dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx * Ny) return;
    for (int v = 0; v < NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            d_Mesh[id].U[v][k] += (dt / 6.0) * d_RHS[id][v][k];
        }
    }
}

// 用于计算并存储中间状态 uII，同时更新 d_Mesh 为第 5 步结束后的修正状态
__global__ void update_rk4_intermediate(Element *d_Mesh, double d_U0[Nx*Ny][NUM_VARS][N_MODE], double d_UII[Nx*Ny][NUM_VARS][N_MODE]) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx * Ny) return;
    for (int v = 0; v < NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            double u_n = d_U0[id][v][k];
            double u_5 = d_Mesh[id].U[v][k];
            // 计算中间状态 uII
            double uII = 0.04 * u_n + 0.36 * u_5;
            d_UII[id][v][k] = uII;
            // 更新当前网格用于后续推进：15*uII - 5*u_5 = 0.6*u_n + 0.4*u_5
            d_Mesh[id].U[v][k] = 15.0 * uII - 5.0 * u_5;
        }
    }
}

// 用于第 10 步的最终组合
__global__ void update_rk4_final(Element *d_Mesh, double d_UII[Nx*Ny][NUM_VARS][N_MODE], double d_RHS[Nx*Ny][NUM_VARS][N_MODE], double dt) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= Nx * Ny) return;
    for (int v = 0; v < NUM_VARS; v++) {
        for (int k = 0; k < N_MODE; k++) {
            double u_9 = d_Mesh[id].U[v][k];
            d_Mesh[id].U[v][k] = d_UII[id][v][k] + 0.6 * u_9 + (dt / 10.0) * d_RHS[id][v][k];
        }
    }
}
int main(void) {
    init_quadrature_and_matrices();
    init_condition();

    cudaMemcpyToSymbol(d_nodes_G, nodes_G_h, sizeof(nodes_G_h));
    cudaMemcpyToSymbol(d_weights_G, weights_G_h, sizeof(weights_G_h));
    cudaMemcpyToSymbol(d_mk_map, mk_map_h, sizeof(mk_map_h));
    cudaMemcpyToSymbol(d_nk_map, nk_map_h, sizeof(nk_map_h));
    cudaMemcpyToSymbol(d_r_quad, r_quad_h, sizeof(r_quad_h));
    cudaMemcpyToSymbol(d_s_quad, s_quad_h, sizeof(s_quad_h));
    cudaMemcpyToSymbol(d_w_quad, w_quad_h, sizeof(w_quad_h));
    cudaMemcpyToSymbol(d_M_diag_inv, M_diag_inv_h, sizeof(M_diag_inv_h));
    cudaMemcpyToSymbol(d_phi_vol, phi_vol_h, sizeof(phi_vol_h));
    cudaMemcpyToSymbol(d_dphi_dr_vol, dphi_dr_vol_h, sizeof(dphi_dr_vol_h));
    cudaMemcpyToSymbol(d_dphi_ds_vol, dphi_ds_vol_h, sizeof(dphi_ds_vol_h));
    cudaMemcpyToSymbol(d_phi_face_T, phi_face_T_h, sizeof(phi_face_T_h));
    cudaMemcpyToSymbol(d_phi_face_B, phi_face_B_h, sizeof(phi_face_B_h));
    cudaMemcpyToSymbol(d_phi_face_L, phi_face_L_h, sizeof(phi_face_L_h));
    cudaMemcpyToSymbol(d_phi_face_R, phi_face_R_h, sizeof(phi_face_R_h));
    cudaMemcpyToSymbol(d_phi_ZS, phi_ZS_h, sizeof(phi_ZS_h));

    Element *d_Mesh, *d_Ghost_bottom, *d_Ghost_top, *d_Ghost_left, *d_Ghost_right;
    cudaMalloc(&d_Mesh, Nx * Ny * sizeof(Element));
    cudaMalloc(&d_Ghost_bottom, Nx * sizeof(Element)); cudaMalloc(&d_Ghost_top, Nx * sizeof(Element));
    cudaMalloc(&d_Ghost_left, Ny * sizeof(Element));   cudaMalloc(&d_Ghost_right, Ny * sizeof(Element));

    double (*d_cell_vertex_derivs)[4][N_DERIV][NUM_VARS], (*d_damp_local)[4];
    cudaMalloc(&d_cell_vertex_derivs, Nx * Ny * sizeof(*d_cell_vertex_derivs));
    cudaMalloc(&d_damp_local, Nx * Ny * sizeof(*d_damp_local));

    double (*d_U0)[NUM_VARS][N_MODE], (*d_U_tmp)[NUM_VARS][N_MODE];
    cudaMalloc(&d_U0, Nx * Ny * sizeof(*d_U0)); 
    cudaMalloc(&d_U_tmp, Nx * Ny * sizeof(*d_U_tmp));
    
    double (*d_RHS)[NUM_VARS][N_MODE];
    cudaMalloc(&d_RHS, Nx * Ny * sizeof(*d_RHS));

    cudaMemcpy(d_Mesh, Mesh_h, Nx * Ny * sizeof(Element), cudaMemcpyHostToDevice);

    dim3 blockSize2D(16, 16);
    dim3 gridSize2D((Nx + 15) / 16, (Ny + 15) / 16);
    int blockSize1D = 256;
    int gridSize1D = (Nx * Ny + 255) / 256;
    int gridMaxBoundary = (fmax(Nx, Ny) + blockSize1D - 1) / blockSize1D; 

    thrust::counting_iterator<int> iter(0);

    double current_time = 0.0; 
    int nit = 0;
    printf("%-10s  %-14s %-14s\n", "Step", "Time", "dt");
    printf("------------------------------------------\n");

    while (current_time < T_END) {
        compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
        apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right,current_time);
        
        double max_wave = thrust::transform_reduce(thrust::device, iter, iter + Nx * Ny, WaveSpeedFunctor(d_Mesh), 0.0, thrust::maximum<double>());
        if (max_wave < 1e-9) max_wave = 1.0;
        
        double dt = CFL * fmin(dx, dy) / max_wave;
        if (current_time + dt > T_END) dt = T_END - current_time;


// 1. 备份 u^n
        copy_mesh_to_U0<<<gridSize1D, blockSize1D>>>(d_Mesh, d_U0);

        // 2. 第 1 到 5 步
        for (int stage = 1; stage <= 5; stage++) {
            compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
            apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, current_time);
            compute_rhs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_RHS);

            update_rk4_stage_euler<<<gridSize1D, blockSize1D>>>(d_Mesh, d_RHS, dt);

            // 每步应用 Filter 和 Limiter
            compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
            apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, current_time);
            precompute_vertex_derivs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_cell_vertex_derivs);
            compute_damp_coeffs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs, d_damp_local);
            apply_jump_filter_kernel<<<gridSize1D, blockSize1D>>>(d_Mesh, d_damp_local, dt);
           apply_zhang_shu_limiter_kernel<<<gridSize1D, blockSize1D>>>(d_Mesh);
        }

        // 3. 计算 uII 并更新网格状态
        update_rk4_intermediate<<<gridSize1D, blockSize1D>>>(d_Mesh, d_U0, d_U_tmp); // 此处 d_U_tmp 被用作 uII

        // 4. 第 6 到 9 步
        for (int stage = 6; stage <= 9; stage++) {
            compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
            apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, current_time);
            compute_rhs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_RHS);

            update_rk4_stage_euler<<<gridSize1D, blockSize1D>>>(d_Mesh, d_RHS, dt);

            compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
            apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, current_time);
            precompute_vertex_derivs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_cell_vertex_derivs);
            compute_damp_coeffs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs, d_damp_local);
            apply_jump_filter_kernel<<<gridSize1D, blockSize1D>>>(d_Mesh, d_damp_local, dt);
            apply_zhang_shu_limiter_kernel<<<gridSize1D, blockSize1D>>>(d_Mesh);
        }

        // 5. 第 10 步 (最终组合)
        compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
        apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, current_time);
        compute_rhs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_RHS);

        update_rk4_final<<<gridSize1D, blockSize1D>>>(d_Mesh, d_U_tmp, d_RHS, dt); // d_U_tmp 保存的是 uII

        // 最终滤波
        compute_boundary_faces_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh);
        apply_ghost_cells_kernel<<<gridMaxBoundary, blockSize1D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, current_time);
        precompute_vertex_derivs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_cell_vertex_derivs);
        compute_damp_coeffs_kernel<<<gridSize2D, blockSize2D>>>(d_Mesh, d_Ghost_bottom, d_Ghost_top, d_Ghost_left, d_Ghost_right, d_cell_vertex_derivs, d_damp_local);
        apply_jump_filter_kernel<<<gridSize1D, blockSize1D>>>(d_Mesh, d_damp_local, dt);
        apply_zhang_shu_limiter_kernel<<<gridSize1D, blockSize1D>>>(d_Mesh);

        current_time += dt; 
        nit++;
        printf("%-10d  %-14.6e %-14.6e \n", nit, current_time, dt);
    }

    cudaMemcpy(Mesh_h, d_Mesh, Nx * Ny * sizeof(Element), cudaMemcpyDeviceToHost);
    printf("GPU Computation Complete.\n");
    output_results(T_END);

    cudaFree(d_Mesh); cudaFree(d_Ghost_bottom); cudaFree(d_Ghost_top); cudaFree(d_Ghost_left); cudaFree(d_Ghost_right);
    cudaFree(d_cell_vertex_derivs); cudaFree(d_damp_local);
    cudaFree(d_U0); cudaFree(d_U_tmp); cudaFree(d_RHS);
    return 0;
}