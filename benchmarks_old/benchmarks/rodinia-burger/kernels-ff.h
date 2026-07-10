#define idx(i,j)   (i)*y_points+(j)

__global__ 
void core (
    FFArr64 u_new,
    FFArr64 v_new,
    FFArr64 u,
    FFArr64 v,
    const size_t x_points,
    const size_t y_points,
    const double nu,
    const double del_t,
    const double del_x,
    const double del_y)
{
  size_t i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  size_t j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j < x_points - 1 && i < y_points - 1) {
    u_new[idx(i,j)] = u[idx(i,j)] + 
      (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
      (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
      (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
      (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

    v_new[idx(i,j)] = v[idx(i,j)] +
      (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
      (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
      (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
      (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
  }
}

__global__ 
void bound_h (
    FFArr64 u_new,
    FFArr64 v_new,
    const size_t x_points,
    const size_t y_points)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < x_points) {
    u_new[idx(0,i)] = 1.0;
    v_new[idx(0,i)] = 1.0;
    u_new[idx(y_points-1,i)] = 1.0;
    v_new[idx(y_points-1,i)] = 1.0;
  }
}

__global__ 
void bound_v (
    FFArr64 u_new,
    FFArr64 v_new,
    const size_t x_points,
    const size_t y_points)
{
  size_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < y_points) {
    u_new[idx(j,0)] = 1.0;
    v_new[idx(j,0)] = 1.0;
    u_new[idx(j,x_points-1)] = 1.0;
    v_new[idx(j,x_points-1)] = 1.0;
  }
}

__global__ 
void update (
    FFArr64 u,
    FFArr64 v,
    FFArr64 u_new,
    FFArr64 v_new,
    const size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    u[i] = u_new[i];
    v[i] = v_new[i];
  }
}

