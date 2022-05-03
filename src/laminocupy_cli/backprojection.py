"""
CUDA Raw kernels for computing back-projection
"""

import cupy as cp

source = """
extern "C" {        
    void __global__ adj(float *f, float *data, float *theta, float center, float phi, int s_z, int n, int nz, int deth, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nz)
            return;
        float u = 0;
        float v = 0;
        int ur = 0;
        int vr = 0;        
        
        float f0 = 0;
        float theta0 = 0;
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        float R[6] = {};
        
        for (int t = 0; t<ntheta; t++)
        {
            theta0 = theta[t];            
            float ctheta = __cosf(theta0);
            float stheta = __sinf(theta0);
            R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
            R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
            u = R[0]*(tx-n/2)+R[1]*(ty-n/2) + center;
            v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(tz+s_z) + deth/2;//s_z==nz/2 in the nonchunk case, st_z-heightz else
            
            ur = (int)u;
            vr = (int)v;            
            
            // linear interp            
            if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < deth - 1))
            {
                u = u-ur;
                v = v-vr;                
                f0 +=   data[ur+0+(vr+0)*n+t*n*deth]*(1-u)*(1-v)+
                        data[ur+1+(vr+0)*n+t*n*deth]*(0+u)*(1-v)+
                        data[ur+0+(vr+1)*n+t*n*deth]*(1-u)*(0+v)+
                        data[ur+1+(vr+1)*n+t*n*deth]*(0+u)*(0+v);
                        
            }
        }
        f[tx + ty * n + tz * n * n] += f0*n;        
    }    
    
    void __global__ adj_try(float *f, float *data, float *theta, float* center, float phi, int s_z, int n, int nz, int deth, int ntheta)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nz)
            return;
        float u = 0;
        float v = 0;
        int ur = 0;
        int vr = 0;        
        
        float f0 = 0;
        float theta0 = 0;
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        float R[6] = {};
        
        for (int t = 0; t<ntheta; t++)
        {
            theta0 = theta[t];            
            float ctheta = __cosf(theta0);
            float stheta = __sinf(theta0);
            R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
            R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
            u = R[0]*(tx-n/2)+R[1]*(ty-n/2) + center[tz];
            v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(s_z) + deth/2;//s_z==nz/2 in the nonchunk case, st_z-heightz else
            
            ur = (int)u;
            vr = (int)v;            
            
            // linear interp            
            if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < deth - 1))
            {
                u = u-ur;
                v = v-vr;                
                f0 +=   data[ur+0+(vr+0)*n+t*n*deth]*(1-u)*(1-v)+
                        data[ur+1+(vr+0)*n+t*n*deth]*(0+u)*(1-v)+
                        data[ur+0+(vr+1)*n+t*n*deth]*(1-u)*(0+v)+
                        data[ur+1+(vr+1)*n+t*n*deth]*(0+u)*(0+v);
                        
            }
        }
        f[tx + ty * n + tz * n * n] += f0*n;        
    }    
}
"""

module = cp.RawModule(code=source)
adj_kernel = module.get_function('adj')
adj_try_kernel = module.get_function('adj_try')


def adj(f, data, theta, center, phi, s_z):
    [nz, n] = f.shape[:2]
    [ntheta, deth] = data.shape[:2]    
    adj_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), nz), (32, 32, 1),
               (f, data, theta, cp.float32(center), cp.float32(phi), s_z, n, nz, deth, ntheta))
    return data

def adj_try(f, data, theta, center, phi, s_z):
    [nz, n] = f.shape[:2]
    [ntheta, deth] = data.shape[:2]
    adj_try_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), nz), (32, 32, 1),
               (f, data, theta, center, cp.float32(phi), s_z, n, nz, deth, ntheta))
    return data
