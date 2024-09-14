#include "AMG_encoder.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <type_traits>
#include <cuda_fp16.h>

#define THREADS_PER_BLOCK 256
template<typename scalar_t>
struct scalar_t4 {
    scalar_t x, y, z, w;
    
    __device__ __host__ scalar_t4() : x(0), y(0), z(0), w(0) {}
    __device__ __host__ scalar_t4(scalar_t x_, scalar_t y_, scalar_t z_, scalar_t w_) : x(x_), y(y_), z(z_), w(w_) {}
};

template<typename scalar_t>
__device__ __host__ inline scalar_t4<scalar_t> make_scalar_t4(scalar_t x, scalar_t y, scalar_t z, scalar_t w) {
    return scalar_t4<scalar_t>(x, y, z, w);
}


template<typename scalar_t>
struct scalar_t3 {
    scalar_t x, y, z;
    
    __device__ __host__ scalar_t3() : x(0), y(0), z(0) {}
    __device__ __host__ scalar_t3(scalar_t x_, scalar_t y_, scalar_t z_) : x(x_), y(y_), z(z_) {}
};

template<typename scalar_t>
__device__ __host__ inline scalar_t3<scalar_t> make_scalar_t3(scalar_t x, scalar_t y, scalar_t z) {
    return scalar_t3<scalar_t>(x, y, z);
}


template<typename scalar_t>
__device__ __forceinline__ scalar_t scalar_exp(scalar_t x);

template<>
__device__ __forceinline__ float scalar_exp<float>(float x) {
    return __expf(x);
}

template<>
__device__ __forceinline__ double scalar_exp<double>(double x) {
    return exp(x);
}

template<>
__device__ __forceinline__ half scalar_exp<half>(half x) {
    return hexp(x);
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t scalar_pow(scalar_t base, scalar_t exponent);

template<>
__device__ __forceinline__ float scalar_pow<float>(float base, float exponent) {
    return __powf(base, exponent);
}

template<>
__device__ __forceinline__ double scalar_pow<double>(double base, double exponent) {
    return pow(base, exponent);
}

template<>
__device__ __forceinline__ half scalar_pow<half>(half base, half exponent) {
    // Convert to float for more accurate calculation
    float base_f = __half2float(base);
    float exponent_f = __half2float(exponent);
    
    // Perform pow operation in float
    float result_f = __powf(base_f, exponent_f);
    
    // Convert back to half
    return __float2half(result_f);
}

template<typename scalar_t>
__device__ scalar_t3<scalar_t> transformPoint(
    const scalar_t* rotationMatrix, 
    const scalar_t* translation,
    const scalar_t3<scalar_t> pos) {
    scalar_t3<scalar_t> out = make_scalar_t3<scalar_t>(
        rotationMatrix[0] * pos.x + rotationMatrix[1] * pos.y + rotationMatrix[2] * pos.z + translation[0],
        rotationMatrix[3] * pos.x + rotationMatrix[4] * pos.y + rotationMatrix[5] * pos.z + translation[1],
        rotationMatrix[6] * pos.x + rotationMatrix[7] * pos.y + rotationMatrix[8] * pos.z + translation[2]
    );
    return out;
}

template <typename scalar_t>
__device__ void trilinearInterpolate(
    const scalar_t* grid, 
    const int C,
    const int D, 
    const int H,
    const int W,
    const scalar_t3<scalar_t> point,
    scalar_t* output) {

    // Follows align_corners=True from Torch
    // Rescale from [-1, 1] to [0, W-1], etc.
    scalar_t x = (W-1) * ((point.x+1.f)/2.f);
    scalar_t y = (H-1) * ((point.y+1.f)/2.f);
    scalar_t z = (D-1) * ((point.z+1.f)/2.f);
    
    int x0 = __float2int_rd(x);
    int y0 = __float2int_rd(y);
    int z0 = __float2int_rd(z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    scalar_t xd = x - x0;
    scalar_t yd = y - y0;
    scalar_t zd = z - z0;

    // Pre-compute weights
    scalar_t w000 = (1-xd)*(1-yd)*(1-zd);
    scalar_t w001 = xd*(1-yd)*(1-zd);
    scalar_t w010 = (1-xd)*yd*(1-zd);
    scalar_t w011 = xd*yd*(1-zd);
    scalar_t w100 = (1-xd)*(1-yd)*zd;
    scalar_t w101 = xd*(1-yd)*zd;
    scalar_t w110 = (1-xd)*yd*zd;
    scalar_t w111 = xd*yd*zd;

    // Iterate over each channel
    for(int i = 0; i < C; ++i) {
        scalar_t result = 0.f;
        int base_idx = i*D*H*W;

        // Use ternary operators to avoid branching
        result += (z0 >= 0 && y0 >= 0 && x0 >= 0) ? grid[base_idx + z0*H*W + y0*W + x0] * w000 : 0.f;
        result += (z0 >= 0 && y0 >= 0 && x1 < W) ? grid[base_idx + z0*H*W + y0*W + x1] * w001 : 0.f;
        result += (z0 >= 0 && y1 < H && x0 >= 0) ? grid[base_idx + z0*H*W + y1*W + x0] * w010 : 0.f;
        result += (z0 >= 0 && y1 < H && x1 < W) ? grid[base_idx + z0*H*W + y1*W + x1] * w011 : 0.f;
        result += (z1 < D && y0 >= 0 && x0 >= 0) ? grid[base_idx + z1*H*W + y0*W + x0] * w100 : 0.f;
        result += (z1 < D && y0 >= 0 && x1 < W) ? grid[base_idx + z1*H*W + y0*W + x1] * w101 : 0.f;
        result += (z1 < D && y1 < H && x0 >= 0) ? grid[base_idx + z1*H*W + y1*W + x0] * w110 : 0.f;
        result += (z1 < D && y1 < H && x1 < W) ? grid[base_idx + z1*H*W + y1*W + x1] * w111 : 0.f;

        output[i] = result;
    }
}

template <typename scalar_t>
__device__ void trilinearInterpolateBackwards(
    scalar_t* dL_dFeatureGrids, 
    const int C,
    const int D, 
    const int H,
    const int W,
    const scalar_t3<scalar_t> point,
    const scalar_t* dL_dFeatureVector) {

    
    // Rescale from [-1, 1] to [0, W-1], etc.
    scalar_t x = (W-1) * ((point.x+1.f)/2.f);
    scalar_t y = (H-1) * ((point.y+1.f)/2.f);
    scalar_t z = (D-1) * ((point.z+1.f)/2.f);
    
    // No clamping - return 0 if out of grid bounds.
    if(x <= -1 || y <= -1 || z <= -1 || x >= W || y >= H || z >= D){
        return;
    }

    int x0 = floor(x);
    int x1 = x0 + 1;
    int y0 = floor(y);
    int y1 = y0 + 1;
    int z0 = floor(z);
    int z1 = z0 + 1;

    scalar_t xd = x - x0;
    scalar_t yd = y - y0;
    scalar_t zd = z - z0;

    bool x0_in = (x0 != -1);
    bool x1_in = (x1 != W);
    bool y0_in = (y0 != -1);
    bool y1_in = (y1 != H);
    bool z0_in = (z0 != -1);
    bool z1_in = (z1 != D);

    // Iterate over each channel
    for(int i = 0; i < C; ++i){
        scalar_t dL_dFeat = dL_dFeatureVector[i];
        // Fetch the 8 grid values at corner points
        if(z0_in && y0_in && x0_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z0*H*W + y0*W + x0], dL_dFeat*(1-xd)*(1 - yd)*(1-zd));
        if(z0_in && y0_in && x1_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z0*H*W + y0*W + x1], dL_dFeat*xd*(1 - yd)*(1-zd));
        if(z0_in && y1_in && x0_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z0*H*W + y1*W + x0], dL_dFeat*(1-xd)*yd*(1-zd));
        if(z0_in && y1_in && x1_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z0*H*W + y1*W + x1], dL_dFeat*xd*yd*(1-zd));
        if(z1_in && y0_in && x0_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z1*H*W + y0*W + x0], dL_dFeat*(1-xd)*(1 - yd)*zd);
        if(z1_in && y0_in && x1_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z1*H*W + y0*W + x1], dL_dFeat*xd*(1 - yd)*zd);
        if(z1_in && y1_in && x0_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z1*H*W + y1*W + x0], dL_dFeat*(1-xd)*yd*zd);
        if(z1_in && y1_in && x1_in) atomicAdd(&dL_dFeatureGrids[i*D*H*W + z1*H*W + y1*W + x1], dL_dFeat*xd*yd*zd);
    }
}

template <typename scalar_t>
__global__ void encodeForwardKernel(
    const int num_points, 
    const int num_grids, 
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const scalar_t* __restrict__ query_points, 
    const scalar_t* __restrict__ rotation_matrices, 
    const scalar_t* __restrict__ translations,
    const scalar_t* __restrict__ feature_grids, 
    scalar_t* __restrict__ output_features) {

    const auto point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_idx = blockIdx.y;

    if (grid_idx >= num_grids || point_idx >= num_points) return;

    scalar_t* output_ptr = &output_features[point_idx*num_grids*features_per_grid + features_per_grid*grid_idx];

    // Load the query point into local registers
    scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(
        query_points[point_idx],
        query_points[num_points + point_idx],
        query_points[2 * num_points + point_idx]
    );

    // Load rotation matrix and translation directly from global memory
    scalar_t rotation_matrix[9];
    scalar_t translation[3];
    for (int i = 0; i < 9; ++i) {
        rotation_matrix[i] = rotation_matrices[9*grid_idx + i];
    }
    for (int i = 0; i < 3; ++i) {
        translation[i] = translations[3*grid_idx + i];
    }

    scalar_t3 point_t = transformPoint(rotation_matrix, translation, point);
    
    // Check if the point is in the grid
    if(point_t.x >= -1 && point_t.y >= -1 && point_t.z >= -1 && point_t.x <= 1 && point_t.y <= 1 && point_t.z <= 1){
        trilinearInterpolate(&feature_grids[grid_idx*features_per_grid*D*H*W], 
            features_per_grid, D, H, W, point_t, output_ptr);
    }
    // If the point is out of bounds, set the output for each channel to 0
    else{
        for(int i = 0; i < features_per_grid; ++i) 
            output_ptr[i] = 0.f;
    }
}

template <typename scalar_t>
__global__ void encodeBackwardKernel(
    const int num_points, 
    const int num_grids, 
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const scalar_t* __restrict__ query_points, 
    const scalar_t* rotation_matrices, 
    const scalar_t* translations,
    const scalar_t* feature_grids, 
    const scalar_t* dL_dFeatureVectors, 
    scalar_t* dL_dFeatureGrids) {
    
    __shared__ scalar_t s_rotation_matrices[9];
    __shared__ scalar_t s_translations[3];

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Grab the query point into local registers
    scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(query_points[3 * idx], query_points[3 * idx + 1], query_points[3 * idx + 2]);

    for(auto grid_idx = 0; grid_idx < num_grids; ++grid_idx){
        if (threadIdx.x < 9) {
            s_rotation_matrices[threadIdx.x] = rotation_matrices[9*grid_idx + threadIdx.x];
        }
        if (threadIdx.x < 3) {
            s_translations[threadIdx.x] = translations[3*grid_idx + threadIdx.x];
        }
        __syncthreads();

        // Transform the point into local space for the grid using pre-computed rotation matrix
        scalar_t3 point_t = transformPoint(s_rotation_matrices, s_translations, point);
        trilinearInterpolateBackwards(&dL_dFeatureGrids[grid_idx*features_per_grid*D*H*W], 
            features_per_grid, D, H, W, point_t, 
            &dL_dFeatureVectors[idx*num_grids*features_per_grid + features_per_grid*grid_idx]
        );
    }
}

template <typename scalar_t>
__global__ void densityForwardKernel(
    const int num_points,
    const int num_grids,
    const scalar_t* __restrict__ query_points,
    const scalar_t* rotation_matrices,
    const scalar_t* translations,
    scalar_t* __restrict__ output_density) {

    extern __shared__ char sharedMemory[];
    scalar_t* R = reinterpret_cast<scalar_t*>(sharedMemory);
    scalar_t* T = R + num_grids*9;

    __syncthreads();
    for(int i = threadIdx.x; i < num_grids*12; i+=THREADS_PER_BLOCK){
        // Load translations
        if(i>=num_grids*9){
            T[i-num_grids*9] = translations[i-num_grids*9];
        }
        // Load rotations
        else{
            R[i] = rotation_matrices[i];
        }
    }
    __syncthreads();

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    scalar_t density = static_cast<scalar_t>(0.0);
    scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(query_points[3*idx], query_points[3*idx+1], query_points[3*idx+2]);

    for(int i = 0; i<num_grids; ++i){
        auto o = 9*i;
        scalar_t3<scalar_t> point_t = transformPoint(&R[o], &T[3*i], point);

        scalar_t det = R[o+0] * (R[o+4]*R[o+8]-R[o+5]*R[o+7]) -
                       R[o+1] * (R[o+3]*R[o+8]-R[o+5]*R[o+6]) +
                       R[o+2] * (R[o+3]*R[o+7]-R[o+4]*R[o+6]); 
        scalar_t g = scalar_exp<scalar_t>(-(scalar_pow<scalar_t>(point_t.x, static_cast<scalar_t>(20)) + 
                                            scalar_pow<scalar_t>(point_t.y, static_cast<scalar_t>(20)) + 
                                            scalar_pow<scalar_t>(point_t.z, static_cast<scalar_t>(20))));
        density += det * g;
    }
    output_density[idx] = density;
}

template <typename scalar_t>
__global__ void densityBackwardKernel(
    const int num_points,
    const int num_grids,
    const scalar_t* __restrict__ query_points,
    const scalar_t* __restrict__ rotation_matrices,
    const scalar_t* __restrict__ translations,
    const scalar_t* __restrict__ dL_dDensity,
    scalar_t* dL_dRotation_matrix,
    scalar_t* dL_dTranslations) {
    
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared array for the rotation matrix and translation gradients
    // Reduce total number of atomic adds by putting them here and then
    // aggregating.
    __shared__ scalar_t shared_grads[THREADS_PER_BLOCK * 12];
    __shared__ scalar_t shared_sum[12];
    extern __shared__ scalar_t sharedMemory[];
    scalar_t* R = sharedMemory + 0;
    scalar_t* T = sharedMemory + num_grids*9;

    auto s = threadIdx.x*12;

    scalar_t3<scalar_t> point;
    scalar_t dL_dD;
    if(idx < num_points){
        point = make_scalar_t3<scalar_t>(query_points[idx*3], query_points[3*idx+1], query_points[3*idx+2]);
        dL_dD = dL_dDensity[idx];
    }
    // Use shared memory to load all translations/rotations
    __syncthreads();
    for(int i = threadIdx.x; i < num_grids*12; i+=THREADS_PER_BLOCK){
        // Load translations
        if(i>=num_grids*9){
            T[i-num_grids*9] = translations[i-num_grids*9];
        }
        // Load rotations
        else{
            R[i] = rotation_matrices[i];
        }
    }
    __syncthreads();

    // Gradients for each rotation matrix/translation
    for(int i = 0; i<num_grids; ++i){
        auto o = i*9;
        // Only process if the threadID is within the num_points
        if (idx < num_points){
            // Manual point transformation
            scalar_t3<scalar_t> point_t = transformPoint(&R[o], &T[3*i], point);

            scalar_t det = R[o + 0] * (R[o + 4]*R[o + 8]-R[o + 5]*R[o + 7]) -
                    R[o + 1] * (R[o + 3]*R[o + 8]-R[o + 5]*R[o + 6]) +
                    R[o + 2] * (R[o + 3]*R[o + 7]-R[o + 4]*R[o + 6]); 
            
            scalar_t tx19 = scalar_pow<scalar_t>(point_t.x, static_cast<scalar_t>(19));
            scalar_t ty19 = scalar_pow<scalar_t>(point_t.y, static_cast<scalar_t>(19));
            scalar_t tz19 = scalar_pow<scalar_t>(point_t.z, static_cast<scalar_t>(19)); 

            scalar_t g = scalar_exp<scalar_t>(-(scalar_pow<scalar_t>(point_t.x, static_cast<scalar_t>(20)) + 
                                                scalar_pow<scalar_t>(point_t.y, static_cast<scalar_t>(20)) + 
                                                scalar_pow<scalar_t>(point_t.z, static_cast<scalar_t>(20))));
            scalar_t det20g = static_cast<scalar_t>(-20.0) * det * g;

            //0-8 is rotation matrix grads, 9-12 is translation
            shared_grads[s + 0] = dL_dD*det20g * tx19 * point.x +
                    dL_dD*g * (R[o + 4]*R[o + 8]-R[o + 5]*R[o + 7]);
            shared_grads[s + 1] = dL_dD*det20g * tx19 * point.y +
                    dL_dD*g * -(R[o + 3]*R[o + 8]-R[o + 5]*R[o + 6]); 
            shared_grads[s + 2] = dL_dD*det20g * tx19 * point.z +
                    dL_dD*g * (R[o + 3]*R[o + 7]-R[o + 4]*R[o + 6]); 

            shared_grads[s + 3] = dL_dD*det20g * ty19 * point.x +
                    dL_dD*g * (-R[o + 1]*R[o + 8] + R[o+2]*R[o+7]);
            shared_grads[s + 4] = dL_dD*det20g * ty19 * point.y +
                    dL_dD*g * (R[o+0]*R[o + 8] - R[o+2]*R[o+6]);
            shared_grads[s + 5] = dL_dD*det20g * ty19 * point.z +
                    dL_dD*g * (-R[o+0]*R[o + 7] + R[o+1]*R[o+6]);

            shared_grads[s + 6] = dL_dD*det20g * tz19 * point.x +
                    dL_dD*g * (R[o+1]*R[o + 5] - R[o+2]*R[o+4]);
            shared_grads[s + 7] = dL_dD*det20g * tz19 * point.y +
                    dL_dD*g * (-R[o+0]*R[o + 5] + R[o+2]*R[o+3]);
            shared_grads[s + 8] = dL_dD*det20g * tz19 * point.z +
                    dL_dD*g * (R[o+0]*R[o + 4] - R[o+1]*R[o+3]);

            shared_grads[s + 9] = dL_dD*det20g * tx19;
            shared_grads[s + 10] = dL_dD*det20g * ty19;
            shared_grads[s + 11] = dL_dD*det20g * tz19;
        
        }
        else{
            for(int j = 0; j<12; ++j) shared_grads[s+j]=static_cast<scalar_t>(0.0);
        }
       
        __syncthreads();
        // Reduce shared gradient data via summing every 12th index
        if (threadIdx.x < 12) { 
            shared_sum[threadIdx.x] = static_cast<scalar_t>(0.0);
            for (int j = 0; j < THREADS_PER_BLOCK; j++) {
                shared_sum[threadIdx.x] += shared_grads[j * 12 + threadIdx.x];
            }
        }
        __syncthreads();
        // Only the first thread updates global array
        if (threadIdx.x < 9) {
            atomicAdd(&dL_dRotation_matrix[o+threadIdx.x], shared_sum[threadIdx.x]);
        }
        else if(threadIdx.x < 12){
            atomicAdd(&dL_dTranslations[3*i+threadIdx.x-9], shared_sum[threadIdx.x]);            
        }
    }
}

template <typename scalar_t>
__global__ void combineTransformationsKernel(
    const int numTransforms,
    const scalar_t4<scalar_t>* __restrict__ quaternions, 
    const scalar_t* __restrict__ scales, 
    const scalar_t* __restrict__ translations, 
    scalar_t* __restrict__ output) {
        
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;
    auto o = idx * 16;

    scalar_t4 q;
    scalar_t s[3];
    scalar_t t[3];

    // Load the quaternion, scale, and translation for this thread
    q = quaternions[idx];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx * 3 + i];
    for (int i = 0; i < 3; ++i) t[i] = translations[idx * 3 + i];

    scalar_t wx = q.x * q.w;
    scalar_t wy = q.y * q.w;
    scalar_t wz = q.z * q.w;
    scalar_t xx = q.x * q.x;
    scalar_t xy = q.x * q.y;
    scalar_t xz = q.x * q.z;
    scalar_t yy = q.y * q.y;
    scalar_t yz = q.y * q.z;
    scalar_t zz = q.z * q.z;

    output[o + 0] = s[0] * (1.f - 2.f * (yy + zz));
    output[o + 1] = s[1] * (2.f * (xy - wz));
    output[o + 2] = s[2] * (2.f * (xz + wy));
    output[o + 3] = t[0];

    output[o + 4] = s[0] * (2.f * (xy + wz));
    output[o + 5] = s[1] * (1.f - 2.f * (xx + zz));
    output[o + 6] = s[2] * (2.f * (yz - wx));    
    output[o + 7] = t[1];

    output[o + 8] = s[0] * (2.f * (xz - wy));
    output[o + 9] = s[1] * (2.f * (yz + wx));
    output[o + 10] = s[2] * (1.f - 2.f * (xx + yy));
    output[o + 11] = t[2];

    output[o + 12] = 0.0f;
    output[o + 13] = 0.0f;
    output[o + 14] = 0.0f;
    output[o + 15] = 1.0f;
}


template <typename scalar_t>
__global__ void combineTransformationsKernelBackward(
    const int numTransforms,
    const scalar_t4* __restrict__ quaternions, 
    const scalar_t* __restrict__ scales, 
    const scalar_t* __restrict__ translations, 
    scalar_t4* __restrict__ dQuaternions,
    scalar_t* __restrict__ dScales,
    scalar_t* __restrict__ dTranslations,
    const scalar_t* __restrict__ dOut) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;

    scalar_t4 q;
    scalar_t s[3];

    q = quaternions[idx];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx * 3 + i];

    scalar_t wx = q.x * q.w;
    scalar_t wy = q.y * q.w;
    scalar_t wz = q.z * q.w;
    scalar_t xx = q.x * q.x;
    scalar_t xy = q.x * q.y;
    scalar_t xz = q.x * q.z;
    scalar_t yy = q.y * q.y;
    scalar_t yz = q.y * q.z;
    scalar_t zz = q.z * q.z;

    dScales[idx * 3 + 0] = 
        (dOut[idx * 16 + 0]*(1 - 2 * (yy + zz))) +
        (dOut[idx * 16 + 4]*(2 * (xy + wz)))+
        (dOut[idx * 16 + 8]*(2 * (xz - wy)));
    dScales[idx * 3 + 1] = 
        (dOut[idx * 16 + 1]*(2 * (xy - wz))) +
        (dOut[idx * 16 + 5]*(1 - 2 * (xx + zz)))+
        (dOut[idx * 16 + 9]*(2 * (yz + wx)));
    dScales[idx * 3 + 2] = 
        (dOut[idx * 16 + 2]*(2 * (xz + wy))) +
        (dOut[idx * 16 + 6]*(2 * (yz - wx)))+
        (dOut[idx * 16 + 10]*(1 - 2 * (xx + yy)));    
   
    dTranslations[idx * 3 + 0] = dOut[idx * 16 + 3];
    dTranslations[idx * 3 + 1] = dOut[idx * 16 + 7];
    dTranslations[idx * 3 + 2] = dOut[idx * 16 + 11];

    dQuaternions[idx].x =   -4 * q.x * (dOut[idx * 16 + 5] * s[1] + dOut[idx * 16 + 10] * s[2]) +
                            2 * q.y * (dOut[idx * 16 + 1] * s[1] + dOut[idx * 16 + 4] * s[0]) +
                            2 * q.z * (dOut[idx * 16 + 2] * s[2] + dOut[idx * 16 + 8] * s[0]) + 
                            2 * q.w * (dOut[idx * 16 + 6] * -s[2] + dOut[idx * 16 + 9] * s[1]);
    dQuaternions[idx].y =   2 * q.x * (dOut[idx * 16 + 1] * s[1] + dOut[idx * 16 + 4] * s[0]) +
                            -4 * q.y * (dOut[idx * 16 + 0] * s[0] + dOut[idx * 16 + 10] * s[2]) +
                            2 * q.z * (dOut[idx * 16 + 6] * s[2] + dOut[idx * 16 + 9] * s[1]) + 
                            2 * q.w * (dOut[idx * 16 + 2] * s[2] + dOut[idx * 16 + 8] * -s[0]);
    dQuaternions[idx].z =   2 * q.x * (dOut[idx * 16 + 2] * s[2] + dOut[idx * 16 + 8] * s[0]) +
                            2 * q.y * (dOut[idx * 16 + 6] * s[2] + dOut[idx * 16 + 9] * s[1]) +
                            -4 * q.z * (dOut[idx * 16 + 0] * s[0] + dOut[idx * 16 + 5] * s[1]) + 
                            2 * q.w * (dOut[idx * 16 + 1] * -s[1] + dOut[idx * 16 + 4] * s[0]);
    dQuaternions[idx].w =   2 * q.x * (dOut[idx * 16 + 6] * -s[2] + dOut[idx * 16 + 9] * s[1]) +
                            2 * q.y * (dOut[idx * 16 + 2] * s[2] + dOut[idx * 16 + 8] * -s[0]) +
                            2 * q.z * (dOut[idx * 16 + 1] * -s[1] + dOut[idx * 16 + 4] * s[0]); 

}


__global__ void quaternionScaleToRotationMatrix(
    const int numTransforms,
    const float4* __restrict__ quaternions, 
    const float* __restrict__ scales, 
    float* __restrict__ output) {
        
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;
    auto o = idx*9;

    float4 q = quaternions[idx];
    float sx = scales[idx * 3];
    float sy = scales[idx * 3 + 1];
    float sz = scales[idx * 3 + 2];

    float wx = q.x * q.w;
    float wy = q.y * q.w;
    float wz = q.z * q.w;

    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;

    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float zz = q.z * q.z;

    output[o + 0] = sx * (1.f - 2.f * (yy + zz));
    output[o + 1] = sy * (2.f * (xy - wz));
    output[o + 2] = sz * (2.f * (xz + wy));

    output[o + 3] = sx * (2.f * (xy + wz));
    output[o + 4] = sy * (1.f - 2.f * (xx + zz));
    output[o + 5] = sz * (2.f * (yz - wx));    

    output[o + 6] = sx * (2.f * (xz - wy));
    output[o + 7] = sy * (2.f * (yz + wx));
    output[o + 8] = sz * (1.f - 2.f * (xx + yy));
}


template <typename scalar_t>
__global__ void rotationMatrixBackward(
    const int numTransforms,
    const scalar_t4* __restrict__ quaternions, 
    const scalar_t* __restrict__ scales, 
    scalar_t4* __restrict__ dQuaternions,
    scalar_t* __restrict__ dScales,
    const scalar_t* __restrict__ dMatrix) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;
    auto o = idx * 9;

    scalar_t4 q = quaternions[idx];
    scalar_t sx = scales[idx * 3];
    scalar_t sy = scales[idx * 3 + 1];
    scalar_t sz = scales[idx * 3 + 2];

    scalar_t wx = q.x * q.w;
    scalar_t wy = q.y * q.w;
    scalar_t wz = q.z * q.w;
    scalar_t xx = q.x * q.x;
    scalar_t xy = q.x * q.y;
    scalar_t xz = q.x * q.z;
    scalar_t yy = q.y * q.y;
    scalar_t yz = q.y * q.z;
    scalar_t zz = q.z * q.z;

    dScales[idx * 3 + 0] = 
        (dMatrix[o + 0]*(1 - 2 * (yy + zz))) +
        (dMatrix[o + 3]*(2 * (xy + wz)))+
        (dMatrix[o + 6]*(2 * (xz - wy)));
    dScales[idx * 3 + 1] = 
        (dMatrix[o + 1]*(2 * (xy - wz))) +
        (dMatrix[o + 4]*(1 - 2 * (xx + zz)))+
        (dMatrix[o + 7]*(2 * (yz + wx)));
    dScales[idx * 3 + 2] = 
        (dMatrix[o + 2]*(2 * (xz + wy))) +
        (dMatrix[o + 5]*(2 * (yz - wx)))+
        (dMatrix[o + 8]*(1 - 2 * (xx + yy)));    

    dQuaternions[idx].x =   -4 * q.x * (dMatrix[o + 4] * sx + dMatrix[o + 8] * sz) +
                            2 * q.y * (dMatrix[o + 1] * sy + dMatrix[o + 3] * sx) +
                            2 * q.z * (dMatrix[o + 2] * sz + dMatrix[o + 6] * sx) + 
                            2 * q.w * (dMatrix[o + 5] * -sz + dMatrix[o + 7] * sy);
    dQuaternions[idx].y =   2 * q.x * (dMatrix[o + 1] * sy + dMatrix[o + 3] * sx) +
                            -4 * q.y * (dMatrix[o + 0] * sx + dMatrix[o + 8] * sz) +
                            2 * q.z * (dMatrix[o + 5] * sz + dMatrix[o + 7] * sy) + 
                            2 * q.w * (dMatrix[o + 2] * sz + dMatrix[o + 6] * -sx);
    dQuaternions[idx].z =   2 * q.x * (dMatrix[o + 2] * sz + dMatrix[o + 6] * sx) +
                            2 * q.y * (dMatrix[o + 5] * sz + dMatrix[o + 7] * sy) +
                            -4 * q.z * (dMatrix[o + 0] * sx + dMatrix[o + 4] * sy) + 
                            2 * q.w * (dMatrix[o + 1] * -sy + dMatrix[o + 3] * sx);
    dQuaternions[idx].w =   2 * q.x * (dMatrix[o + 5] * -sz + dMatrix[o + 7] * sy) +
                            2 * q.y * (dMatrix[o + 2] * sz + dMatrix[o + 6] * -sx) +
                            2 * q.z * (dMatrix[o + 1] * -sy + dMatrix[o + 3] * sx); 

}

template <typename scalar_t>
void launch_create_transformation_matrices(
    const int numTransforms,
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    scalar_t* out)
{
    auto blocksPerGrid = (numTransforms + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    combineTransformationsKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        numTransforms,
        (scalar_t4*)rotations,
        scales, 
        translations, 
        out
    );
}

template <typename scalar_t>
void launch_create_transformation_matrices_backward(
    const int numTransforms,
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    scalar_t* dRotations, 
    scalar_t* dScales, 
    scalar_t* dTranslations, 
    const scalar_t* dL_dMatrix)
{
    auto blocksPerGrid = (numTransforms + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    combineTransformationsKernelBackward<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        numTransforms,
        (scalar_t4*)rotations,
        scales, 
        translations, 
        (scalar_t4*)dRotations,
        dScales, 
        dTranslations, 
        dL_dMatrix
    );
}

template <typename scalar_t>
void launch_encode_forward(
    const int num_points,
    const int num_grids,
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const scalar_t* query_points, 
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    const scalar_t* feature_grids, 
    scalar_t* output_features)
{
    // First, preprocess rotation matrices    
    scalar_t* rotation_matrices;
    cudaMalloc((void **)&rotation_matrices, num_grids*3*3*sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (scalar_t4*)rotations, 
        scales, 
        rotation_matrices
    );
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
    dim3 numBlocks((num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_grids);
    encodeForwardKernel<<<numBlocks, threadsPerBlock>>>(
        num_points, num_grids, features_per_grid, D, H, W,
        query_points, rotation_matrices, translations, feature_grids, output_features
    );
    
    cudaFree(rotation_matrices);
}

template <typename scalar_t>
void launch_encode_backward(
    const int num_points,
    const int num_grids,
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const scalar_t* query_points, 
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    const scalar_t* feature_grids, 
    const scalar_t* dL_dFeature_vectors,
    scalar_t* dL_dFeatureGrids)
{
    // First, preprocess rotation matrices    
    scalar_t* rotation_matrices;
    cudaMalloc((void **)&rotation_matrices, num_grids*3*3*sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (scalar_t4*)rotations, 
        scales, 
        rotation_matrices
    );

    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Next, perform interpolation (backward)
    encodeBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_points, 
        num_grids, 
        features_per_grid,
        D, H, W,
        query_points, 
        rotation_matrices, 
        translations,
        feature_grids, 
        dL_dFeature_vectors,
        dL_dFeatureGrids);
    
    cudaFree(rotation_matrices);
}

template <typename scalar_t>
void launch_density_forward(
    const int num_points,
    const int num_grids,
    const scalar_t* query_points, 
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    scalar_t* output_density){

    // First, preprocess rotation matrices    
    scalar_t* rotation_matrices;
    cudaMalloc((void **)&rotation_matrices, num_grids*3*3*sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (scalar_t4*)rotations, 
        scales, 
        rotation_matrices
    );

    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityForwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK, num_grids*12*sizeof(scalar_t)>>>(
        num_points,
        num_grids,
        query_points,
        rotation_matrices,
        translations,
        output_density
    );
    cudaFree(rotation_matrices);
}
template <typename scalar_t>
void launch_density_backward(
    const int num_points,
    const int num_grids,
    const scalar_t* query_points, 
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    const scalar_t* dL_dDensity,
    scalar_t* dL_dRotations,
    scalar_t* dL_dScales,
    scalar_t* dL_dTranslations){

    // First, preprocess rotation matrices    
    scalar_t* rotation_matrices;
    scalar_t* dL_dRotation_matrices;
    size_t size = num_grids*3*3*sizeof(scalar_t);
    cudaMalloc((void **)&rotation_matrices, size);
    cudaMalloc((void **)&dL_dRotation_matrices, size);
    cudaMemset(dL_dRotation_matrices, 0, size);

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (scalar_t4*)rotations, 
        scales, 
        rotation_matrices
    );

    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK, num_grids*12*sizeof(scalar_t)>>>(
        num_points,
        num_grids,
        query_points,
        rotation_matrices,
        translations,
        dL_dDensity,
        dL_dRotation_matrices,
        dL_dTranslations
    );

    blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rotationMatrixBackward<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids,
        (scalar_t4*)rotations,
        scales,
        (scalar_t4*)dL_dRotations,
        dL_dScales,
        dL_dRotation_matrices
    );

    cudaFree(rotation_matrices);
    cudaFree(dL_dRotation_matrices);
}
