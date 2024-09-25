#include "AMG_encoder.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
template <typename scalar_t>
struct __align__(16) scalar_t3 {
    scalar_t x;
    scalar_t y;
    scalar_t z;
    scalar_t padding;  // Add padding to ensure 16-byte alignment

    __device__ __host__ scalar_t3() : x(0), y(0), z(0), padding(0) {}
    __device__ __host__ scalar_t3(scalar_t x_, scalar_t y_, scalar_t z_) : x(x_), y(y_), z(z_), padding(0) {}
};
template <typename scalar_t>
__host__ __device__ __forceinline__ scalar_t3<scalar_t> make_scalar_t3(scalar_t x, scalar_t y, scalar_t z) {
    return scalar_t3<scalar_t>(x, y, z);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t3<scalar_t> transformPoint(
    const int grid_idx,
    const scalar_t* rotationMatrices,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits>& translation,
    const scalar_t3<scalar_t>& pos) {
    const int offset = grid_idx * 9;  // 9 elements per 3x3 matrix
    return make_scalar_t3<scalar_t>(
        rotationMatrices[offset + 0] * pos.x + rotationMatrices[offset + 1] * pos.y + rotationMatrices[offset + 2] * pos.z + translation[grid_idx][0],
        rotationMatrices[offset + 3] * pos.x + rotationMatrices[offset + 4] * pos.y + rotationMatrices[offset + 5] * pos.z + translation[grid_idx][1],
        rotationMatrices[offset + 6] * pos.x + rotationMatrices[offset + 7] * pos.y + rotationMatrices[offset + 8] * pos.z + translation[grid_idx][2]
    );
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t3<scalar_t> transformPoint(
    const scalar_t* rotationMatrix,
    const scalar_t* translations,
    const scalar_t3<scalar_t>& pos) {
        
    return make_scalar_t3<scalar_t>(
        rotationMatrix[0] * pos.x + rotationMatrix[1] * pos.y + rotationMatrix[2] * pos.z + translations[0],
        rotationMatrix[3] * pos.x + rotationMatrix[4] * pos.y + rotationMatrix[5] * pos.z + translations[1],
        rotationMatrix[6] * pos.x + rotationMatrix[7] * pos.y + rotationMatrix[8] * pos.z + translations[2]
    );
}


template <typename scalar_t>
__device__ void trilinearInterpolate(
    const int grid_idx,
    const int point_idx,
    const at::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits>& grid,
    const scalar_t3<scalar_t>& point,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output) {

    auto W = grid.size(2);  
    auto H = grid.size(1);
    auto D = grid.size(0);
    auto C = grid.size(4);


    scalar_t x = (W-1) * ((point.x+1.f)/2.f);
    scalar_t y = (H-1) * ((point.y+1.f)/2.f);
    scalar_t z = (D-1) * ((point.z+1.f)/2.f);
    
    if(x <= -1.f || y <= -1.f || z <= -1.f || x >= W || y >= H || z >= D){
        #pragma unroll
        for(int i = 0; i < C; ++i) 
            output[point_idx][grid_idx*C+i] = static_cast<scalar_t>(0);
        return;
    }

    int x0 = floor(x);
    int y0 = floor(y);
    int z0 = floor(z);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    scalar_t xd = x - x0;
    scalar_t yd = y - y0;
    scalar_t zd = z - z0;

    scalar_t w000 = (1-xd)*(1-yd)*(1-zd);
    scalar_t w001 = xd*(1-yd)*(1-zd);
    scalar_t w010 = (1-xd)*yd*(1-zd);
    scalar_t w011 = xd*yd*(1-zd);
    scalar_t w100 = (1-xd)*(1-yd)*zd;
    scalar_t w101 = xd*(1-yd)*zd;
    scalar_t w110 = (1-xd)*yd*zd;
    scalar_t w111 = xd*yd*zd;

    #pragma unroll
    for(int i = 0; i < C; ++i) {
        scalar_t result = static_cast<scalar_t>(0);

        result += (z0 >= 0 && y0 >= 0 && x0 >= 0) ? grid[z0][y0][x0][grid_idx][i] * w000 : static_cast<scalar_t>(0);
        result += (z0 >= 0 && y0 >= 0 && x1 < W) ? grid[z0][y0][x1][grid_idx][i] * w001 : static_cast<scalar_t>(0);
        result += (z0 >= 0 && y1 < H && x0 >= 0) ? grid[z0][y1][x0][grid_idx][i] * w010 : static_cast<scalar_t>(0);
        result += (z0 >= 0 && y1 < H && x1 < W) ? grid[z0][y1][x1][grid_idx][i] * w011 : static_cast<scalar_t>(0);
        result += (z1 < D && y0 >= 0 && x0 >= 0) ? grid[z1][y0][x0][grid_idx][i] * w100 : static_cast<scalar_t>(0);
        result += (z1 < D && y0 >= 0 && x1 < W) ? grid[z1][y0][x1][grid_idx][i] * w101 : static_cast<scalar_t>(0);
        result += (z1 < D && y1 < H && x0 >= 0) ? grid[z1][y1][x0][grid_idx][i] * w110 : static_cast<scalar_t>(0);
        result += (z1 < D && y1 < H && x1 < W) ? grid[z1][y1][x1][grid_idx][i] * w111 : static_cast<scalar_t>(0);

        output[point_idx][grid_idx*C+i] = result;
    }
}

template <typename scalar_t>
__device__ void trilinearInterpolateBackwards(
    const int grid_idx,
    const int point_idx,
    at::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dL_dFeatureGrids,
    const scalar_t3<scalar_t>& point,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dL_dFeatureVector) {
    
    auto C = dL_dFeatureGrids.size(4);
    auto W = dL_dFeatureGrids.size(2);
    auto H = dL_dFeatureGrids.size(1);
    auto D = dL_dFeatureGrids.size(0);

    

    scalar_t x = (W-1) * ((point.x+1.f)/2.f);
    scalar_t y = (H-1) * ((point.y+1.f)/2.f);
    scalar_t z = (D-1) * ((point.z+1.f)/2.f);

    if(x <= -1.f || y <= -1.f || z <= -1.f || x >= W || y >= H || z >= D){
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

    scalar_t w[8] = {
        (1-xd)*(1-yd)*(1-zd),
        xd*(1-yd)*(1-zd),
        (1-xd)*yd*(1-zd),
        xd*yd*(1-zd),
        (1-xd)*(1-yd)*zd,
        xd*(1-yd)*zd,
        (1-xd)*yd*zd,
        xd*yd*zd
    };

    int3 corners[8] = {
        {z0, y0, x0}, {z0, y0, x1}, {z0, y1, x0}, {z0, y1, x1},
        {z1, y0, x0}, {z1, y0, x1}, {z1, y1, x0}, {z1, y1, x1}
    };

    #pragma unroll
    for(int i = 0; i < C; ++i) {
        scalar_t dL_dFeat = dL_dFeatureVector[point_idx][grid_idx*C+i];
        #pragma unroll
        for(int j = 0; j < 8; ++j) {
            int3 c = corners[j];
            if(c.x >= 0 && c.x < D && c.y >= 0 && c.y < H && c.z >= 0 && c.z < W) {
                gpuAtomicAdd(&dL_dFeatureGrids[c.x][c.y][c.z][grid_idx][i], dL_dFeat * w[j]);
            }
        }
    }
}

template <typename scalar_t>
__global__ void encodeForwardKernel(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> query_points,
    const scalar_t* rotation_matrices,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> translations,
    const at::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> feature_grids,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_features) {

    const auto point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto grid_idx = blockIdx.y;

    if (grid_idx >= feature_grids.size(3) || point_idx >= query_points.size(1)) return;

    scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(query_points[0][point_idx], query_points[1][point_idx], query_points[2][point_idx]);

    scalar_t3<scalar_t> point_t = transformPoint<scalar_t>(grid_idx, rotation_matrices, translations, point);
    
    trilinearInterpolate<scalar_t>(
        grid_idx,
        point_idx,
        feature_grids,
        point_t,
        output_features
    );
    
}

template <typename scalar_t>
__global__ void encodeBackwardKernel(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> query_points,
    const scalar_t* rotation_matrices,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> translations,
    const at::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> feature_grids,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dL_dFeatureVectors,
    at::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> dL_dFeatureGrids) {
    
    const auto point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto grid_idx = blockIdx.y;
    if (grid_idx >= feature_grids.size(3) || point_idx >= query_points.size(1)) return;

    __shared__ scalar_t s_rotation[9];
    __shared__ scalar_t s_translation[3];
    if (threadIdx.x < 9) s_rotation[threadIdx.x] = rotation_matrices[grid_idx * 9 + threadIdx.x];
    if (threadIdx.x < 3) s_translation[threadIdx.x] = translations[grid_idx][threadIdx.x];
    __syncthreads();

    scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(query_points[0][point_idx], query_points[1][point_idx], query_points[2][point_idx]);
    scalar_t3<scalar_t> point_t = transformPoint<scalar_t>(s_rotation, s_translation, point);
    
    trilinearInterpolateBackwards<scalar_t>(grid_idx, point_idx, dL_dFeatureGrids, 
        point_t, dL_dFeatureVectors);
    
}

template <typename scalar_t>
__global__ void densityForwardKernel(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> query_points,
    const scalar_t* rotation_matrices,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> translations,
    at::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> output_density) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= query_points.size(0)) return;
    
    scalar_t density = static_cast<scalar_t>(0.0);
    scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(
        query_points[idx][0],
        query_points[idx][1],
        query_points[idx][2]
    );

    for(int i = 0; i < translations.size(0); ++i){
        const scalar_t* rotation_matrix = &rotation_matrices[i * 9];
        scalar_t3<scalar_t> translation = make_scalar_t3<scalar_t>(
            translations[i][0],
            translations[i][1],
            translations[i][2]
        );

        scalar_t3<scalar_t> point_t = make_scalar_t3<scalar_t>(
            rotation_matrix[0] * point.x + rotation_matrix[1] * point.y + rotation_matrix[2] * point.z + translation.x,
            rotation_matrix[3] * point.x + rotation_matrix[4] * point.y + rotation_matrix[5] * point.z + translation.y,
            rotation_matrix[6] * point.x + rotation_matrix[7] * point.y + rotation_matrix[8] * point.z + translation.z
        );

        scalar_t det = rotation_matrix[0] * (rotation_matrix[4]*rotation_matrix[8]-rotation_matrix[5]*rotation_matrix[7]) -
                       rotation_matrix[1] * (rotation_matrix[3]*rotation_matrix[8]-rotation_matrix[5]*rotation_matrix[6]) +
                       rotation_matrix[2] * (rotation_matrix[3]*rotation_matrix[7]-rotation_matrix[4]*rotation_matrix[6]); 
        float x = static_cast<float>(point_t.x);
        float y = static_cast<float>(point_t.y);
        float z = static_cast<float>(point_t.z);
        float g = __expf(-(powf(x, 20.0f) + powf(y, 20.0f) + powf(z, 20.0f)));
        scalar_t g_scalar = static_cast<scalar_t>(g);
        density += det * g_scalar;
    }
    output_density[idx] = density;
}

template <typename scalar_t>
__global__ void densityBackwardKernel(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> query_points,
    const scalar_t* rotation_matrices,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> translations,
    const at::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> dL_dDensity,
    scalar_t* dL_dRotation_matrix,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dL_dTranslations) {
    
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_grads[THREADS_PER_BLOCK * 12];
    __shared__ float shared_sum[12];
    extern __shared__ float sharedMemory[];
    float* R = sharedMemory;
    float* T = sharedMemory + translations.size(0)*9;

    auto s = threadIdx.x*12;

    float3 point;
    float dL_dD;
    if(idx < query_points.size(0)){
        point = make_float3(
            static_cast<float>(query_points[idx][0]),
            static_cast<float>(query_points[idx][1]),
            static_cast<float>(query_points[idx][2])
        );
        dL_dD = static_cast<float>(dL_dDensity[idx]);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < translations.size(0)*12; i+=THREADS_PER_BLOCK){
        auto offset = translations.size(0)*9;
        if(i>=offset){
            auto ind0 = (i-offset)/3;
            auto ind1 = (i-offset)%3;
            T[i-offset] = static_cast<float>(translations[ind0][ind1]);
        }
        else{
            R[i] = static_cast<float>(rotation_matrices[i]);
        }
    }
    __syncthreads();

    for(int i = 0; i<translations.size(0); ++i){
        auto o = i*9;
        if (idx < query_points.size(0)){
            float3 point_t = make_float3(
                R[o + 0] * point.x + R[o + 1] * point.y + R[o + 2] * point.z + T[3*i + 0],
                R[o + 3] * point.x + R[o + 4] * point.y + R[o + 5] * point.z + T[3*i + 1],
                R[o + 6] * point.x + R[o + 7] * point.y + R[o + 8] * point.z + T[3*i + 2]
            );

            float det = R[o + 0] * (R[o + 4]*R[o + 8]-R[o + 5]*R[o + 7]) -
                    R[o + 1] * (R[o + 3]*R[o + 8]-R[o + 5]*R[o + 6]) +
                    R[o + 2] * (R[o + 3]*R[o + 7]-R[o + 4]*R[o + 6]); 
            
            float tx19 = powf(point_t.x, 19.0f);
            float ty19 = powf(point_t.y, 19.0f);
            float tz19 = powf(point_t.z, 19.0f); 

            float g = expf(-(powf(point_t.x, 20.0f) + powf(point_t.y, 20.0f) + powf(point_t.z, 20.0f)));
            float det20g = -20.0f * det * g;

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
            for(int j = 0; j<12; ++j) shared_grads[s+j]=0.0f;
        }
       
        __syncthreads();
        if (threadIdx.x < 12) { 
            shared_sum[threadIdx.x] = 0.0f;
            for (int j = 0; j < THREADS_PER_BLOCK; j++) {
                shared_sum[threadIdx.x] += shared_grads[j * 12 + threadIdx.x];
            }
        }
        __syncthreads();
        if (threadIdx.x < 9) {
            gpuAtomicAdd(&dL_dRotation_matrix[i*9 + threadIdx.x], static_cast<scalar_t>(shared_sum[threadIdx.x]));
        }
        else if(threadIdx.x < 12){
            gpuAtomicAdd(&dL_dTranslations[i][threadIdx.x-9], static_cast<scalar_t>(shared_sum[threadIdx.x]));            
        }
    }
}

template <typename scalar_t>
__global__ void combineTransformationsKernel(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> quaternions,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> scales,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> translations,
    at::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> output) {
        
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quaternions.size(0)) return;

    scalar_t q[4], s[3], t[3];

    // Load the quaternion, scale, and translation for this thread
    for (int i = 0; i < 4; ++i) q[i] = quaternions[idx][i];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx][i];
    for (int i = 0; i < 3; ++i) t[i] = translations[idx][i];

    scalar_t wx = q[0] * q[3];
    scalar_t wy = q[1] * q[3];
    scalar_t wz = q[2] * q[3];
    scalar_t xx = q[0] * q[0];
    scalar_t xy = q[0] * q[1];
    scalar_t xz = q[0] * q[2];
    scalar_t yy = q[1] * q[1];
    scalar_t yz = q[1] * q[2];
    scalar_t zz = q[2] * q[2];

    output[idx][0][0] = s[0] * (1 - 2 * (yy + zz));
    output[idx][0][1] = s[1] * (2 * (xy - wz));
    output[idx][0][2] = s[2] * (2 * (xz + wy));

    output[idx][1][0] = s[0] * (2 * (xy + wz));
    output[idx][1][1] = s[1] * (1 - 2 * (xx + zz));
    output[idx][1][2] = s[2] * (2 * (yz - wx));    

    output[idx][2][0] = s[0] * (2 * (xz - wy));
    output[idx][2][1] = s[1] * (2 * (yz + wx));
    output[idx][2][2] = s[2] * (1 - 2 * (xx + yy));
    // Add the translation column
    output[idx][0][3] = t[0];
    output[idx][1][3] = t[1];
    output[idx][2][3] = t[2];

    // Add the bottom row
    output[idx][3][0] = static_cast<scalar_t>(0);
    output[idx][3][1] = static_cast<scalar_t>(0);
    output[idx][3][2] = static_cast<scalar_t>(0);
    output[idx][3][3] = static_cast<scalar_t>(1);

}

template <typename scalar_t>
__global__ void combineTransformationsKernelBackward(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> quaternions,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> scales,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> translations,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dQuaternions,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dScales,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dTranslations,
    const at::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> dOut) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quaternions.size(0)) return;

    scalar_t q[4], s[3];

    for (int i = 0; i < 4; ++i) q[i] = quaternions[idx][i];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx][i];

    scalar_t wx = q[0] * q[3];
    scalar_t wy = q[1] * q[3];
    scalar_t wz = q[2] * q[3];
    scalar_t xx = q[0] * q[0];
    scalar_t xy = q[0] * q[1];
    scalar_t xz = q[0] * q[2];
    scalar_t yy = q[1] * q[1];
    scalar_t yz = q[1] * q[2];
    scalar_t zz = q[2] * q[2];

    dScales[idx][0] = 
        (dOut[idx][0][0] * (1 - 2 * (yy + zz))) +
        (dOut[idx][1][0] * (2 * (xy + wz))) +
        (dOut[idx][2][0] * (2 * (xz - wy)));
    dScales[idx][1] = 
        (dOut[idx][0][1] * (2 * (xy - wz))) +
        (dOut[idx][1][1] * (1 - 2 * (xx + zz))) +
        (dOut[idx][2][1] * (2 * (yz + wx)));
    dScales[idx][2] = 
        (dOut[idx][0][2] * (2 * (xz + wy))) +
        (dOut[idx][1][2] * (2 * (yz - wx))) +
        (dOut[idx][2][2] * (1 - 2 * (xx + yy)));    
   
    dTranslations[idx][0] = dOut[idx][0][3];
    dTranslations[idx][1] = dOut[idx][1][3];
    dTranslations[idx][2] = dOut[idx][2][3];

    dQuaternions[idx][0] = -4 * q[0] * (dOut[idx][1][1] * s[1] + dOut[idx][2][2] * s[2]) +
                            2 * q[1] * (dOut[idx][0][1] * s[1] + dOut[idx][1][0] * s[0]) +
                            2 * q[2] * (dOut[idx][0][2] * s[2] + dOut[idx][2][0] * s[0]) + 
                            2 * q[3] * (dOut[idx][1][2] * -s[2] + dOut[idx][2][1] * s[1]);
    dQuaternions[idx][1] = 2 * q[0] * (dOut[idx][0][1] * s[1] + dOut[idx][1][0] * s[0]) +
                            -4 * q[1] * (dOut[idx][0][0] * s[0] + dOut[idx][2][2] * s[2]) +
                            2 * q[2] * (dOut[idx][1][2] * s[2] + dOut[idx][2][1] * s[1]) + 
                            2 * q[3] * (dOut[idx][0][2] * s[2] + dOut[idx][2][0] * -s[0]);
    dQuaternions[idx][2] = 2 * q[0] * (dOut[idx][0][2] * s[2] + dOut[idx][2][0] * s[0]) +
                            2 * q[1] * (dOut[idx][1][2] * s[2] + dOut[idx][2][1] * s[1]) +
                            -4 * q[2] * (dOut[idx][0][0] * s[0] + dOut[idx][1][1] * s[1]) + 
                            2 * q[3] * (dOut[idx][0][1] * -s[1] + dOut[idx][1][0] * s[0]);
    dQuaternions[idx][3] = 2 * q[0] * (dOut[idx][1][2] * -s[2] + dOut[idx][2][1] * s[1]) +
                            2 * q[1] * (dOut[idx][0][2] * s[2] + dOut[idx][2][0] * -s[0]) +
                            2 * q[2] * (dOut[idx][0][1] * -s[1] + dOut[idx][1][0] * s[0]); 
}


template <typename scalar_t>
__global__ void quaternionScaleToRotationMatrix(
    const at::PackedTensorAccessor32<scalar_t,2,at::RestrictPtrTraits> quaternions,
    const at::PackedTensorAccessor32<scalar_t,2,at::RestrictPtrTraits> scales,
    scalar_t* output) {
        
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quaternions.size(0)) return;

    scalar_t qx = quaternions[idx][0];
    scalar_t qy = quaternions[idx][1];
    scalar_t qz = quaternions[idx][2];
    scalar_t qw = quaternions[idx][3];
    scalar_t sx = scales[idx][0];
    scalar_t sy = scales[idx][1];
    scalar_t sz = scales[idx][2];

    scalar_t wx = qx * qw;
    scalar_t wy = qy * qw;
    scalar_t wz = qz * qw;

    scalar_t xx = qx * qx;
    scalar_t xy = qx * qy;
    scalar_t xz = qx * qz;

    scalar_t yy = qy * qy;
    scalar_t yz = qy * qz;
    scalar_t zz = qz * qz;

    auto matrix_offset = idx * 9;  // 3x3 matrix for each grid

    output[matrix_offset + 0] = sx * (1 - 2 * (yy + zz));
    output[matrix_offset + 1] = sy * (2 * (xy - wz));
    output[matrix_offset + 2] = sz * (2 * (xz + wy));

    output[matrix_offset + 3] = sx * (2 * (xy + wz));
    output[matrix_offset + 4] = sy * (1 - 2 * (xx + zz));
    output[matrix_offset + 5] = sz * (2 * (yz - wx));    

    output[matrix_offset + 6] = sx * (2 * (xz - wy));
    output[matrix_offset + 7] = sy * (2 * (yz + wx));
    output[matrix_offset + 8] = sz * (1 - 2 * (xx + yy));
}


template <typename scalar_t>
__global__ void rotationMatrixBackward(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> quaternions,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> scales,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dQuaternions,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dScales,
    const scalar_t* dMatrix) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= quaternions.size(0)) return;

    scalar_t qx = quaternions[idx][0];
    scalar_t qy = quaternions[idx][1];
    scalar_t qz = quaternions[idx][2];
    scalar_t qw = quaternions[idx][3];
    scalar_t sx = scales[idx][0];
    scalar_t sy = scales[idx][1];
    scalar_t sz = scales[idx][2];

    scalar_t wx = qx * qw;
    scalar_t wy = qy * qw;
    scalar_t wz = qz * qw;
    scalar_t xx = qx * qx;
    scalar_t xy = qx * qy;
    scalar_t xz = qx * qz;
    scalar_t yy = qy * qy;
    scalar_t yz = qy * qz;
    scalar_t zz = qz * qz;

    int matrix_offset = idx * 9;  // 3x3 matrix for each grid

    dScales[idx][0] = 
        (dMatrix[matrix_offset + 0]*(1 - 2 * (yy + zz))) +
        (dMatrix[matrix_offset + 3]*(2 * (xy + wz)))+
        (dMatrix[matrix_offset + 6]*(2 * (xz - wy)));
    dScales[idx][1] = 
        (dMatrix[matrix_offset + 1]*(2 * (xy - wz))) +
        (dMatrix[matrix_offset + 4]*(1 - 2 * (xx + zz)))+
        (dMatrix[matrix_offset + 7]*(2 * (yz + wx)));
    dScales[idx][2] = 
        (dMatrix[matrix_offset + 2]*(2 * (xz + wy))) +
        (dMatrix[matrix_offset + 5]*(2 * (yz - wx)))+
        (dMatrix[matrix_offset + 8]*(1 - 2 * (xx + yy)));    

    dQuaternions[idx][0] = -4 * qx * (dMatrix[matrix_offset + 4] * sx + dMatrix[matrix_offset + 8] * sz) +
                            2 * qy * (dMatrix[matrix_offset + 1] * sy + dMatrix[matrix_offset + 3] * sx) +
                            2 * qz * (dMatrix[matrix_offset + 2] * sz + dMatrix[matrix_offset + 6] * sx) + 
                            2 * qw * (dMatrix[matrix_offset + 5] * -sz + dMatrix[matrix_offset + 7] * sy);
    dQuaternions[idx][1] =  2 * qx * (dMatrix[matrix_offset + 1] * sy + dMatrix[matrix_offset + 3] * sx) +
                            -4 * qy * (dMatrix[matrix_offset + 0] * sx + dMatrix[matrix_offset + 8] * sz) +
                            2 * qz * (dMatrix[matrix_offset + 5] * sz + dMatrix[matrix_offset + 7] * sy) + 
                            2 * qw * (dMatrix[matrix_offset + 2] * sz + dMatrix[matrix_offset + 6] * -sx);
    dQuaternions[idx][2] =  2 * qx * (dMatrix[matrix_offset + 2] * sz + dMatrix[matrix_offset + 6] * sx) +
                            2 * qy * (dMatrix[matrix_offset + 5] * sz + dMatrix[matrix_offset + 7] * sy) +
                            -4 * qz * (dMatrix[matrix_offset + 0] * sx + dMatrix[matrix_offset + 4] * sy) + 
                            2 * qw * (dMatrix[matrix_offset + 1] * -sy + dMatrix[matrix_offset + 3] * sx);
    dQuaternions[idx][3] =  2 * qx * (dMatrix[matrix_offset + 5] * -sz + dMatrix[matrix_offset + 7] * sy) +
                            2 * qy * (dMatrix[matrix_offset + 2] * sz + dMatrix[matrix_offset + 6] * -sx) +
                            2 * qz * (dMatrix[matrix_offset + 1] * -sy + dMatrix[matrix_offset + 3] * sx); 
}

void launch_create_transformation_matrices(
    const torch::Tensor& quaternions,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    torch::Tensor& output)
{
    auto blocksPerGrid = (quaternions.size(0) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(quaternions.type(), "combineTransformationsKernel", ([&] {
        combineTransformationsKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            quaternions.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
        );
    }));
}


void launch_create_transformation_matrices_backward(
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    torch::Tensor& dRotations,
    torch::Tensor& dScales,
    torch::Tensor& dTranslations,
    const torch::Tensor& dL_dMatrix)
{
    auto blocksPerGrid = (rotations.size(0) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rotations.type(), "combineTransformationsKernelBackward", ([&] {
        combineTransformationsKernelBackward<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
            rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            dRotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            dScales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            dTranslations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            dL_dMatrix.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>()
        );
    }));
}

template <typename scalar_t>
void launch_encode_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids,
    torch::Tensor& output_features)
{
    const int num_points = query_points.size(1);
    const int num_grids = rotations.size(0);

    // Allocate memory for rotation matrices
    scalar_t* rotation_matrices;
    cudaMalloc(&rotation_matrices, num_grids * 3 * 3 * sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );

    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
    dim3 numBlocks((num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_grids);
    encodeForwardKernel<<<numBlocks, threadsPerBlock>>>(
        query_points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices,
        translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        feature_grids.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        output_features.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
    );

    // Free the allocated memory
    cudaFree(rotation_matrices);
}

template <typename scalar_t>
void launch_encode_backward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids,
    const torch::Tensor& dL_dFeature_vectors,
    torch::Tensor& dL_dFeatureGrids)
{
    const int num_points = query_points.size(1);
    const int num_grids = rotations.size(0);

    // Allocate memory for rotation matrices
    scalar_t* rotation_matrices;
    cudaMalloc(&rotation_matrices, num_grids * 3 * 3 * sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );

    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1);
    dim3 numBlocks((num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_grids);
    encodeBackwardKernel<<<numBlocks, threadsPerBlock>>>(
        query_points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices,
        translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        feature_grids.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        dL_dFeature_vectors.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        dL_dFeatureGrids.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()
    );

    // Free the allocated memory
    cudaFree(rotation_matrices);
}

template <typename scalar_t>
void launch_density_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    torch::Tensor& output_density) {

    const int num_points = query_points.size(0);
    const int num_grids = rotations.size(0);

    // First, preprocess rotation matrices
    scalar_t* rotation_matrices;
    cudaMalloc(&rotation_matrices, num_grids * 3 * 3 * sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<scalar_t><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );

    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityForwardKernel<scalar_t><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        query_points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices,
        translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output_density.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
    );

    // Free the allocated memory
    cudaFree(rotation_matrices);
}

template <typename scalar_t>
void launch_density_backward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    const torch::Tensor& dL_dDensity,
    torch::Tensor& dL_dRotations,
    torch::Tensor& dL_dScales,
    torch::Tensor& dL_dTranslations) {

    const int num_points = query_points.size(0);
    const int num_grids = rotations.size(0);

    // First, preprocess rotation matrices    
    scalar_t* rotation_matrices;
    scalar_t* dL_dRotation_matrices;
    cudaMalloc(&rotation_matrices, num_grids * 3 * 3 * sizeof(scalar_t));
    cudaMalloc(&dL_dRotation_matrices, num_grids * 3 * 3 * sizeof(scalar_t));
    cudaMemset(dL_dRotation_matrices, 0, num_grids * 3 * 3 * sizeof(scalar_t));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    quaternionScaleToRotationMatrix<scalar_t><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );
    
    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityBackwardKernel<scalar_t><<<blocksPerGrid, THREADS_PER_BLOCK, num_grids*12*sizeof(float)>>>(
        query_points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices,
        translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        dL_dDensity.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        dL_dRotation_matrices,
        dL_dTranslations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
    );

    blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rotationMatrixBackward<scalar_t><<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        dL_dRotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        dL_dScales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        dL_dRotation_matrices
    );

    // Free the allocated memory
    cudaFree(rotation_matrices);
    cudaFree(dL_dRotation_matrices);
}

template void launch_density_forward<float>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
template void launch_density_forward<double>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
template void launch_density_forward<c10::Half>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);   

template void launch_density_backward<float>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
template void launch_density_backward<double>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
template void launch_density_backward<c10::Half>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);

template void launch_encode_forward<float>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
template void launch_encode_forward<double>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
template void launch_encode_forward<c10::Half>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);

template void launch_encode_backward<float>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
template void launch_encode_backward<double>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);
template void launch_encode_backward<c10::Half>(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&);