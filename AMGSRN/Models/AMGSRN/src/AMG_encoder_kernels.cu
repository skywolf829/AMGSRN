#include "AMG_encoder.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    scalar_t3<scalar_t> result;
    
    if constexpr (std::is_same<scalar_t, float>::value) {
        result.x = __fmaf_rn(rotationMatrices[offset + 0], pos.x,
                    __fmaf_rn(rotationMatrices[offset + 1], pos.y,
                        __fmaf_rn(rotationMatrices[offset + 2], pos.z, translation[grid_idx][0])));
        
        result.y = __fmaf_rn(rotationMatrices[offset + 3], pos.x,
                    __fmaf_rn(rotationMatrices[offset + 4], pos.y,
                        __fmaf_rn(rotationMatrices[offset + 5], pos.z, translation[grid_idx][1])));
        
        result.z = __fmaf_rn(rotationMatrices[offset + 6], pos.x,
                    __fmaf_rn(rotationMatrices[offset + 7], pos.y,
                        __fmaf_rn(rotationMatrices[offset + 8], pos.z, translation[grid_idx][2])));
    } else if constexpr (std::is_same<scalar_t, c10::Half>::value) {
        result.x = __hfma(rotationMatrices[offset + 0], pos.x,
                    __hfma(rotationMatrices[offset + 1], pos.y,
                        __hfma(rotationMatrices[offset + 2], pos.z, translation[grid_idx][0])));
        
        result.y = __hfma(rotationMatrices[offset + 3], pos.x,
                    __hfma(rotationMatrices[offset + 4], pos.y,
                        __hfma(rotationMatrices[offset + 5], pos.z, translation[grid_idx][1])));
        
        result.z = __hfma(rotationMatrices[offset + 6], pos.x,
                    __hfma(rotationMatrices[offset + 7], pos.y,
                        __hfma(rotationMatrices[offset + 8], pos.z, translation[grid_idx][2])));
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        result.x = __fma_rn(rotationMatrices[offset + 0], pos.x,
                    __fma_rn(rotationMatrices[offset + 1], pos.y,
                        __fma_rn(rotationMatrices[offset + 2], pos.z, translation[grid_idx][0])));
        
        result.y = __fma_rn(rotationMatrices[offset + 3], pos.x,
                    __fma_rn(rotationMatrices[offset + 4], pos.y,
                        __fma_rn(rotationMatrices[offset + 5], pos.z, translation[grid_idx][1])));
        
        result.z = __fma_rn(rotationMatrices[offset + 6], pos.x,
                    __fma_rn(rotationMatrices[offset + 7], pos.y,
                        __fma_rn(rotationMatrices[offset + 8], pos.z, translation[grid_idx][2])));
    } else {
        result.x = rotationMatrices[offset + 0] * pos.x + rotationMatrices[offset + 1] * pos.y + 
                   rotationMatrices[offset + 2] * pos.z + translation[grid_idx][0];
        result.y = rotationMatrices[offset + 3] * pos.x + rotationMatrices[offset + 4] * pos.y + 
                   rotationMatrices[offset + 5] * pos.z + translation[grid_idx][1];
        result.z = rotationMatrices[offset + 6] * pos.x + rotationMatrices[offset + 7] * pos.y + 
                   rotationMatrices[offset + 8] * pos.z + translation[grid_idx][2];
    }
    
    return result;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t3<scalar_t> transformPoint(
    const scalar_t* rotationMatrix,
    const scalar_t* translations,
    const scalar_t3<scalar_t>& pos) {
        
    scalar_t3<scalar_t> result;

    if constexpr (std::is_same<scalar_t, float>::value) {
        result.x = __fmaf_rn(rotationMatrix[0], pos.x,
                    __fmaf_rn(rotationMatrix[1], pos.y,
                        __fmaf_rn(rotationMatrix[2], pos.z, translations[0])));
        
        result.y = __fmaf_rn(rotationMatrix[3], pos.x,
                    __fmaf_rn(rotationMatrix[4], pos.y,
                        __fmaf_rn(rotationMatrix[5], pos.z, translations[1])));
        
        result.z = __fmaf_rn(rotationMatrix[6], pos.x,
                    __fmaf_rn(rotationMatrix[7], pos.y,
                        __fmaf_rn(rotationMatrix[8], pos.z, translations[2])));
    } else if constexpr (std::is_same<scalar_t, c10::Half>::value) {
        result.x = __hfma(rotationMatrix[0], pos.x,
                    __hfma(rotationMatrix[1], pos.y,
                        __hfma(rotationMatrix[2], pos.z, translations[0])));
        
        result.y = __hfma(rotationMatrix[3], pos.x,
                    __hfma(rotationMatrix[4], pos.y,
                        __hfma(rotationMatrix[5], pos.z, translations[1])));
        
        result.z = __hfma(rotationMatrix[6], pos.x,
                    __hfma(rotationMatrix[7], pos.y,
                        __hfma(rotationMatrix[8], pos.z, translations[2])));
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        result.x = __fma_rn(rotationMatrix[0], pos.x,
                    __fma_rn(rotationMatrix[1], pos.y,
                        __fma_rn(rotationMatrix[2], pos.z, translations[0])));
        
        result.y = __fma_rn(rotationMatrix[3], pos.x,
                    __fma_rn(rotationMatrix[4], pos.y,
                        __fma_rn(rotationMatrix[5], pos.z, translations[1])));
        
        result.z = __fma_rn(rotationMatrix[6], pos.x,
                    __fma_rn(rotationMatrix[7], pos.y,
                        __fma_rn(rotationMatrix[8], pos.z, translations[2])));
    } else {
        result.x = rotationMatrix[0] * pos.x + rotationMatrix[1] * pos.y + 
                   rotationMatrix[2] * pos.z + translations[0];
        result.y = rotationMatrix[3] * pos.x + rotationMatrix[4] * pos.y + 
                   rotationMatrix[5] * pos.z + translations[1];
        result.z = rotationMatrix[6] * pos.x + rotationMatrix[7] * pos.y + 
                   rotationMatrix[8] * pos.z + translations[2];
    }
    
    return result;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t3<scalar_t> transformPoint(
    const scalar_t* rotationMatrix,
    const scalar_t3<scalar_t>& translations,
    const scalar_t3<scalar_t>& pos) {
        
    scalar_t3<scalar_t> result;

    if constexpr (std::is_same<scalar_t, float>::value) {
        result.x = __fmaf_rn(rotationMatrix[0], pos.x,
                    __fmaf_rn(rotationMatrix[1], pos.y,
                        __fmaf_rn(rotationMatrix[2], pos.z, translations.x)));
        
        result.y = __fmaf_rn(rotationMatrix[3], pos.x,
                    __fmaf_rn(rotationMatrix[4], pos.y,
                        __fmaf_rn(rotationMatrix[5], pos.z, translations.y)));
        
        result.z = __fmaf_rn(rotationMatrix[6], pos.x,
                    __fmaf_rn(rotationMatrix[7], pos.y,
                        __fmaf_rn(rotationMatrix[8], pos.z, translations.z)));
    } else if constexpr (std::is_same<scalar_t, c10::Half>::value) {
        result.x = __hfma(rotationMatrix[0], pos.x,
                    __hfma(rotationMatrix[1], pos.y,
                        __hfma(rotationMatrix[2], pos.z, translations.x)));
        
        result.y = __hfma(rotationMatrix[3], pos.x,
                    __hfma(rotationMatrix[4], pos.y,
                        __hfma(rotationMatrix[5], pos.z, translations.y)));
        
        result.z = __hfma(rotationMatrix[6], pos.x,
                    __hfma(rotationMatrix[7], pos.y,
                        __hfma(rotationMatrix[8], pos.z, translations.z)));
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        result.x = __fma_rn(rotationMatrix[0], pos.x,
                    __fma_rn(rotationMatrix[1], pos.y,
                        __fma_rn(rotationMatrix[2], pos.z, translations.x)));
        
        result.y = __fma_rn(rotationMatrix[3], pos.x,
                    __fma_rn(rotationMatrix[4], pos.y,
                        __fma_rn(rotationMatrix[5], pos.z, translations.y)));
        
        result.z = __fma_rn(rotationMatrix[6], pos.x,
                    __fma_rn(rotationMatrix[7], pos.y,
                        __fma_rn(rotationMatrix[8], pos.z, translations.z)));
    } else {
        result.x = rotationMatrix[0] * pos.x + rotationMatrix[1] * pos.y + 
                   rotationMatrix[2] * pos.z + translations.x;
        result.y = rotationMatrix[3] * pos.x + rotationMatrix[4] * pos.y + 
                   rotationMatrix[5] * pos.z + translations.y;
        result.z = rotationMatrix[6] * pos.x + rotationMatrix[7] * pos.y + 
                   rotationMatrix[8] * pos.z + translations.z;
    }
    
    return result;
}


template <typename scalar_t>
__device__ void trilinearInterpolate(
    const int grid_idx,
    const int point_idx,
    const at::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits>& grid,
    const scalar_t3<scalar_t>& point,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output) {

    auto W = grid.size(4);  
    auto H = grid.size(3);
    auto D = grid.size(2);
    auto C = grid.size(1);

    const scalar_t one = static_cast<scalar_t>(1);
    const scalar_t two = static_cast<scalar_t>(2);
    scalar_t x = (W-1) * ((point.x + one) / two);
    scalar_t y = (H-1) * ((point.y + one) / two);
    scalar_t z = (D-1) * ((point.z + one) / two);
    
    if(x <= static_cast<scalar_t>(-1) || y <= static_cast<scalar_t>(-1) || z <= static_cast<scalar_t>(-1) || 
        x >= W || y >= H || z >= D){
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

    #pragma unroll
    for(int i = 0; i < C; ++i) {
        scalar_t result = static_cast<scalar_t>(0);

        result += (z0 >= 0 && y0 >= 0 && x0 >= 0) ? grid[grid_idx][i][z0][y0][x0] * (one-xd)*(one-yd)*(one-zd) : static_cast<scalar_t>(0);
        result += (z0 >= 0 && y0 >= 0 && x1 < W) ? grid[grid_idx][i][z0][y0][x1] * xd*(one-yd)*(one-zd) : static_cast<scalar_t>(0);
        result += (z0 >= 0 && y1 < H && x0 >= 0) ? grid[grid_idx][i][z0][y1][x0] * (one-xd)*yd*(one-zd) : static_cast<scalar_t>(0);
        result += (z0 >= 0 && y1 < H && x1 < W) ? grid[grid_idx][i][z0][y1][x1] * xd*yd*(one-zd) : static_cast<scalar_t>(0);
        result += (z1 < D && y0 >= 0 && x0 >= 0) ? grid[grid_idx][i][z1][y0][x0] * (one-xd)*(one-yd)*zd : static_cast<scalar_t>(0);
        result += (z1 < D && y0 >= 0 && x1 < W) ? grid[grid_idx][i][z1][y0][x1] * xd*(one-yd)*zd : static_cast<scalar_t>(0);
        result += (z1 < D && y1 < H && x0 >= 0) ? grid[grid_idx][i][z1][y1][x0] * (one-xd)*yd*zd : static_cast<scalar_t>(0);
        result += (z1 < D && y1 < H && x1 < W) ? grid[grid_idx][i][z1][y1][x1] * xd*yd*zd : static_cast<scalar_t>(0);

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
    
    auto C = dL_dFeatureGrids.size(1);
    auto W = dL_dFeatureGrids.size(4);
    auto H = dL_dFeatureGrids.size(3);
    auto D = dL_dFeatureGrids.size(2);

    

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
                gpuAtomicAdd(&dL_dFeatureGrids[grid_idx][i][c.x][c.y][c.z], dL_dFeat * w[j]);
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

    if (point_idx >= query_points.size(1)) return;

    const scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(
        query_points[0][point_idx], 
        query_points[1][point_idx], 
        query_points[2][point_idx]);

    __shared__ scalar_t shared_translations[3];
    __shared__ scalar_t shared_rotations[9];

    for (int grid_idx = 0; grid_idx < feature_grids.size(0); ++grid_idx) {
        // Load translations and rotations into shared memory
        if (threadIdx.x < 3) {
            shared_translations[threadIdx.x] = translations[grid_idx][threadIdx.x];
        }
        else if (threadIdx.x < 12) {
            shared_rotations[threadIdx.x-3] = rotation_matrices[grid_idx*9 + threadIdx.x-3];
        }
        __syncthreads();

        const scalar_t3<scalar_t> point_t = transformPoint<scalar_t>(
            shared_rotations,
            shared_translations,
            point
        );
        
        trilinearInterpolate<scalar_t>(
            grid_idx,
            point_idx,
            feature_grids,
            point_t,
            output_features
        );

        __syncthreads();
    }
    
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
    if (point_idx >= query_points.size(1)) return;

    __shared__ scalar_t shared_translations[3];
    __shared__ scalar_t shared_rotations[9];

    const scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(query_points[0][point_idx], query_points[1][point_idx], query_points[2][point_idx]);

    for (int grid_idx = 0; grid_idx < feature_grids.size(0); ++grid_idx) {
        // Load translations and rotations into shared memory
        if (threadIdx.x < 3) {
            shared_translations[threadIdx.x] = translations[grid_idx][threadIdx.x];
        }
        else if (threadIdx.x < 12) {
            shared_rotations[threadIdx.x-3] = rotation_matrices[grid_idx*9 + threadIdx.x-3];
        }
        __syncthreads();

        const scalar_t3<scalar_t> point_t = transformPoint<scalar_t>(shared_rotations, shared_translations, point);
        
        trilinearInterpolateBackwards<scalar_t>(grid_idx, point_idx, dL_dFeatureGrids, 
            point_t, dL_dFeatureVectors);

        __syncthreads();
    }
    
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
    const scalar_t3<scalar_t> point = make_scalar_t3<scalar_t>(
        query_points[idx][0],
        query_points[idx][1],
        query_points[idx][2]
    );

    for(int i = 0; i < translations.size(0); ++i){
        const scalar_t* rotation_matrix = &rotation_matrices[i * 9];
        const scalar_t3<scalar_t> translation = make_scalar_t3<scalar_t>(
            translations[i][0],
            translations[i][1],
            translations[i][2]
        );

        const scalar_t3<scalar_t> point_t = transformPoint<scalar_t>(rotation_matrix, translation, point);

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
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> quaternions,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> scales,
    scalar_t* output) {

    __shared__ scalar_t s_quaternions[4];
    __shared__ scalar_t s_scales[3];

    const int idx = blockIdx.x; // grid idx
    const int tid = threadIdx.x; // thread idx

    // Load data into shared memory
    if (tid < 4) s_quaternions[tid] = quaternions[idx][tid];
    else if (tid < 7) s_scales[tid-4] = scales[idx][tid-4];
    __syncthreads();

    scalar_t qx = s_quaternions[0], qy = s_quaternions[1], qz = s_quaternions[2], qw = s_quaternions[3];
    scalar_t sx = s_scales[0], sy = s_scales[1], sz = s_scales[2];

    scalar_t wx = qx * qw, wy = qy * qw, wz = qz * qw;
    scalar_t xx = qx * qx, xy = qx * qy, xz = qx * qz;
    scalar_t yy = qy * qy, yz = qy * qz, zz = qz * qz;

    // Compute output values
    if (tid == 0) output[idx * 9 + 0] = sx * (1 - 2 * (yy + zz));
    else if (tid == 1) output[idx * 9 + 1] = sy * (2 * (xy - wz));
    else if (tid == 2) output[idx * 9 + 2] = sz * (2 * (xz + wy));
    else if (tid == 3) output[idx * 9 + 3] = sx * (2 * (xy + wz));
    else if (tid == 4) output[idx * 9 + 4] = sy * (1 - 2 * (xx + zz));
    else if (tid == 5) output[idx * 9 + 5] = sz * (2 * (yz - wx));
    else if (tid == 6) output[idx * 9 + 6] = sx * (2 * (xz - wy));
    else if (tid == 7) output[idx * 9 + 7] = sy * (2 * (yz + wx));
    else if (tid == 8) output[idx * 9 + 8] = sz * (1 - 2 * (xx + yy));

}


template <typename scalar_t>
__global__ void rotationMatrixBackward(
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> quaternions,
    const at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> scales,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dQuaternions,
    at::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dScales,
    const scalar_t* dMatrix) {

    __shared__ scalar_t s_quaternions[4];
    __shared__ scalar_t s_scales[3];
    __shared__ scalar_t s_dMatrix[9];

    const int idx = blockIdx.x; // grid idx
    const int tid = threadIdx.x; // thread idx

    // Load data into shared memory
    if (tid < 4) s_quaternions[tid] = quaternions[idx][tid];
    else if (tid < 7) s_scales[tid-4] = scales[idx][tid-4];
    else if (tid < 18) s_dMatrix[tid-11] = dMatrix[idx * 9 + tid-11];
    __syncthreads();

    scalar_t qx = s_quaternions[0], qy = s_quaternions[1], qz = s_quaternions[2], qw = s_quaternions[3];
    scalar_t sx = s_scales[0], sy = s_scales[1], sz = s_scales[2];

    scalar_t wx = qx * qw, wy = qy * qw, wz = qz * qw;
    scalar_t xx = qx * qx, xy = qx * qy, xz = qx * qz;
    scalar_t yy = qy * qy, yz = qy * qz, zz = qz * qz;

    // Compute gradients
    if (tid == 0) {
        dScales[idx][0] = 
            (s_dMatrix[0] * (1 - 2 * (yy + zz))) +
            (s_dMatrix[3] * (2 * (xy + wz))) +
            (s_dMatrix[6] * (2 * (xz - wy)));
    } else if (tid == 1) {
        dScales[idx][1] = 
            (s_dMatrix[1] * (2 * (xy - wz))) +
            (s_dMatrix[4] * (1 - 2 * (xx + zz))) +
            (s_dMatrix[7] * (2 * (yz + wx)));
    } else if (tid == 2) {
        dScales[idx][2] = 
            (s_dMatrix[2] * (2 * (xz + wy))) +
            (s_dMatrix[5] * (2 * (yz - wx))) +
            (s_dMatrix[8] * (1 - 2 * (xx + yy)));
    }
    else if (tid == 3) {
        dQuaternions[idx][0] = -4 * qx * (s_dMatrix[4] * sx + s_dMatrix[8] * sz) +
                                2 * qy * (s_dMatrix[1] * sy + s_dMatrix[3] * sx) +
                                2 * qz * (s_dMatrix[2] * sz + s_dMatrix[6] * sx) + 
                                2 * qw * (s_dMatrix[5] * -sz + s_dMatrix[7] * sy);
    } else if (tid == 4) {
        dQuaternions[idx][1] =  2 * qx * (s_dMatrix[1] * sy + s_dMatrix[3] * sx) +
                                -4 * qy * (s_dMatrix[0] * sx + s_dMatrix[8] * sz) +
                                2 * qz * (s_dMatrix[5] * sz + s_dMatrix[7] * sy) + 
                                2 * qw * (s_dMatrix[2] * sz + s_dMatrix[6] * -sx);
    } else if (tid == 5) {
        dQuaternions[idx][2] =  2 * qx * (s_dMatrix[2] * sz + s_dMatrix[6] * sx) +
                                2 * qy * (s_dMatrix[5] * sz + s_dMatrix[7] * sy) +
                                -4 * qz * (s_dMatrix[0] * sx + s_dMatrix[4] * sy) + 
                                2 * qw * (s_dMatrix[1] * -sy + s_dMatrix[3] * sx);
    } else if (tid == 6) {
        dQuaternions[idx][3] =  2 * qx * (s_dMatrix[5] * -sz + s_dMatrix[7] * sy) +
                                2 * qy * (s_dMatrix[2] * sz + s_dMatrix[6] * -sx) +
                                2 * qz * (s_dMatrix[1] * -sy + s_dMatrix[3] * sx);
    }
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

    dim3 threadsPerBlock(9);
    dim3 numBlocks(num_grids);
    quaternionScaleToRotationMatrix<<<numBlocks, threadsPerBlock>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );
    threadsPerBlock.x = THREADS_PER_BLOCK;
    threadsPerBlock.y = 1;
    numBlocks.x = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    numBlocks.y = 1;
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

    dim3 threadsPerBlock(9);
    dim3 numBlocks(num_grids);
    quaternionScaleToRotationMatrix<<<numBlocks, threadsPerBlock>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );

    threadsPerBlock.x = THREADS_PER_BLOCK;
    threadsPerBlock.y = 1;
    numBlocks.x = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    numBlocks.y = 1;
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

    dim3 threadsPerBlock(9);
    dim3 numBlocks(num_grids);
    quaternionScaleToRotationMatrix<scalar_t><<<numBlocks, threadsPerBlock>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );

    dim3 blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
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

    dim3 threadsPerBlock(9);
    dim3 numBlocks(num_grids);
    quaternionScaleToRotationMatrix<scalar_t><<<numBlocks, threadsPerBlock>>>(
        rotations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        scales.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices
    );
    
    dim3 blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityBackwardKernel<scalar_t><<<blocksPerGrid, THREADS_PER_BLOCK, num_grids*12*sizeof(float)>>>(
        query_points.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        rotation_matrices,
        translations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        dL_dDensity.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
        dL_dRotation_matrices,
        dL_dTranslations.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
    );

    threadsPerBlock.x = 18;
    threadsPerBlock.y = 1;
    numBlocks.x = num_grids;
    numBlocks.y = 1;
    rotationMatrixBackward<scalar_t><<<numBlocks, threadsPerBlock>>>(
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