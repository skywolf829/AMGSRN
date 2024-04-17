#include "AMG_encoder.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_PER_BLOCK 256

__device__ float3 transformPoint(
    const float* rotationMatrix, 
    const float* translation,
    const float3 pos) {
    float3 out = make_float3(
        rotationMatrix[0] * pos.x + rotationMatrix[1] * pos.y + rotationMatrix[2] * pos.z + translation[0],
        rotationMatrix[3] * pos.x + rotationMatrix[4] * pos.y + rotationMatrix[5] * pos.z + translation[1],
        rotationMatrix[6] * pos.x + rotationMatrix[7] * pos.y + rotationMatrix[8] * pos.z + translation[2]
    );
    return out;
}

__device__ void trilinearInterpolate(
    const float* grid, 
    const int C,
    const int D, 
    const int H,
    const int W,
    const float3 point,
    const int start_out_ind,
    const int feature_vector_length,
    float* output) {

    // Follows align_corners=True from Torch
    // Rescale from [-1, 1] to [0, W-1], etc.
    float x = (W-1) * ((point.x+1.f)/2.f);
    float y = (H-1) * ((point.y+1.f)/2.f);
    float z = (D-1) * ((point.z+1.f)/2.f);
    
    // 0 if truly out of bounds including zero padding
    if(x <= -1 || y <= -1 || z <= -1 || x >= W || y >= H || z >= D){
        return;
    }
    int x0 = floor(x);
    int x1 = x0 + 1;
    int y0 = floor(y);
    int y1 = y0 + 1;
    int z0 = floor(z);
    int z1 = z0 + 1;

    float xd = x - x0;
    float yd = y - y0;
    float zd = z - z0;

    bool x0_in = (x0 != -1);
    bool x1_in = (x1 != W);
    bool y0_in = (y0 != -1);
    bool y1_in = (y1 != H);
    bool z0_in = (z0 != -1);
    bool z1_in = (z1 != D);

    for(auto i = 0; i < C; ++i){
        auto o = i*H*W*D;
        float c000 = z0_in && y0_in && x0_in ? grid[o + z0 * H * W + y0 * W + x0] : 0;
        float c001 = z0_in && y0_in && x1_in ? grid[o + z0 * H * W + y0 * W + x1] : 0;
        float c010 = z0_in && y1_in && x0_in ? grid[o + z0 * H * W + y1 * W + x0] : 0;
        float c011 = z0_in && y1_in && x1_in ? grid[o + z0 * H * W + y1 * W + x1] : 0;
        float c100 = z1_in && y0_in && x0_in ? grid[o + z1 * H * W + y0 * W + x0] : 0;
        float c101 = z1_in && y0_in && x1_in ? grid[o + z1 * H * W + y0 * W + x1] : 0;
        float c110 = z1_in && y1_in && x0_in ? grid[o + z1 * H * W + y1 * W + x0] : 0;
        float c111 = z1_in && y1_in && x1_in ? grid[o + z1 * H * W + y1 * W + x1] : 0;

        // Interpolate
        float c00 = c000 * (1 - xd) + c001 * xd;
        float c01 = c010 * (1 - xd) + c011 * xd;
        float c10 = c100 * (1 - xd) + c101 * xd;
        float c11 = c110 * (1 - xd) + c111 * xd;

        float c0 = c00 * (1 - yd) + c01 * yd;
        float c1 = c10 * (1 - yd) + c11 * yd;

        float c = c0 * (1 - zd) + c1 * zd;
        atomicAdd(&output[(start_out_ind + i) % feature_vector_length], c);
    }
}

__device__ void trilinearInterpolateBackwards(
    float* dL_dFeatureGrids, 
    const int C,
    const int D, 
    const int H,
    const int W,
    const int feature_vector_length,
    const int grid_idx,
    const float3 point,
    const float* dL_dFeatureVector) {

    
    // Rescale from [-1, 1] to [0, W-1], etc.
    float x = (W-1) * ((point.x+1.f)/2.f);
    float y = (H-1) * ((point.y+1.f)/2.f);
    float z = (D-1) * ((point.z+1.f)/2.f);
    
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

    float xd = x - x0;
    float yd = y - y0;
    float zd = z - z0;

    bool x0_in = (x0 != -1);
    bool x1_in = (x1 != W);
    bool y0_in = (y0 != -1);
    bool y1_in = (y1 != H);
    bool z0_in = (z0 != -1);
    bool z1_in = (z1 != D);

    // Iterate over each channel
    for(int i = 0; i < C; ++i){
        float dL_dFeat = dL_dFeatureVector[(grid_idx*C+i)%feature_vector_length];
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

__global__ void encodeForwardKernel(
    const int num_points, 
    const int num_grids, 
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const int feature_vector_length,
    const float* __restrict__ query_points, 
    const float* rotation_matrices, 
    const float* translations,
    const float* __restrict__ feature_grids, 
    float* output_features) {

    extern __shared__ float sharedGridData[];

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_idx = blockIdx.y;

    __syncthreads();
    for(auto i = threadIdx.x; i < features_per_grid*D*H*W; i += THREADS_PER_BLOCK){
        auto s = grid_idx*features_per_grid*D*H*W + i;
        sharedGridData[i] = feature_grids[s];
    }
    __syncthreads();
    if (idx >= num_points) return;

    // Grab the query point into local registers
    float3 point = make_float3(query_points[3 * idx],
        query_points[3 * idx + 1], 
        query_points[3 * idx + 2]);

    // Transform the point into local space for the grid using pre-computed rotation matrix
    float3 point_t = transformPoint(&rotation_matrices[9*grid_idx], &translations[3*grid_idx], point);
    // Perform feature grid trilinear interpolation and write the result directly to the output
    trilinearInterpolate(sharedGridData, 
        features_per_grid,
        D, H, W, point_t, 
        grid_idx*features_per_grid, feature_vector_length,
        &output_features[idx*feature_vector_length]);
        
    
}

__global__ void encodeBackwardKernel(
    const int num_points, 
    const int num_grids, 
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const int feature_vector_length,
    const float* __restrict__ query_points, 
    const float* rotation_matrices, 
    const float* translations,
    const float* feature_grids, 
    const float* __restrict__ dL_dFeatureVectors, 
    float* dL_dFeatureGrids) {
    
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto grid_idx = blockIdx.y;
    if (idx >= num_points) return;

    // Grab the query point into local registers
    float3 point = make_float3(query_points[3 * idx],query_points[3 * idx + 1], query_points[3 * idx + 2]);

    // Transform the point into local space for the grid using pre-computed rotation matrix
    float3 point_t = transformPoint(&rotation_matrices[9*grid_idx], &translations[3*grid_idx], point);
    // Perform feature grid trilinear interpolation and write the result directly to the output
    trilinearInterpolateBackwards(&dL_dFeatureGrids[grid_idx*features_per_grid*D*H*W], 
        features_per_grid, D, H, W, feature_vector_length, grid_idx, point_t, 
        &dL_dFeatureVectors[idx*feature_vector_length]
    );
    
}

__global__ void densityForwardKernel(
    const int num_points,
    const int num_grids,
    const float* __restrict__ query_points,
    const float* rotation_matrices,
    const float* translations,
    float* __restrict__ output_density) {

    extern __shared__ float sharedMemory[];
    float* R = sharedMemory + 0;
    float* T = sharedMemory + num_grids*9;

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

    float density = 0.0f;
    float3 point = make_float3(query_points[3*idx], query_points[3*idx+1], query_points[3*idx+2]);

    for(int i = 0; i<num_grids; ++i){
        auto o = 9*i;
        float3 point_t = transformPoint(&R[o], &T[3*i], point);

        float det = R[o+0] * (R[o+4]*R[o+8]-R[o+5]*R[o+7]) -
                   R[o+1] * (R[o+3]*R[o+8]-R[o+5]*R[o+6]) +
                   R[o+2] * (R[o+3]*R[o+7]-R[o+4]*R[o+6]); 
        float g = expf(-(powf(point_t.x, 20) + powf(point_t.y, 20) + powf(point_t.z, 20)));
        density += det * g;
    }
    output_density[idx] = density;
}

__global__ void densityBackwardKernel(
    const int num_points,
    const int num_grids,
    const float* __restrict__ query_points,
    const float* __restrict__ rotation_matrices,
    const float* __restrict__ translations,
    const float* __restrict__ dL_dDensity,
    float* dL_dRotation_matrix,
    float* dL_dTranslations) {
    
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared array for the rotation matrix and translation gradients
    // Reduce total number of atomic adds by putting them here and then
    // aggregating.
    __shared__ float shared_grads[THREADS_PER_BLOCK * 12];
    __shared__ float shared_sum[12];
    extern __shared__ float sharedMemory[];
    float* R = sharedMemory + 0;
    float* T = sharedMemory + num_grids*9;

    auto s = threadIdx.x*12;

    float3 point;
    float dL_dD;
    if(idx < num_points){
        point = make_float3(query_points[idx*3], query_points[3*idx+1], query_points[3*idx+2]);
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

    // Gradients for each rotaiton matrix/translation
    for(int i = 0; i<num_grids; ++i){
        auto o = i*9;
        // Only process if the threadID is within the num_points
        if (idx < num_points){
            // Manual point transformation
            float3 point_t = transformPoint(&R[o], &T[3*i], point);

            float det = R[o + 0] * (R[o + 4]*R[o + 8]-R[o + 5]*R[o + 7]) -
                    R[o + 1] * (R[o + 3]*R[o + 8]-R[o + 5]*R[o + 6]) +
                    R[o + 2] * (R[o + 3]*R[o + 7]-R[o + 4]*R[o + 6]); 
            
            float tx19 = powf(point_t.x, 19);
            float ty19 = powf(point_t.y, 19);
            float tz19 = powf(point_t.z, 19); 

            float g = expf(-(powf(point_t.x, 20) + powf(point_t.y, 20) + powf(point_t.z, 20)));
            float det20g = -20.f * det * g;

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
            for(int j = 0; j<12; ++j) shared_grads[s+j]=0.f;
        }
       
        __syncthreads();
        // Reduce shared gradient data via summing every 12th index
        if (threadIdx.x < 12) { 
            shared_sum[threadIdx.x] = 0.f;
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

__global__ void combineTransformationsKernel(
    const int numTransforms,
    const float4* __restrict__ quaternions, 
    const float* __restrict__ scales, 
    const float* __restrict__ translations, 
    float* __restrict__ output) {
        
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;
    auto o = idx * 16;

    float4 q;
    float s[3];
    float t[3];

    // Load the quaternion, scale, and translation for this thread
    q = quaternions[idx];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx * 3 + i];
    for (int i = 0; i < 3; ++i) t[i] = translations[idx * 3 + i];

    float wx = q.x * q.w;
    float wy = q.y * q.w;
    float wz = q.z * q.w;
    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float zz = q.z * q.z;

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


__global__ void combineTransformationsKernelBackward(
    const int numTransforms,
    const float4* __restrict__ quaternions, 
    const float* __restrict__ scales, 
    const float* __restrict__ translations, 
    float4* __restrict__ dQuaternions,
    float* __restrict__ dScales,
    float* __restrict__ dTranslations,
    const float* __restrict__ dOut) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;

    float4 q;
    float s[3];

    q = quaternions[idx];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx * 3 + i];

    float wx = q.x * q.w;
    float wy = q.y * q.w;
    float wz = q.z * q.w;
    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float zz = q.z * q.z;

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

    float4 q;
    float s[3];

    // Load the quaternion, scale, and translation for this thread
    q = quaternions[idx];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx * 3 + i];

    float wx = q.x * q.w;
    float wy = q.y * q.w;
    float wz = q.z * q.w;

    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;

    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float zz = q.z * q.z;

    output[o + 0] = s[0] * (1.f - 2.f * (yy + zz));
    output[o + 1] = s[1] * (2.f * (xy - wz));
    output[o + 2] = s[2] * (2.f * (xz + wy));

    output[o + 3] = s[0] * (2.f * (xy + wz));
    output[o + 4] = s[1] * (1.f - 2.f * (xx + zz));
    output[o + 5] = s[2] * (2.f * (yz - wx));    

    output[o + 6] = s[0] * (2.f * (xz - wy));
    output[o + 7] = s[1] * (2.f * (yz + wx));
    output[o + 8] = s[2] * (1.f - 2.f * (xx + yy));
}


__global__ void rotationMatrixBackward(
    const int numTransforms,
    const float4* __restrict__ quaternions, 
    const float* __restrict__ scales, 
    float4* __restrict__ dQuaternions,
    float* __restrict__ dScales,
    const float* __restrict__ dMatrix) {

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTransforms) return;
    auto o = idx * 9;

    float4 q;
    float s[3];

    q = quaternions[idx];
    for (int i = 0; i < 3; ++i) s[i] = scales[idx * 3 + i];

    float wx = q.x * q.w;
    float wy = q.y * q.w;
    float wz = q.z * q.w;
    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float zz = q.z * q.z;

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

    dQuaternions[idx].x =   -4 * q.x * (dMatrix[o + 4] * s[1] + dMatrix[o + 8] * s[2]) +
                            2 * q.y * (dMatrix[o + 1] * s[1] + dMatrix[o + 3] * s[0]) +
                            2 * q.z * (dMatrix[o + 2] * s[2] + dMatrix[o + 6] * s[0]) + 
                            2 * q.w * (dMatrix[o + 5] * -s[2] + dMatrix[o + 7] * s[1]);
    dQuaternions[idx].y =   2 * q.x * (dMatrix[o + 1] * s[1] + dMatrix[o + 3] * s[0]) +
                            -4 * q.y * (dMatrix[o + 0] * s[0] + dMatrix[o + 8] * s[2]) +
                            2 * q.z * (dMatrix[o + 5] * s[2] + dMatrix[o + 7] * s[1]) + 
                            2 * q.w * (dMatrix[o + 2] * s[2] + dMatrix[o + 6] * -s[0]);
    dQuaternions[idx].z =   2 * q.x * (dMatrix[o + 2] * s[2] + dMatrix[o + 6] * s[0]) +
                            2 * q.y * (dMatrix[o + 5] * s[2] + dMatrix[o + 7] * s[1]) +
                            -4 * q.z * (dMatrix[o + 0] * s[0] + dMatrix[o + 4] * s[1]) + 
                            2 * q.w * (dMatrix[o + 1] * -s[1] + dMatrix[o + 3] * s[0]);
    dQuaternions[idx].w =   2 * q.x * (dMatrix[o + 5] * -s[2] + dMatrix[o + 7] * s[1]) +
                            2 * q.y * (dMatrix[o + 2] * s[2] + dMatrix[o + 6] * -s[0]) +
                            2 * q.z * (dMatrix[o + 1] * -s[1] + dMatrix[o + 3] * s[0]); 

}

void launch_create_transformation_matrices(
    const int numTransforms,
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    float* out)
{
    auto blocksPerGrid = (numTransforms + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    combineTransformationsKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        numTransforms,
        (float4*)rotations,
        scales, 
        translations, 
        out
    );
}

void launch_create_transformation_matrices_backward(
    const int numTransforms,
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    float* dRotations, 
    float* dScales, 
    float* dTranslations, 
    const float* dL_dMatrix)
{
    auto blocksPerGrid = (numTransforms + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    combineTransformationsKernelBackward<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        numTransforms,
        (float4*)rotations,
        scales, 
        translations, 
        (float4*)dRotations,
        dScales, 
        dTranslations, 
        dL_dMatrix
    );
}

void launch_encode_forward(
    const int num_points,
    const int num_grids,
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const int feature_vector_length,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    const float* feature_grids, 
    float* output_features)
{
    // First, preprocess rotation matrices    
    float* rotation_matrices;
    cudaMalloc((void **)&rotation_matrices, num_grids*3*3*sizeof(float));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (float4*)rotations, 
        scales, 
        rotation_matrices
    );

    dim3 gridDims((num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_grids);
    // Next, perform interpolation
    encodeForwardKernel<<<gridDims, THREADS_PER_BLOCK, sizeof(float)*D*H*W*features_per_grid>>>(
        num_points, 
        num_grids, 
        features_per_grid,
        D, H, W,
        feature_vector_length,
        query_points, 
        rotation_matrices, 
        translations,
        feature_grids, 
        output_features);
    
    cudaFree(rotation_matrices);
}

void launch_encode_backward(
    const int num_points,
    const int num_grids,
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const int feature_vector_length,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    const float* feature_grids, 
    const float* dL_dFeature_vectors,
    float* dL_dFeatureGrids)
{
    // First, preprocess rotation matrices    
    float* rotation_matrices;
    cudaMalloc((void **)&rotation_matrices, num_grids*3*3*sizeof(float));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (float4*)rotations, 
        scales, 
        rotation_matrices
    );

    dim3 gridDims((num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_grids);
    // Next, perform interpolation (backward)
    encodeBackwardKernel<<<gridDims, THREADS_PER_BLOCK>>>(
        num_points, 
        num_grids, 
        features_per_grid,
        D, H, W,
        feature_vector_length,
        query_points, 
        rotation_matrices, 
        translations,
        feature_grids, 
        dL_dFeature_vectors,
        dL_dFeatureGrids);
    
    cudaFree(rotation_matrices);
}

void launch_density_forward(
    const int num_points,
    const int num_grids,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    float* output_density){

    // First, preprocess rotation matrices    
    float* rotation_matrices;
    cudaMalloc((void **)&rotation_matrices, num_grids*3*3*sizeof(float));

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (float4*)rotations, 
        scales, 
        rotation_matrices
    );

    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityForwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK, num_grids*12*sizeof(float)>>>(
        num_points,
        num_grids,
        query_points,
        rotation_matrices,
        translations,
        output_density
    );
    cudaFree(rotation_matrices);
}


void launch_density_backward(
    const int num_points,
    const int num_grids,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    const float* dL_dDensity,
    float* dL_dRotations,
    float* dL_dScales,
    float* dL_dTranslations){

    // First, preprocess rotation matrices    
    float* rotation_matrices;
    float* dL_dRotation_matrices;
    size_t size = num_grids*3*3*sizeof(float);
    cudaMalloc((void **)&rotation_matrices, size);
    cudaMalloc((void **)&dL_dRotation_matrices, size);
    cudaMemset(dL_dRotation_matrices, 0, size);

    auto blocksPerGrid = (num_grids + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    quaternionScaleToRotationMatrix<<<blocksPerGrid, THREADS_PER_BLOCK>>>(
        num_grids, 
        (float4*)rotations, 
        scales, 
        rotation_matrices
    );

    blocksPerGrid = (num_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    densityBackwardKernel<<<blocksPerGrid, THREADS_PER_BLOCK, num_grids*12*sizeof(float)>>>(
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
        (float4*)rotations,
        scales,
        (float4*)dL_dRotations,
        dL_dScales,
        dL_dRotation_matrices
    );

    cudaFree(rotation_matrices);
    cudaFree(dL_dRotation_matrices);
}
