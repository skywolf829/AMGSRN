#pragma once
#include <torch/extension.h>


void launch_create_transformation_matrices(
    const int numTransforms,
    const float *rotations, 
    const float *scales, 
    const float *translations,
    float *out);

void launch_create_transformation_matrices_backward(
    const int numTransforms,
    const float *rotations, 
    const float *scales, 
    const float *translations, 
    float *dRotations, 
    float *dScales, 
    float *dTranslations, 
    const float *dL_dMatrix);

void launch_encode_forward(
    const int num_points,
    const int num_grids,
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    const float* feature_grids, 
    float* output_features);

void launch_encode_backward(
    const int num_points,
    const int num_grids,
    const int features_per_grid,
    const int D, 
    const int H, 
    const int W,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    const float* feature_grids, 
    const float* feature_vectors,
    float* dL_dFeatureGrids);

void launch_density_forward(
    const int num_points,
    const int num_grids,
    const float* query_points, 
    const float* rotations, 
    const float* scales, 
    const float* translations, 
    float* output_density);

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
    float* dL_dTranslations);

torch::Tensor quaternion_to_rotation_matrix(const torch::Tensor& q);