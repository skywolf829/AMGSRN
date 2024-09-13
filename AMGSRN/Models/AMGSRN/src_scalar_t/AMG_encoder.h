#pragma once
#include <torch/extension.h>
#include <tuple>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename scalar_t>
void launch_create_transformation_matrices(
    const int numTransforms,
    const scalar_t *rotations, 
    const scalar_t *scales, 
    const scalar_t *translations,
    scalar_t *out);

template <typename scalar_t>
void launch_create_transformation_matrices_backward(
    const int numTransforms,
    const scalar_t *rotations, 
    const scalar_t *scales, 
    const scalar_t *translations, 
    scalar_t *dRotations, 
    scalar_t *dScales, 
    scalar_t *dTranslations, 
    const scalar_t *dL_dMatrix);

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
    scalar_t* output_features);

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
    const scalar_t* feature_vectors,
    scalar_t* dL_dFeatureGrids);

template <typename scalar_t>
void launch_density_forward(
    const int num_points,
    const int num_grids,
    const scalar_t* query_points, 
    const scalar_t* rotations, 
    const scalar_t* scales, 
    const scalar_t* translations, 
    scalar_t* output_density);

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
    scalar_t* dL_dTranslations);

torch::Tensor quaternion_to_rotation_matrix(const torch::Tensor& q);