#pragma once
#include <torch/extension.h>

void launch_create_transformation_matrices(
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    torch::Tensor& out);

void launch_create_transformation_matrices_backward(
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    torch::Tensor& dRotations,
    torch::Tensor& dScales,
    torch::Tensor& dTranslations,
    const torch::Tensor& dL_dMatrix);

template <typename scalar_t>
void launch_encode_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids,
    torch::Tensor& out_features);

template <typename scalar_t>
void launch_encode_backward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids,
    const torch::Tensor& dL_dFeatureVectors,
    torch::Tensor& dL_dFeatureGrids);

template <typename scalar_t>
void launch_density_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    torch::Tensor& density);

template <typename scalar_t>
void launch_density_backward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& translations,
    const torch::Tensor& dL_dDensity,
    torch::Tensor& dL_dRotations,
    torch::Tensor& dL_dScales,
    torch::Tensor& dL_dTranslations);
