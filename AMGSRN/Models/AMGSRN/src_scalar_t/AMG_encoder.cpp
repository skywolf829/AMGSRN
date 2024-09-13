#include "AMG_encoder.h"
#include <torch/extension.h>

template <typename scalar_t>
torch::Tensor quaternion_to_rotation_matrix(const torch::Tensor& quaternions) {
    auto qw = quaternions.select(1, 0);
    auto qx = quaternions.select(1, 1);
    auto qy = quaternions.select(1, 2);
    auto qz = quaternions.select(1, 3);

    auto qw2 = qw * qw;
    auto qx2 = qx * qx;
    auto qy2 = qy * qy;
    auto qz2 = qz * qz;

    auto qwx = qw * qx;
    auto qwy = qw * qy;
    auto qwz = qw * qz;
    auto qxy = qx * qy;
    auto qxz = qx * qz;
    auto qyz = qy * qz;

    auto m00 = 1.0 - 2.0 * (qy2 + qz2);
    auto m01 = 2.0 * (qxy - qwz);
    auto m02 = 2.0 * (qxz + qwy);

    auto m10 = 2.0 * (qxy + qwz);
    auto m11 = 1.0 - 2.0 * (qx2 + qz2);
    auto m12 = 2.0 * (qyz - qwx);

    auto m20 = 2.0 * (qxz - qwy);
    auto m21 = 2.0 * (qyz + qwx);
    auto m22 = 1.0 - 2.0 * (qx2 + qy2);

    auto rotation_matrices = torch::stack({m00, m01, m02,
                                           m10, m11, m12,
                                           m20, m21, m22}, 1);

    return rotation_matrices.reshape({-1, 3, 3});
}

torch::Tensor create_transformation_matrices(
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations)
{
    const int num_grids = (int)rotations.size(0);
    auto options = rotations.options();
    torch::Tensor out = torch::empty({num_grids, 4, 4}, options);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_grids + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rotations.scalar_type(), "create_transformation_matrices", ([&] {
        launch_create_transformation_matrices(
            num_grids, 
            rotations.contiguous().data_ptr<scalar_t>(),
            scales.contiguous().data_ptr<scalar_t>(), 
            translations.contiguous().data_ptr<scalar_t>(), 
            out.contiguous().data_ptr<scalar_t>()
        );
    }));

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_transformation_matrices_backward(
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& dL_dMatrix)
{
    const int num_grids = (int)rotations.size(0);
    auto options = rotations.options();

    torch::Tensor dRotations = torch::empty({num_grids, 4}, options);
    torch::Tensor dScales = torch::empty({num_grids, 3}, options);
    torch::Tensor dTranslations = torch::empty({num_grids, 3}, options);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_grids + threadsPerBlock - 1) / threadsPerBlock;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rotations.scalar_type(), "create_transformation_matrices_backward", ([&] {
        launch_create_transformation_matrices_backward(
            num_grids,
            rotations.contiguous().data_ptr<scalar_t>(),
            scales.contiguous().data_ptr<scalar_t>(), 
            translations.contiguous().data_ptr<scalar_t>(), 
            dRotations.contiguous().data_ptr<scalar_t>(),
            dScales.contiguous().data_ptr<scalar_t>(), 
            dTranslations.contiguous().data_ptr<scalar_t>(), 
            dL_dMatrix.contiguous().data_ptr<scalar_t>()
        );
    }));

    return std::make_tuple(dRotations, dScales, dTranslations);
}

torch::Tensor encode_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids)
{
    const auto num_points = query_points.size(0);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto options = query_points.options();

    torch::Tensor out_features = torch::empty({num_points, num_grids*features_per_grid}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "encode_forward", ([&] {
        launch_encode_forward(
            num_points,
            num_grids,
            features_per_grid,
            D, H, W, 
            query_points.transpose(0, 1).contiguous().data_ptr<scalar_t>(), 
            rotations.contiguous().data_ptr<scalar_t>(),
            scales.contiguous().data_ptr<scalar_t>(), 
            translations.contiguous().data_ptr<scalar_t>(), 
            feature_grids.contiguous().data_ptr<scalar_t>(), 
            out_features.contiguous().data_ptr<scalar_t>()
        );
    }));

    return out_features;
}

torch::Tensor encode_backward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids,
    const torch::Tensor& dL_dFeatureVectors)
{
    const auto num_points = query_points.size(0);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto options = query_points.options();

    torch::Tensor dL_dFeatureGrids = torch::zeros({num_grids, features_per_grid, D, H, W}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "encode_backward", ([&] {
        launch_encode_backward(
            num_points,
            num_grids,
            features_per_grid,
            D, H, W, 
            query_points.contiguous().data_ptr<scalar_t>(), 
            rotations.contiguous().data_ptr<scalar_t>(),
            scales.contiguous().data_ptr<scalar_t>(), 
            translations.contiguous().data_ptr<scalar_t>(), 
            feature_grids.contiguous().data_ptr<scalar_t>(), 
            dL_dFeatureVectors.contiguous().data_ptr<scalar_t>(),         
            dL_dFeatureGrids.contiguous().data_ptr<scalar_t>()
        );
    }));

    return dL_dFeatureGrids;
}

torch::Tensor feature_density_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations)
{
    const auto num_points = query_points.size(0);
    const auto num_grids = rotations.size(0);
    auto options = query_points.options();
    torch::Tensor density = torch::empty({num_points, 1}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "feature_density_forward", ([&] {
        launch_density_forward(
            num_points,
            num_grids,
            query_points.contiguous().data_ptr<scalar_t>(), 
            rotations.contiguous().data_ptr<scalar_t>(), 
            scales.contiguous().data_ptr<scalar_t>(), 
            translations.contiguous().data_ptr<scalar_t>(), 
            density.contiguous().data_ptr<scalar_t>());
    }));

    return density;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> feature_density_backward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& dL_dDensity)
{
    const auto num_grids = rotations.size(0);
    const auto num_points = query_points.size(0);
    auto options = rotations.options();

    torch::Tensor dL_dRotations = torch::empty({num_grids, 4}, options);
    torch::Tensor dL_dScales = torch::empty({num_grids, 3}, options);
    torch::Tensor dL_dTranslations = torch::zeros({num_grids, 3}, options);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(rotations.scalar_type(), "feature_density_backward", ([&] {
        launch_density_backward(
            num_points,
            num_grids,
            query_points.contiguous().data_ptr<scalar_t>(), 
            rotations.contiguous().data_ptr<scalar_t>(), 
            scales.contiguous().data_ptr<scalar_t>(), 
            translations.contiguous().data_ptr<scalar_t>(), 
            dL_dDensity.contiguous().data_ptr<scalar_t>(),
            dL_dRotations.contiguous().data_ptr<scalar_t>(),
            dL_dScales.contiguous().data_ptr<scalar_t>(),
            dL_dTranslations.contiguous().data_ptr<scalar_t>()
        );
    }));

    return std::make_tuple(dL_dRotations, dL_dScales, dL_dTranslations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encodeForward", &encode_forward, "Encode positions to feature vectors (forward)");
    m.def("encodeBackward", &encode_backward, "Encode positions to feature vectors (backward)");    
    m.def("featureDensityForward", &feature_density_forward, "Estimate feature density for points (forward)");
    m.def("featureDensityBackward", &feature_density_backward, "Estimate feature density for points (backward)");
    m.def("createTransformationMatricesForward", &create_transformation_matrices, "Create transformation matrices (forward)");
    m.def("createTransformationMatricesBackward", &create_transformation_matrices_backward, "Create transformation matrices (backward)");
}