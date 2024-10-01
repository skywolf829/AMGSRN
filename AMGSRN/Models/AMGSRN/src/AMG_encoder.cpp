#include "AMG_encoder.h"

torch::Tensor create_transformation_matrices(
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations)
{
    const int num_grids = rotations.size(0);
    auto options = torch::TensorOptions().dtype(rotations.dtype()).device(rotations.device());
    torch::Tensor out = torch::empty({num_grids, 4, 4}, options);

    launch_create_transformation_matrices(
        rotations,
        scales, 
        translations, 
        out
    );

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_transformation_matrices_backward(
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& dL_dMatrix)
{
    const int num_grids = rotations.size(0);
    auto options = torch::TensorOptions().dtype(rotations.dtype()).device(rotations.device());
    torch::Tensor dRotations = torch::empty({num_grids, 4}, options);
    torch::Tensor dScales = torch::empty({num_grids, 3}, options);
    torch::Tensor dTranslations = torch::empty({num_grids, 3}, options);

    launch_create_transformation_matrices_backward(
        rotations,
        scales, 
        translations, 
        dRotations,
        dScales, 
        dTranslations, 
        dL_dMatrix
    );

    return std::make_tuple(dRotations, dScales, dTranslations);
}

torch::Tensor encode_forward(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids)
{
    const auto num_points = query_points.size(1);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto options = torch::TensorOptions().dtype(query_points.dtype()).device(query_points.device());

    torch::Tensor out_features = torch::empty({num_points, num_grids*features_per_grid}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "launch_encode_forward", ([&] {
        launch_encode_forward<scalar_t>(
            query_points, 
            rotations,
            scales, 
            translations, 
            feature_grids, 
            out_features
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
    const auto num_points = query_points.size(1);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto options = torch::TensorOptions().dtype(query_points.dtype()).device(query_points.device());

    torch::Tensor dL_dFeatureGrids = torch::zeros({num_grids, features_per_grid, D, H, W}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "launch_encode_backward", ([&] {
        launch_encode_backward<scalar_t>(
            query_points, 
            rotations,
            scales, 
            translations, 
            feature_grids, 
            dL_dFeatureVectors,         
            dL_dFeatureGrids
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
    auto options = torch::TensorOptions().dtype(query_points.dtype()).device(query_points.device());
    torch::Tensor density = torch::empty({num_points}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "launch_density_forward", ([&] {
        launch_density_forward<scalar_t>(
            query_points, 
            rotations, 
            scales, 
            translations, 
            density
        );
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
    auto options = torch::TensorOptions().dtype(query_points.dtype()).device(query_points.device());

    torch::Tensor dL_dRotations = torch::empty({num_grids, 4}, options);
    torch::Tensor dL_dScales = torch::empty({num_grids, 3}, options);
    torch::Tensor dL_dTranslations = torch::zeros({num_grids, 3}, options);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_points.scalar_type(), "launch_density_backward", ([&] {
        launch_density_backward<scalar_t>(
            query_points, 
            rotations, 
            scales, 
            translations, 
            dL_dDensity,
            dL_dRotations,
            dL_dScales,
            dL_dTranslations
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