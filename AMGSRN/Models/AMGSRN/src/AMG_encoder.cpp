#include "AMG_encoder.h"

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
    auto float_opts = rotations.options().dtype(torch::kFloat32);
    torch::Tensor out = torch::empty({num_grids, 4, 4}, float_opts);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_grids + threadsPerBlock - 1) / threadsPerBlock;
    launch_create_transformation_matrices(
        num_grids, 
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(), 
        translations.contiguous().data_ptr<float>(), 
        out.contiguous().data_ptr<float>()
    );

    return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_transformation_matrices_backward(
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& dL_dMatrix)
{
    const int num_grids = (int)rotations.size(0);
    auto float_opts = rotations.options().dtype(torch::kFloat32);

    torch::Tensor dRotations = torch::empty({num_grids, 4}, float_opts);
    torch::Tensor dScales = torch::empty({num_grids, 3}, float_opts);
    torch::Tensor dTranslations = torch::empty({num_grids, 3}, float_opts);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_grids + threadsPerBlock - 1) / threadsPerBlock;
    launch_create_transformation_matrices_backward(
        num_grids,
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(), 
        translations.contiguous().data_ptr<float>(), 
        dRotations.contiguous().data_ptr<float>(),
        dScales.contiguous().data_ptr<float>(), 
        dTranslations.contiguous().data_ptr<float>(), 
        dL_dMatrix.contiguous().data_ptr<float>()
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
    // Get size information from the tensors
    const auto num_points = query_points.size(0);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto float_opts = query_points.options().dtype(torch::kFloat32);

    // Create temporary array for rotation matrices and output tensor
    torch::Tensor out_features = torch::empty({num_points, num_grids*features_per_grid}, float_opts);

    launch_encode_forward(
        num_points,
        num_grids,
        features_per_grid,
        D, H, W, 
        query_points.transpose(0, 1).contiguous().data_ptr<float>(), 
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(), 
        translations.contiguous().data_ptr<float>(), 
        feature_grids.contiguous().data_ptr<float>(), 
        out_features.contiguous().data_ptr<float>()
    );

    return out_features;
}

// Not working atm
torch::Tensor encode_forward_v2(
    const torch::Tensor& query_points,
    const torch::Tensor& rotations, 
    const torch::Tensor& scales, 
    const torch::Tensor& translations,
    const torch::Tensor& feature_grids)
{
    // Get size information from the tensors
    const auto num_points = query_points.size(0);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto float_opts = query_points.options().dtype(torch::kFloat32);

    // Create output tensor
    torch::Tensor out_features = torch::empty({num_points, num_grids*features_per_grid}, float_opts);

    // Convert quaternions to rotation matrices
    torch::Tensor rotation_matrices = quaternion_to_rotation_matrix(rotations);

    // Iterate over each grid
    for (int i = 0; i < num_grids; i++) {
        // Transform query points to local space
        torch::Tensor local_points = torch::matmul(rotation_matrices[i], query_points.t()).t();
        //local_points = local_points * scales[i].unsqueeze(0);
        //local_points = local_points + translations[i].unsqueeze(0);

        // Rescale points from [-1, 1] to [0, D-1], [0, H-1], [0, W-1]
        local_points = (local_points + 1) * torch::tensor({D-1, H-1, W-1}, float_opts) / 2;

        // Perform trilinear interpolation
        torch::Tensor interpolated = torch::nn::functional::grid_sample(
            feature_grids[i].unsqueeze(0),  // Add batch dimension
            local_points.unsqueeze(0).unsqueeze(0).unsqueeze(0),  // Add extra dimensions for grid_sample
            torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(true)
        );

        // Store the result in the output tensor
        out_features.narrow(1, i*features_per_grid, features_per_grid) = interpolated.squeeze().transpose(0,1);
    }

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
    // Get size information from the tensors
    const auto num_points = query_points.size(0);
    const auto num_grids = feature_grids.size(0);
    const auto features_per_grid = feature_grids.size(1);
    const auto D = feature_grids.size(2);
    const auto H = feature_grids.size(3);
    const auto W = feature_grids.size(4);    

    auto float_opts = query_points.options().dtype(torch::kFloat32);

    // Create temporary array for rotation matrices and output tensor
    torch::Tensor dL_dFeatureGrids = torch::zeros({num_grids, features_per_grid, D, H, W}, float_opts);

    launch_encode_backward(
        num_points,
        num_grids,
        features_per_grid,
        D, H, W, 
        query_points.contiguous().data_ptr<float>(), 
        rotations.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(), 
        translations.contiguous().data_ptr<float>(), 
        feature_grids.contiguous().data_ptr<float>(), 
        dL_dFeatureVectors.contiguous().data_ptr<float>(),         
        dL_dFeatureGrids.contiguous().data_ptr<float>()
    );

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
    auto float_opts = query_points.options().dtype(torch::kFloat32);
    torch::Tensor density = torch::empty({num_points, 1}, float_opts);

    launch_density_forward(
        num_points,
        num_grids,
        query_points.contiguous().data_ptr<float>(), 
        rotations.contiguous().data_ptr<float>(), 
        scales.contiguous().data_ptr<float>(), 
        translations.contiguous().data_ptr<float>(), 
        density.contiguous().data_ptr<float>());

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
    auto float_opts = rotations.options().dtype(torch::kFloat32);

    torch::Tensor dL_dRotations = torch::empty({num_grids, 4}, float_opts);
    torch::Tensor dL_dScales = torch::empty({num_grids, 3}, float_opts);
    torch::Tensor dL_dTranslations = torch::zeros({num_grids, 3}, float_opts);
    
    launch_density_backward(
        num_points,
        num_grids,
        query_points.contiguous().data_ptr<float>(), 
        rotations.contiguous().data_ptr<float>(), 
        scales.contiguous().data_ptr<float>(), 
        translations.contiguous().data_ptr<float>(), 
        dL_dDensity.contiguous().data_ptr<float>(),
        dL_dRotations.contiguous().data_ptr<float>(),
        dL_dScales.contiguous().data_ptr<float>(),
        dL_dTranslations.contiguous().data_ptr<float>()
    );


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