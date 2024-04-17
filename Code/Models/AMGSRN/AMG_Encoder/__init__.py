import torch
from . import _C

class CreateTransformationMatricesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotations, scales, translations):
        # Store for use in backward
        ctx.save_for_backward(rotations, scales, translations)

        # Assuming create_transformation_matrices is your compiled CUDA function for forward pass
        result = _C.createTransformationMatricesForward(rotations, scales, translations)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        rotations, scales, translations = ctx.saved_tensors

        # Assuming create_transformation_matrices_backward is your compiled CUDA function for backward pass
        grad_rotations, grad_scales, grad_translations = _C.createTransformationMatricesBackward(rotations, scales, translations, grad_output)

        return grad_rotations, grad_scales, grad_translations

class EncodeCoordinates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_coordinates, rotations, scales, translations, feature_grids, feature_vector_length):

        # Assuming create_transformation_matrices is your compiled CUDA function for forward pass
        feature_vectors = _C.encodeForward(query_coordinates, rotations, 
                                           scales, translations, feature_grids,
                                           feature_vector_length)

        # Store for use in backward
        ctx.save_for_backward(query_coordinates, rotations, 
                        scales, translations, feature_grids)

        return feature_vectors

    @staticmethod
    def backward(ctx, grad_output):
        query_coordinates, rotations, scales, \
            translations, feature_grids = ctx.saved_tensors

        # Assuming create_transformation_matrices_backward is your compiled CUDA function for backward pass
        grad_feature_grids = _C.encodeBackward(query_coordinates, 
            rotations, scales, translations, feature_grids, grad_output)
        return None, None, None, None, grad_feature_grids, None

class FeatureDensity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_coordinates, rotations, scales, translations):

        # Assuming create_transformation_matrices is your compiled CUDA function for forward pass
        density = _C.featureDensityForward(query_coordinates, rotations, 
                                           scales, translations)

        # Store for use in backward
        ctx.save_for_backward(query_coordinates, rotations, 
                        scales, translations)

        return density

    @staticmethod
    def backward(ctx, grad_output):
        query_coordinates, rotations, \
            scales, translations = ctx.saved_tensors
        
        # Assuming create_transformation_matrices_backward is your compiled CUDA function for backward pass
        dL_dRotations, dL_dScales, dL_dTranslations = _C.featureDensityBackward(query_coordinates, 
            rotations, scales, translations, grad_output)

        return None, dL_dRotations, dL_dScales, dL_dTranslations


def create_transformation_matrices(rotations, scales, translations) -> torch.Tensor:
    return CreateTransformationMatricesFunction.apply(rotations, scales, translations)

def encode(query_coordinates, rotations, scales, translations, feature_grids, feature_vector_length=None) -> torch.Tensor:
    if(feature_vector_length is None):
        feature_vector_length = feature_grids.shape[0]*feature_grids.shape[1]
    return EncodeCoordinates.apply(query_coordinates, 
        rotations, scales, translations, feature_grids, feature_vector_length)

def feature_density(query_coordinates, rotations, scales, translations) -> torch.Tensor:
    return FeatureDensity.apply(query_coordinates, 
        rotations, scales, translations)