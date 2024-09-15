import torch
from . import _C
from torch.amp import custom_fwd, custom_bwd

class CreateTransformationMatricesFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, rotations, scales, translations):
        # Store for use in backward
        ctx.save_for_backward(rotations, scales, translations)

        # Assuming create_transformation_matrices is your compiled CUDA function for forward pass
        result = _C.createTransformationMatricesForward(rotations, scales, translations)

        return result

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        rotations, scales, translations = ctx.saved_tensors

        # Assuming create_transformation_matrices_backward is your compiled CUDA function for backward pass
        grad_rotations, grad_scales, grad_translations = _C.createTransformationMatricesBackward(rotations, scales, translations, grad_output)

        return grad_rotations, grad_scales, grad_translations

class EncodeCoordinates(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda", cast_inputs=torch.float16)
    def forward(ctx, query_coordinates, rotations, scales, translations, feature_grids):

        # Assuming create_transformation_matrices is your compiled CUDA function for forward pass
        feature_vectors = _C.encodeForward(query_coordinates, rotations, 
                                           scales, translations, feature_grids)

        # Store for use in backward
        ctx.save_for_backward(query_coordinates, rotations, 
                        scales, translations, feature_grids)

        return feature_vectors

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        query_coordinates, rotations, scales, \
            translations, feature_grids = ctx.saved_tensors

        # Assuming create_transformation_matrices_backward is your compiled CUDA function for backward pass
        grad_feature_grids = _C.encodeBackward(query_coordinates, 
            rotations, scales, translations, feature_grids, grad_output)
        return None, None, None, None, grad_feature_grids

class FeatureDensity(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda") # This one doesn't benefit from float16, actually hurts performance
    def forward(ctx, query_coordinates, rotations, scales, translations):

        # Assuming create_transformation_matrices is your compiled CUDA function for forward pass
        density = _C.featureDensityForward(query_coordinates, rotations, 
                                           scales, translations)

        # Store for use in backward
        ctx.save_for_backward(query_coordinates, rotations, 
                        scales, translations)

        return density

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        query_coordinates, rotations, \
            scales, translations = ctx.saved_tensors
        
        # Assuming create_transformation_matrices_backward is your compiled CUDA function for backward pass
        dL_dRotations, dL_dScales, dL_dTranslations = _C.featureDensityBackward(query_coordinates, 
            rotations, scales, translations, grad_output)

        return None, dL_dRotations, dL_dScales, dL_dTranslations


def create_transformation_matrices(rotations, scales, translations) -> torch.Tensor:
    return CreateTransformationMatricesFunction.apply(rotations, scales, translations)

def encode(query_coordinates, rotations, scales, translations, feature_grids) -> torch.Tensor:
    return EncodeCoordinates.apply(query_coordinates, 
        rotations, scales, translations, feature_grids)

def feature_density(query_coordinates, rotations, scales, translations) -> torch.Tensor:
    return FeatureDensity.apply(query_coordinates, 
        rotations, scales, translations)