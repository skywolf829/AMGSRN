import torch
from AMG_Encoder import encode
import unittest
from pytorch_ops import encode_pytorch
from utils import get_random_rst, get_random_feature_grids, get_random_points, random_quaternion

class TestEncode(unittest.TestCase):

    def test_bottom_back_left_torch(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[-1., -1., -1.]], device='cuda', dtype=torch.float32)
        
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_torch, feature_grids[0:1,:,0,0,0], rtol=0.1, atol=1e-8)

    def test_bottom_back_left_cuda(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[-1., -1., -1.]], device='cuda', dtype=torch.float32)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, feature_grids[0:1,:,0,0,0], rtol=0.1, atol=1e-8)

    def test_top_front_right_torch(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        
        # Verify
        torch.testing.assert_close(out_feats_torch, feature_grids[0:1,:,-1,-1,-1], rtol=0.1, atol=1e-8)

    def test_top_front_right_cuda(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, feature_grids[0:1,:,-1,-1,-1], rtol=0.1, atol=1e-8)

    def test_top_front_left_torch(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[-1., 1., 1.]], device='cuda', dtype=torch.float32)
        
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        
        # Verify
        torch.testing.assert_close(out_feats_torch, feature_grids[0:1,:,-1,-1,0], rtol=0.1, atol=1e-8)

    def test_top_front_left_cuda(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[-1., 1., 1.]], device='cuda', dtype=torch.float32)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, feature_grids[0:1,:,-1,-1,0], rtol=0.1, atol=1e-8)

    def test_OOB_1(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[-1-(1/14), -1-(1/14), -1-(1/14)]], device='cuda', dtype=torch.float32)

        out_feats_pytorch = encode_pytorch(feature_grids, r, s, t, points)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_pytorch, rtol=0.1, atol=1e-8)
    
    def test_OOB_2(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = torch.tensor([[-1-(1/17), -1-(1/17), -1-(1/17)]], device='cuda', dtype=torch.float32)

        out_feats_pytorch = encode_pytorch(feature_grids, r, s, t, points)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_pytorch, rtol=0.1, atol=1e-8)

    def test_random_single_point_no_transform(self):
        
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(1)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=1e-8)
    
    def test_random_batch_points_no_transform(self):
        batch_size=2**20
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.01, atol=1e-6)

    def test_random_batch_points_translate(self):
        batch_size=2**20
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.rand([1, 3], device='cuda', dtype=torch.float32)*2-1
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-8)

    def test_random_batch_points_scale(self):
        batch_size=2**20
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.rand([1, 3], device='cuda', dtype=torch.float32) + 0.1
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-8)

    def test_random_batch_points_rotation(self):
        batch_size=2**20
        r = random_quaternion(1)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-8)
    
    def test_random_batch_points_rotation_scale(self):
        batch_size=2**20
        r = random_quaternion(1)
        s = torch.rand([1, 3], device='cuda', dtype=torch.float32) + 0.1
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-6)
    
    def test_random_batch_points_rotation_translation(self):
        batch_size=2**20
        r = random_quaternion(1)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.rand([1, 3], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-6)
    
    def test_random_batch_points_scale_translation(self):
        batch_size=2**20
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.rand([1, 3], device='cuda', dtype=torch.float32) + 0.1
        t = torch.rand([1, 3], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-6)

    def test_random_batch_points_rotation_scale_translation(self):
        batch_size=2**20
        r = random_quaternion(1)
        s = torch.rand([1, 3], device='cuda', dtype=torch.float32) + 0.1
        t = torch.rand([1, 3], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [32,32,32])
        points = get_random_points(batch_size)
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=batch_size*1e-6)

    def test_operation_single_feature(self):
        num_grids = 16
        num_points = 1
        r, s, t = get_random_rst(batch_size=num_grids)
        feature_grids = get_random_feature_grids(num_grids, 1, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=1e-5)

    def test_operation_single_feature_single_grid(self):
        num_grids = 1
        num_points = 1
        r, s, t = get_random_rst(batch_size=num_grids)
        feature_grids = get_random_feature_grids(num_grids, 1, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=1e-5)

    def test_operation(self):
        num_grids = 16
        num_points = 1
        r, s, t = get_random_rst(batch_size=num_grids)
        feature_grids = get_random_feature_grids(num_grids, 2, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        out_feats_cuda = encode(points, r, s, t, feature_grids)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=1e-5)

    def test_operation_batch(self):
        num_grids = 16
        num_points = 2**20
        r, s, t = get_random_rst(batch_size=num_grids)
        feature_grids = get_random_feature_grids(num_grids, 2, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)

        # Verify
        torch.testing.assert_close(out_feats_cuda, out_feats_torch, rtol=0.1, atol=num_points*1e-6)

    def test_gradient_single_point_no_transform_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [4,4,4])
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone().detach()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone().detach()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)
    
    def test_gradient_single_point_no_transform_2(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [4,4,4])
        points = torch.tensor([[-1., -1., -1.]], device='cuda', dtype=torch.float32)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone().detach()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone().detach()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)
 
    def test_gradient_single_point_no_transform_3(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [4,4,4])
        points = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone().detach()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone().detach()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)

    def test_gradient_single_point_no_transform_4(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 1, [2,2,2])
        points = get_random_points(1)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone().detach()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone().detach()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)
            
    def test_gradient_single_point_rotation(self):
        r = random_quaternion(1)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        feature_grids = get_random_feature_grids(1, 2, [4,4,4])
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone().detach()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone().detach()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)
        
    def test_gradient_single_grid_batch_no_transform(self):
        
        num_grids = 1
        num_points = 2**20
        r = torch.zeros([num_grids, 4], device='cuda', dtype=torch.float32)
        r[:,3] = 1.
        s = torch.ones([num_grids, 3], device='cuda', dtype=torch.float32)
        t = torch.zeros([num_grids, 3], device='cuda', dtype=torch.float32)

        feature_grids = get_random_feature_grids(num_grids, 2, [4,4,4])
        points = get_random_points(batch_size=num_points)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=num_points*1e-6)
    
    def test_gradient_multi_grid_batch_no_transform_1(self):
        
        num_grids = 16
        num_points = 2**20
        r = torch.zeros([num_grids, 4], device='cuda', dtype=torch.float32)
        r[:,3] = 1.
        s = torch.ones([num_grids, 3], device='cuda', dtype=torch.float32)
        t = torch.zeros([num_grids, 3], device='cuda', dtype=torch.float32)

        feature_grids = get_random_feature_grids(num_grids, 2, [4,4,4])
        points = get_random_points(batch_size=num_points)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=num_points*1e-6)
    
    def test_gradient_multi_grid_batch_no_transform_2(self):
        
        num_grids = 16
        num_points = 2**20
        r = torch.zeros([num_grids, 4], device='cuda', dtype=torch.float32)
        r[:,3] = 1.
        s = torch.ones([num_grids, 3], device='cuda', dtype=torch.float32)
        t = torch.zeros([num_grids, 3], device='cuda', dtype=torch.float32)

        feature_grids = get_random_feature_grids(num_grids, 4, [4,4,4])
        points = get_random_points(batch_size=num_points)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=num_points*1e-6)
        
    def test_gradient_batch_scale(self):
        num_grids = 16
        num_points = 2**20
        r = torch.zeros([num_grids, 4], device='cuda', dtype=torch.float32)
        r[:,3] = 1.
        s = torch.rand([num_grids, 3], device='cuda', dtype=torch.float32) + 0.1
        t = torch.zeros([num_grids, 3], device='cuda', dtype=torch.float32)

        feature_grids = get_random_feature_grids(num_grids, 2, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)
    
    def test_gradient_batch_rotation(self):
        num_grids = 16
        num_points = 2**20
        r = random_quaternion(num_grids)
        s = torch.ones([num_grids, 3], device='cuda', dtype=torch.float32)
        t = torch.zeros([num_grids, 3], device='cuda', dtype=torch.float32)

        feature_grids = get_random_feature_grids(num_grids, 2, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)
        
    def test_gradient_batch(self):
        num_grids = 16
        num_points = 2**20
        r, s, t = get_random_rst(batch_size=num_grids)
        feature_grids = get_random_feature_grids(num_grids, 2, [32,32,32])
        points = get_random_points(batch_size=num_points)
        
        # Torch pass
        out_feats_torch = encode_pytorch(feature_grids, r, s, t, points)
        l_torch = (out_feats_torch**2).sum()
        l_torch.backward()
        feat_grid_grads_torch = feature_grids.grad.clone()

        feature_grids.grad = None

        # CUDA pass
        out_feats_cuda = encode(points, r, s, t, feature_grids)
        l_cuda = (out_feats_cuda**2).sum()
        l_cuda.backward()
        feat_grid_grads_cuda = feature_grids.grad.clone()

        # Verify
        torch.testing.assert_close(feat_grid_grads_cuda, feat_grid_grads_torch, 
            rtol=0.1, atol=1e-5)

if __name__ == '__main__':
    unittest.main()