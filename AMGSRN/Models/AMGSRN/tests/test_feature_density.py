import torch
from AMG_Encoder import feature_density
import unittest
from pytorch_ops import feature_density_pytorch
from utils import get_random_rst, get_random_points, random_quaternion

class TestFeatureDensity(unittest.TestCase):
    def test_single_point_in_bounds_no_transforms_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_no_transforms_2(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[-1., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)
    
    def test_single_point_in_bounds_no_transforms_3(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)
    
    def test_single_point_OOB_no_transforms_4(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[2., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_scale_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[2., 2., 2.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_scale_2(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_scale_3(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[0.1, 0.1, 0.1]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_scale_4(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[100., 100., 100.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_scale_5(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[123., 100., 750.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)
    
    def test_single_point_in_bounds_translate_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)
    
    def test_single_point_in_bounds_translate_2(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_translate_3(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0.0, 0.0, 0.0]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[-0.5, -0.5, -0.5]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_in_bounds_rotate(self):
        r = random_quaternion(1)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32)
        t = torch.tensor([[0.0, 0.0, 0.0]], device='cuda', dtype=torch.float32)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-8)

    def test_single_point_random_transform(self):
        
        r, s, t = get_random_rst()
        points = get_random_points(1)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-6)
    
    def test_batch_random_transform(self):
        
        batch_size = 2**20
        r, s, t = get_random_rst(1)
        points = get_random_points(batch_size)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-6)

    def test_single_point_random_transform_batch_multigrid_1(self):
        
        batch_size = 1
        num_grids = 2

        r, s, t = get_random_rst(num_grids)
        points = get_random_points(batch_size)

        m_torch1 = feature_density_pytorch(points, r[0:1], s[0:1], t[0:1])
        m_cuda1 = feature_density(points, r[0:1], s[0:1], t[0:1])

        m_torch2 = feature_density_pytorch(points, r[1:2], s[1:2], t[1:2])
        m_cuda2 = feature_density(points, r[1:2], s[1:2], t[1:2])

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        print(m_torch1)
        print(m_cuda1)
        print(m_torch2)
        print(m_cuda2)
        print(m_torch)
        print(m_cuda)
        torch.testing.assert_close(m_cuda1, m_torch1, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(m_cuda2, m_torch2, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-6)

    def test_single_point_random_transform_batch_multigrid_2(self):
        
        batch_size = 1
        num_grids = 16

        r, s, t = get_random_rst(num_grids)
        points = get_random_points(batch_size)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-6)

    def test_batch_random_transform_batch_multigrid(self):
        
        batch_size = 2**20
        num_grids = 16

        r, s, t = get_random_rst(num_grids)
        points = get_random_points(batch_size)

        m_torch = feature_density_pytorch(points, r, s, t)
        m_cuda = feature_density(points, r, s, t)

        # Verify
        torch.testing.assert_close(m_cuda, m_torch, rtol=1e-4, atol=1e-6)

    def test_backward_single_point_in_bounds_no_transforms_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)
    
    def test_backward_single_point_in_bounds_no_transforms_2(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[-0.0001, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_in_bounds_no_transforms_3(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[1.5, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_in_bounds_no_transforms_4(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.rand([1, 3], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_in_bounds_no_transforms_5(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[-1., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_in_bounds_no_transforms_6(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.rand([1,3], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_translate_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[1., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[0.0, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_translate_2(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[-1., 0., 0.]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[0.0, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_translate_3(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[1.0, -0.4, 0.3]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[0.0, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_translate_4(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.rand([1,3], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[0.0, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_single_point_translate_5(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.rand([1,3], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.rand([1,3], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_batch_translate(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[1., 1., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.rand([1,3], device='cuda', dtype=torch.float32, requires_grad=True)
        points = get_random_points(2**20)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-2, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-2, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-2, atol=1e-8)

    def test_backward_single_point_scale_1(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.tensor([[0.5, 0.5, 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0,0,0]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = torch.tensor([[0.0, 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_batch_scale(self):
        r = torch.tensor([[0., 0., 0., 1.]], device='cuda', dtype=torch.float32, requires_grad=True)
        s = torch.rand([1,3], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0,0,0]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = get_random_points(2**20)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_batch_rotation(self):
        r = random_quaternion(1)
        s = torch.tensor([[1,1,1]], device='cuda', dtype=torch.float32, requires_grad=True)
        t = torch.tensor([[0,0,0]], device='cuda', dtype=torch.float32, requires_grad=True)
        points = get_random_points(2**20)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-2, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-2, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-2, atol=1e-8)

    def test_backward_single_point_multigrid(self):
        num_grids = 16
        r,s,t = get_random_rst(num_grids)
        points = torch.tensor([[0., 0., 0.]], device='cuda', dtype=torch.float32)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-8)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-8)

    def test_backward_batch_multigrid_1(self):
        num_grids = 16
        r,s,t = get_random_rst(num_grids)
        points = get_random_points(2**20)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-6)

    def test_backward_batch_multigrid_2(self):
        num_grids = 32
        r,s,t = get_random_rst(num_grids)
        points = get_random_points(2**20)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda)
        print(grad_r_torch)
        print(grad_s_cuda)
        print(grad_s_torch)
        print(grad_t_cuda)
        print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-4, atol=1e-6)

    def test_backward_batch_multigrid_3(self):
        num_grids = 64
        r,s,t = get_random_rst(num_grids)
        points = get_random_points(2**20)

        m_torch = feature_density_pytorch(points, r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = feature_density(points, r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify
        print(grad_r_cuda[38,2])
        print(grad_r_torch[38,2])
        #print(grad_s_cuda)
        #print(grad_s_torch)
        #print(grad_t_cuda)
        #print(grad_t_torch)
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, rtol=1e-3, atol=1e-6)

if __name__ == '__main__':
    unittest.main()