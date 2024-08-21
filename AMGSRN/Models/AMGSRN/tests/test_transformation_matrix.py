import torch
from AMG_Encoder import create_transformation_matrices
import unittest
from pytorch_ops import composite_transformation
from utils import get_random_rst

class TestCompositeTransform(unittest.TestCase):
    def test_operation(self):
        r, s, t = get_random_rst()

        m_torch = composite_transformation(r, s, t)
        m_cuda = create_transformation_matrices(r, s, t)

        # Verify
        torch.testing.assert_close(m_torch, m_cuda, rtol=1e-4, atol=1e-6)

    def test_operation_batch(self):
        
        batch_size = 2**20
        
        r, s, t = get_random_rst(batch_size=batch_size)

        m_torch = composite_transformation(r, s, t)
        m_cuda = create_transformation_matrices(r, s, t)

        # Verify
        torch.testing.assert_close(m_torch, m_cuda, rtol=1e-4, atol=1e-6)

    def test_gradient(self):
        r, s, t = get_random_rst()

        # Torch pass
        m_torch = composite_transformation(r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        # Reset grads
        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = create_transformation_matrices(r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify gradients
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, 
                                    rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, 
                                    rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, 
                                    rtol=1e-4, atol=1e-6)
    
    def test_gradient_batch(self):
        batch_size = 2**20

        r, s, t = get_random_rst(batch_size=batch_size)

        # Torch pass
        m_torch = composite_transformation(r, s, t)
        l_torch = (m_torch**2).sum()
        l_torch.backward()
        grad_r_torch = r.grad.clone().detach()
        grad_s_torch = s.grad.clone().detach()
        grad_t_torch = t.grad.clone().detach()

        # Reset grads
        r.grad = None
        s.grad = None
        t.grad = None

        m_cuda = create_transformation_matrices(r, s, t)
        l_cuda = (m_cuda**2).sum()
        l_cuda.backward()
        grad_r_cuda = r.grad.clone().detach()
        grad_s_cuda = s.grad.clone().detach()
        grad_t_cuda = t.grad.clone().detach()

        # Verify gradients
        torch.testing.assert_close(grad_r_cuda, grad_r_torch, 
                                    rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_s_cuda, grad_s_torch, 
                                    rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(grad_t_cuda, grad_t_torch, 
                                    rtol=1e-4, atol=1e-6)

if __name__ == '__main__':
    unittest.main()