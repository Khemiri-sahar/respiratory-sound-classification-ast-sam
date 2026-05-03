import torch
import torch.optim as optim


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Save original weights
                self.state[p]["old_p"] = p.data.clone()
                
                if group["adaptive"]:
                    # Adaptive: weight by parameter magnitude
                    e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                else:
                    # Standard: unit-norm direction
                    e_w = p.grad * scale.to(p)
                
                # Perturb weights
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Restore original weights
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if group["adaptive"]:
                        grad_norm = (torch.abs(p) * p.grad).norm(p=2)
                    else:
                        grad_norm = p.grad.norm(p=2)
                    
                    norms.append(grad_norm.to(shared_device))
        
        # Return total norm
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        
        total_norm = torch.norm(torch.stack(norms), p=2)
        return total_norm
