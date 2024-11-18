import torch
import numpy as np

def diffusion_loss(model, x_0, t, config):
    x_0 = x_0.to(config.model.device)
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.diffusion_steps, dtype=np.float64)
    betas_tensor = torch.tensor(betas, dtype=torch.float32, device=config.model.device)
    noise = torch.randn_like(x_0, device=x_0.device)
    alpha_t = (1 - betas_tensor).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    noisy_input = alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * noise
    model_output = model(noisy_input, t.float())
    loss = (noise - model_output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    return loss

def sample(y0, x, seq, model, config, w):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(t.long(),config)
            at_next = compute_alpha(next_t.long(),config)
            xt = xs[-1].to(x.device)
            et = model(xt, t)
            if w==0:
                et_hat = et
            else:
                guidance = condition_score(model, xt, et, y0, t, config)
                et_hat = et + (1 - at).sqrt() * w * guidance
            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
            xs.append(xt_next.to('cpu'))
    return xs

def sample_mask(y0, mask, seq, model, config, w): 
    test_steps = torch.Tensor([config.model.test_steps]).type(torch.int64).to(config.model.device)
    at = compute_alpha(test_steps.long(),config)
    z = torch.randn_like(y0)
    x = at.sqrt() * y0 + (1- at).sqrt() * z.to(config.model.device)
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        for index, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(t.long(),config)
            at_next = compute_alpha(next_t.long(),config)
            xt = xs[-1].to(x.device)
            xt_noise = at.sqrt() * y0 + (1- at).sqrt() * z.to(config.model.device)
            xt = torch.where(mask==1, xt, xt_noise)

            et = model(xt, t)
            if w==0:
                et_hat = et
            else:
                guidance = condition_score(model, xt, et, y0, t, config)
                et_hat = et + (1 - at).sqrt() * w * guidance

            x0_t = (xt - et_hat * (1 - at).sqrt()) / at.sqrt()
            c1 = (
                ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et_hat
            xs.append(xt_next.to('cpu'))
    return xs

def condition_score(model, xt, et, x_guidance, t, config):
    with torch.enable_grad():
        x_in = xt.detach().requires_grad_(True)
        et = model(x_in, t)
        at = compute_alpha(t.long(),config)
        x_0_hat = (x_in - et * (1 - at).sqrt()) / at.sqrt()
        difference_x = x_0_hat-x_guidance
        norm_x = torch.linalg.norm(difference_x)
        test_grad = torch.autograd.grad(outputs=norm_x, inputs=x_in)[0]
    return test_grad

def compute_alpha(t, config):
    betas = np.linspace(config.model.beta_start, config.model.beta_end, config.model.diffusion_steps, dtype=np.float64)
    betas = torch.tensor(betas).type(torch.float)
    beta = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    beta = beta.to(config.model.device)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a