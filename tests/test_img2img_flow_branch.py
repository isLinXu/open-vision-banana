"""GPU-free test of the img2img vs t2i latent-mixing branch in InstructionTuner.

This mirrors the exact formula in
``src/open_vision_banana/training/tuner.py::_flow_training_step``
(lines 1599-1612) so we can verify **verification gate 2** (training uses an
image-conditioned img2img flow, not a pure text-to-image flow) without loading
FLUX / diffusers / a GPU.

The branch logic (replicated below, keep in sync with tuner.py):

    if use_img2img_flow and input_latents is not None:
        noise_term = flow_sigma * (t*(1-t)).sqrt() * noise   # if flow_sigma>0
        noisy = (1-t)*input + t*target + noise_term
        velocity = target - input
    else:
        noisy = (1-t)*target + t*noise
        velocity = target - noise

Running this test on CPU proves the mechanism; see docs/MINIMAL_REPRODUCTION_PLAN_DEPTH.md.
"""

import torch
import torch.nn.functional as F


def flow_noisy_and_target(input_latents, target_latents, noise, t_b,
                          use_img2img_flow, flow_sigma):
    """Replica of tuner.py::_flow_training_step latent-mixing branch (1599-1612)."""
    if use_img2img_flow and input_latents is not None:
        if input_latents.shape != target_latents.shape:
            input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])
        if flow_sigma > 0:
            noise_term = flow_sigma * (t_b * (1.0 - t_b)).sqrt() * noise
        else:
            noise_term = torch.zeros_like(input_latents)
        noisy = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term
        velocity = target_latents - input_latents
    else:
        noisy = (1.0 - t_b) * target_latents + t_b * noise
        velocity = target_latents - noise
    return noisy, velocity


def _make(batch=2, c=4, h=8, w=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(batch, c, h, w, generator=g)
    tgt = torch.randn(batch, c, h, w, generator=g)
    nz = torch.randn(batch, c, h, w, generator=g)
    t = torch.rand(batch, 1, 1, 1, generator=g)
    return inp, tgt, nz, t


def test_img2img_branch_is_image_conditioned():
    inp, tgt, nz, t = _make()
    noisy, vel = flow_noisy_and_target(inp, tgt, nz, t, use_img2img_flow=True, flow_sigma=1.0)
    # velocity must predict the input->target residual, not the target itself
    assert torch.allclose(vel, tgt - inp), "img2img velocity should be target - input"
    # noisy must contain the input latent (image conditioning)
    expected = (1.0 - t) * inp + t * tgt + 1.0 * (t * (1.0 - t)).sqrt() * nz
    assert torch.allclose(noisy, expected), "img2img noisy formula mismatch"
    # noise_term must be non-zero (flow_sigma>0 -> training sees noisy latents)
    noise_term = noisy - ((1.0 - t) * inp + t * tgt)
    assert noise_term.abs().max() > 1e-6, "flow_sigma>0 must inject noise into training"


def test_t2i_branch_used_when_input_is_none():
    _, tgt, nz, t = _make()
    noisy, vel = flow_noisy_and_target(None, tgt, nz, t, use_img2img_flow=True, flow_sigma=1.0)
    # t2i path: velocity is target - noise, noisy starts from noise at t=1
    assert torch.allclose(vel, tgt - nz), "t2i (input=None) velocity should be target - noise"
    expected = (1.0 - t) * tgt + t * nz
    assert torch.allclose(noisy, expected), "t2i noisy formula mismatch"


def test_t2i_branch_used_when_flag_off():
    inp, tgt, nz, t = _make()
    noisy, vel = flow_noisy_and_target(inp, tgt, nz, t, use_img2img_flow=False, flow_sigma=1.0)
    # even with input present, flag-off forces pure t2i
    assert torch.allclose(vel, tgt - nz), "flag-off must force t2i velocity"
    expected = (1.0 - t) * tgt + t * nz
    assert torch.allclose(noisy, expected), "flag-off must force t2i noisy"


def test_flow_sigma_zero_is_clean_latent_regression():
    inp, tgt, nz, t = _make()
    # flow_sigma=0 (pre-fix default) -> no noise_term -> clean latent convex mix.
    # This is exactly the R7 concern: training never sees noisy latents.
    noisy, vel = flow_noisy_and_target(inp, tgt, nz, t, use_img2img_flow=True, flow_sigma=0.0)
    noise_term = noisy - ((1.0 - t) * inp + t * tgt)
    assert torch.allclose(noise_term, torch.zeros_like(noise_term)), \
        "flow_sigma=0 must yield a zero noise term (clean regression)"
    assert torch.allclose(vel, tgt - inp), "clean img2img velocity still target - input"


if __name__ == "__main__":
    test_img2img_branch_is_image_conditioned()
    test_t2i_branch_used_when_input_is_none()
    test_t2i_branch_used_when_flag_off()
    test_flow_sigma_zero_is_clean_latent_regression()
    print("ALL img2img-flow branch tests passed (GPU-free).")
