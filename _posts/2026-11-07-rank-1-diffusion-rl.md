---
layout: post
title: "One Direction Per Prompt: The Rank-1 Problem in Forward-Process Diffusion RL"
date: 2026-11-07
permalink: /blog/rank-1-diffusion-rl/
categories: posts
tags: diffusion flow-matching reinforcement-learning research
description: "Pooling rewards into one update direction per prompt makes diffusion RL fast. When a prompt has several valid renderings, it also makes the update unable to keep them all."
giscus_comments: true
---

Recent RL methods for flow-matching image models gain speed by pooling all the rewards of a prompt into a single update direction. This post argues that this pooling has a structural cost: when a prompt can be drawn in more than one valid way, a single direction cannot improve all of them, so reward can only go up by collapsing onto one mode or by leaving the data distribution. I state the result, sketch why it holds, and outline the experiments and estimators I plan to test.

## Background

**Flow matching is regression.** Rectified flow and flow matching [1, 2] interpolate between data $$x_0$$ and Gaussian noise $$\varepsilon$$,

$$x_t = (1-t)\,x_0 + t\,\varepsilon,$$

and train a velocity field $$v_\theta(x_t, t)$$ to regress $$u = \varepsilon - x_0$$. The least-squares optimum is a conditional mean, $$v(x,t) = \mathbb{E}[u \mid x_t = x]$$, and integrating this ODE from noise to data reproduces the data distribution. Stable Diffusion 3 [3] and FLUX are trained this way.

**KL-regularized RL is a tilt.** Maximizing $$\mathbb{E}[R] - \tfrac{1}{\alpha}\mathrm{KL}(p \,\|\, p_0)$$ has a closed-form optimum,

$$p^*(x_0) \propto p_0(x_0)\, e^{\alpha R(x_0)},$$

an exponential tilt of the base model [4, 5]. The target depends only on the distribution of clean samples, not on the denoising path. Its velocity field is the Doob transform of the base velocity [5, 6]:

$$v^* - v_0 = -\frac{t}{1-t}\,\nabla_x \log h_t, \qquad h_t(x) = \mathbb{E}\big[e^{\alpha R} \mid x_t = x\big].$$

To first order in $$\alpha$$ this is

$$v^* - v_0 \approx -\frac{\alpha}{t}\,\mathrm{Cov}\big(R,\ x_0 \mid x_t\big).$$

The direction depends on the current state $$x_t$$. A sample that is heading toward a good image on the left gets pushed toward that image; a sample heading right gets pushed right.

**Two families of diffusion RL.**

- **Reverse-process (policy-gradient) methods** such as DDPO [7] and Flow-GRPO [8] turn the ODE sampler into an SDE so each denoising step has a likelihood, then apply PPO-style ratio clipping. Stepwise-Flow-GRPO [9] and AC-Flow [10] add per-step credit assignment.
- **Forward-process (regression) methods** skip likelihoods. They sample $$x_0$$ from the current model, re-noise it to $$x_t$$, and regress the velocity toward a reward-weighted target. DiffusionNFT [11], AWM [12], AdvantageFlow [13] and RAM [6] belong here. DiffusionART [14] and DiffusionRVGE [15] are the newest members; they replace per-sample targets with pooled or regressed directions to cut variance and cost.

## Where prior work stands

Each line of work fixes one of variance, bias or cost by giving up another.

| Line of work | Methods | Limitation |
| --- | --- | --- |
| Reverse-process policy gradient | DDPO [7], Flow-GRPO [8] | Needs an SDE sampler to define likelihoods; slow (RAM reaches Flow-GRPO's peak with up to 50× fewer steps [6]); one advantage is shared across all steps |
| Per-step credit, reverse process | Stepwise-Flow-GRPO [9], AC-Flow [10] | Extra reward calls per step or a learned critic; still tied to SDE likelihoods |
| Per-sample forward regression | DiffusionNFT [11], AWM [12], AdvantageFlow [13], RAM [6] | Correct in expectation but high-variance, since every sample carries fresh re-noising noise; group-mean baselines are biased when the sampler differs from the model (e.g. under CFG) |
| Pooled forward updates | DiffusionART [14] | One direction per prompt for all states: low variance and fast, but rank-1 |
| Regressed value gradients | DiffusionRVGE [15] | Ridge regression at endpoints over ODE completions; no analysis of bias or mode coverage |
| Diversity fixes | DiverseGRPO [16], GARDO [17] | Diversity bonuses or uncertainty penalties at the reward level; collapse is treated as a reward problem, not a property of the update |

None of these identifies the rank of the per-prompt update as a cause of collapse.

## The rank-1 problem

DiffusionART [14] samples $$K$$ images for a prompt $$c$$ and computes one pooled direction,

$$G(c) = \frac{\mathrm{Cov}(R,\ x_0)}{\mathrm{RMS}\big(\mathrm{Cov}(R,\ x_0)\big)},$$

then adds the same $$G(c)$$ to the cached velocity at every state of that prompt. Pooling averages away sample noise, which is why ART reports 10–12× fewer GPU-hours than DiffusionNFT. But the update no longer depends on $$x_t$$: each round, each prompt gets one direction. Stack the per-state corrections of a prompt into a matrix and it is $$\mathbf{1}\, G(c)^\top$$, which has rank 1.

Compare this with the correct target above, which points in different directions depending on where a sample is heading.

**An example.** The prompt is "a cat", and the base model draws the cat on the left half the time and on the right half the time. The reward likes both. High-reward samples on the left and on the right sit on opposite sides of the sample mean, so their contributions to $$\mathrm{Cov}(R, x_0)$$ largely cancel. What survives is a small push toward whichever side happened to have more good samples in this batch, and that same push is applied to every image, including the ones heading to the other side.

**Proposition (rank-1 limit).** Let the base distribution be point-symmetric with two mirrored good modes, and let $$m_0, m_4$$ be the probability mass the model puts on each. For any correction $$G$$ that does not depend on the state,

$$\min(m_0, m_4) \le \min(m_0, m_4)\big|_{\text{base}}.$$

*Sketch.* Generation has two phases. At high noise, the trajectory commits to a mode: noise space is partitioned into basins, one per mode. At low noise, it refines within that mode. A constant velocity correction acts on the first phase as a shift of the starting noise, and on the second phase as a common translation of all endpoints. Let $$f(a)$$ be the probability that Gaussian noise shifted by $$a$$ lands in the left basin. By symmetry the right basin gets $$f(-a)$$. The map $$a \mapsto f(a)$$ is a Gaussian convolved with a convex indicator, so it is log-concave (Prékopa), and therefore

$$\min\big(f(a), f(-a)\big) \le \sqrt{f(a)\,f(-a)} \le f(0).$$

Shifting the noise can grow one mode only by shrinking the other. The translation of endpoints, meanwhile, can only move mass away from the data support.

**Corollary.** A pooled update can raise reward only by collapsing onto one mode or by pushing samples off the data distribution. The exact tilt, by contrast, multiplies both good modes by the same factor and keeps their ratio fixed.

The proposition needs symmetry. When good modes are close to orthogonal, one direction can raise both. Which case real image layouts fall into in latent space is the empirical question below.

## A related bias

The same lens explains a bias in per-sample methods. With sampler distribution $$q$$, sampler velocity $$m = \mathbb{E}_q[u \mid x_t]$$, model velocity $$v_k$$ and baseline $$b$$:

$$\mathbb{E}_q\big[(w - b)(u - v_k) \mid x_t\big] = h\,(v^*_q - m) + (h - b)(m - v_k).$$

The first term is the tilt we want. The second appears whenever the sampler differs from the model (CFG, distilled samplers, discretization) and the baseline is the group mean rather than the conditional value $$h = \mathbb{E}[w \mid x_t]$$. It is largest at low noise, where $$h$$ is far from the group mean. This is why AWM skips low timesteps, and why RAM reports its own estimator as biased. DiffusionART's "consistency force" is the $$b = \tfrac12$$ case of this term.

## What a fix looks like

The goal is to estimate the per-state direction $$\mathrm{Cov}(R, x_0 \mid x_t)/t$$ while keeping most of pooling's variance reduction. Weighting samples by latent distance does not scale: in a latent space with tens of thousands of dimensions the weights become one-hot. Three estimators that do scale:

1. **Per-sample re-noising with a conditional baseline.** AWM/RAM-style weighted flow matching, with the group mean replaced by a Tweedie baseline $$b = R(\hat x_0(x_t))$$, $$\hat x_0 = x_t - t\,v(x_t)$$. Its target equals the per-state direction in expectation, and the baseline removes the bias term above. It is cheap to train on only two timesteps: on SD3.5-M with AWM and PickScore, training each rollout on the 2 timesteps with the largest predicted signal matches the official 6 timesteps at 0.63× wall-clock.
2. **Two-timestep branching.** At those same two states, branch $$B = 4$$ SDE continuations to the end. The branches are samples from $$p(x_0 \mid x_t)$$, so their reward–image covariance is the per-state direction directly. No temperature, works with binary rewards, at the cost of extra rollouts.
3. **Clustered pooling.** Cluster the $$K$$ samples of a prompt on DINOv2 features [19], keep one pooled direction per cluster, and assign each state to a cluster through its $$\hat x_0$$. One clustering step on top of ART; rank $$k$$ instead of rank 1.

## Experiments

All on SD3.5-Medium (LoRA, 512 px), matching the ART, AWM and RAM setups.

- **Does rank-1 hurt coverage?** On GenEval [18] prompts with several valid layouts (two objects, spatial relations), train DiffusionART and estimator 1, sample 64 images per prompt each epoch, and track cluster entropy on DINOv2 features and an off-manifold score (base-model reconstruction error after re-noising). Prediction: ART's entropy falls from mid-training while its off-manifold score rises; estimator 1 keeps entropy above 70% of the base model at similar success.
- **Estimator comparison.** Estimators 1–3 against ART at fixed GPU-hours. Prediction: branching gives the best coverage with reward within 0.005 of ART.
- **CFG bias.** CFG ∈ {1, 2, 4.5} × baseline {group mean, Tweedie} × {all t, t ≥ 0.36}. Prediction: divergence only for group mean × all t × CFG > 1.
- **Breadth.** Repeat on FLUX.1-dev, with OCR as a single-answer control where rank-1 should be harmless.

The coverage and off-manifold metrics are calibrated first on the base model and on best-of-N sampling, which approximates the exact tilt and gives the reference line for "no collapse".

## References

1. Liu, Gong, Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
2. Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
3. Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
4. Uehara et al. *Understanding Reinforcement Learning-Based Fine-Tuning of Diffusion Models: A Tutorial and Review.* 2024. [arXiv:2407.13734](https://arxiv.org/abs/2407.13734)
5. Domingo-Enrich, Drozdzal, Karrer, Chen. *Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control.* ICLR 2025. [arXiv:2409.08861](https://arxiv.org/abs/2409.08861)
6. Bergmeister, Jegelka, Nüsken, Domingo-Enrich, Pidstrigach. *Reinforce Adjoint Matching.* 2026. [arXiv:2605.10759](https://arxiv.org/abs/2605.10759)
7. Black, Janner, Du, Kostrikov, Levine. *Training Diffusion Models with Reinforcement Learning.* ICLR 2024. [arXiv:2305.13301](https://arxiv.org/abs/2305.13301)
8. Liu et al. *Flow-GRPO: Training Flow Matching Models via Online RL.* 2025. [arXiv:2505.05470](https://arxiv.org/abs/2505.05470)
9. *Stepwise Credit Assignment for GRPO on Flow-Matching Models.* CVPR 2026. [arXiv:2603.28718](https://arxiv.org/abs/2603.28718)
10. Fan et al. *Fine-tuning Flow Matching Generative Models with Intermediate Feedback.* 2025. [arXiv:2510.18072](https://arxiv.org/abs/2510.18072)
11. Zheng et al. *DiffusionNFT: Online Diffusion Reinforcement with Forward Process.* ICLR 2026. [arXiv:2509.16117](https://arxiv.org/abs/2509.16117)
12. *Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models.* ICML 2026. [arXiv:2509.25050](https://arxiv.org/abs/2509.25050)
13. Kveton et al. *AdvantageFlow.* 2026. [arXiv:2605.26013](https://arxiv.org/abs/2605.26013)
14. Wan et al. *DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models.* 2026. [arXiv:2610.12572](https://arxiv.org/abs/2610.12572)
15. *DiffusionRVGE: Reinforcement Learning via Regression-Based Value Gradient Estimation.* 2026. [arXiv:2610.22973](https://arxiv.org/abs/2610.22973)
16. Liu et al. *DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO.* 2025. [arXiv:2512.21514](https://arxiv.org/abs/2512.21514)
17. *GARDO: Reinforcing Diffusion Models without Reward Hacking.* 2025. [arXiv:2512.24138](https://arxiv.org/abs/2512.24138)
18. Ghosh, Hajishirzi, Schmidt. *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.* NeurIPS 2023. [arXiv:2310.11513](https://arxiv.org/abs/2310.11513)
19. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
