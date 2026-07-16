# MoDE — Mixture of Diverse Experts

| | Name | Where experts come from | Status |
|---|---|---|---|
| **v1** | Mixture of *Distilled* Experts | Upcycled from our own dense checkpoints ([Drop-Upcycling](https://arxiv.org/abs/2502.19261)) | superseded |
| **v2** | Mixture of *Diverse* Experts | Harvested frozen from open models | [zen5](https://github.com/zenlm/zen5) |
| **v3** | Mixture of *Diverse* Experts | Harvested frozen; hypermodal; Rust-native | this repo |

v1 taught that expert **diversity must be engineered** — identical experts get identical
gradients and never differentiate. v2 took that to its conclusion: the strongest available
diversity isn't noise injected into copies of one model, it's experts from models independently
trained by different groups on different data. *Upcycling manufactures diversity; harvesting
finds it already made.*

The architecture behind [zen5](https://github.com/zenlm/zen5): route across expert
modules **harvested frozen** from the largest open-weight models, with
complexity-aware hierarchical routing that adapts compute to task difficulty.

**Diverse, not distilled.** Experts are taken as-is from six independently-trained
model families and never fine-tuned. Their diversity is the point — different groups,
different data, different attention mechanisms learn different representations, and a
router can exploit that.

| | |
|---|---|
| Expert pool | ~3.1T params, 1000+ experts, 6 text families + vision/video/3D/audio |
| **Trained** | **~394M (0.013%)** — 207M estimator + 134M alignment + 52M router |
| Active per request | 0.8B (a greeting) → 100B+ (a proof) |
| Experts | **Frozen.** Never fine-tuned, never synced, pinned to a device |

## The two ideas

**1. Harvest, don't train.** The expensive artifact — a trained expert — is already
published. What was missing is a way to use experts from *different families* together:
a learned alignment layer projects each source's hidden states into a shared space.

**2. Placement is a value.** Because harvested experts are frozen, a pinned expert is a
pure function that is permanently resident on one device — no gradient, no optimizer
state, no sync, no reload. Placement is a static map `π : expert → device`, so a router
that emits an expert id has already emitted an address. There is no runtime placement
policy to get wrong.

## Map, Reduce, Route

MoDE extends the [hanzo-kernel](https://github.com/hanzoai/ml) fusion algebra by exactly
one class, and the extension is forced rather than designed:

| Class | Locality | Effect |
|-------|----------|--------|
| `Map` | index-local, same device | **composes** — fuses into one kernel |
| `Reduce` | row-local, same device | fences **within** a device (contraction) |
| `Route` | expert-local, may cross | fences **across** devices (placement) |

The fuser already partitions a trace at its fences. `Reduce` fences split a kernel;
`Route` fences split a *device*. Each Map region is then a fused kernel running entirely
on `π(e)`. The scheduler never learns about experts; the fuser never learns about
devices. Both only read the class — the same rule that makes fusion decidable makes
placement decidable.

Training uses the identical partition: experts are frozen, so no `Route`-fenced region
needs a backward pass. The backward graph is the forward graph with every `Route` region
cut out — which is the algebraic statement of "we train 0.013% of the model."

## Paper

[`mode.tex`](mode.tex) — the architecture, the expert pool, routing, placement, and the
Map/Reduce/Route partition. Evaluation is in progress and reported when the numbers are
measured; this paper publishes none it has not.

## Credit

Every expert is someone else's trained work, used under its own license. See
[`NOTICE`](NOTICE). Hanzo's contribution on top is identity training, agentic-data
fine-tuning, and abliteration — we claim none of the experts' capabilities.
