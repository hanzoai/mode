# MoDE — Mixture of Diverse Experts

Route across expert modules **harvested frozen** from the largest open-weight models, with
complexity-aware hierarchical routing that adapts compute to task difficulty.

**Specified, not built.** No MoDE model exists. The zen SKUs serve today from upstream open
models; zen5-max is intended to be the first that runs MoDE. This repo publishes the
architecture and no numbers it has not measured.

## The line

| | Name | Where experts come from | Status |
|---|---|---|---|
| **v1** | Mixture of *Distilled* Experts | Upcycled from our own dense checkpoints ([Drop-Upcycling](https://arxiv.org/abs/2502.19261)) | Superseded |
| **v2** | Mixture of *Diverse* Experts | Harvested frozen from open models | Specified; never built |
| **v3** | Mixture of *Diverse* Experts | Harvested frozen; hypermodal; Rust-native | This repo |

v1 taught that expert **diversity must be engineered** — identical experts get identical
gradients and never differentiate. v2 took that to its conclusion: the strongest available
diversity isn't noise injected into copies of one model, it's experts from models independently
trained by different groups on different data. *Upcycling manufactures diversity; harvesting
finds it already made.*

v1 is recorded rather than quietly dropped. An architecture line that erases its own
supersessions cannot be audited, and v1's technique is not wrong — it is right wherever
diversity has *not* already been trained by someone else, which is simply not the case for open
frontier text models. The rule is symmetric:

> **Harvest where diversity already exists; upcycle where it does not.**

**MoDE ends at v3.** What follows is not a fourth MoDE but a different line: the diffusion work
([enso](https://github.com/zenlm/enso)) arrived independently at the same idea — route across
experts too diverse to have been trained together — and generalizes past what MoDE binds. That
architecture is **MUEN**, specified in its own repo ([muen](https://github.com/hanzoai/muen));
Map/Reduce/Route is what MoDE contributes to it.

Perception enters through the same seam: JEPA-family encoders
([V-JEPA 2](https://github.com/zenlm/vjepa2)) and the jin multimodal framework are `Route`
targets like any other expert. "Hypermodal" needs no new mechanism — **an encoder is just an
expert whose modality differs**.

## Design targets

Not measurements — nothing here has been built or run.

| | |
|---|---|
| Expert pool | ~3.1T params, 1000+ experts, 6 text families + vision/video/3D/audio |
| Trained surface | **~394M (0.013%)** — 207M estimator + 134M alignment + 52M router |
| Active per request | 0.8B (a greeting) → 100B+ (a proof) |
| Experts | **Frozen.** Never fine-tuned, never synced, pinned to a device |

**Diverse, not distilled.** Experts are taken as-is from six independently-trained
model families and never fine-tuned. Their diversity is the point — different groups,
different data, different attention mechanisms learn different representations, and a
router can exploit that.

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

| Class | Locality | Effect | |
|-------|----------|--------|---|
| `Map` | index-local, same device | **composes** — fuses into one kernel | implemented |
| `Reduce` | row-local, same device | fences **within** a device (contraction) | implemented |
| `Route` | expert-local, may cross | fences **across** devices (placement) | proposed here |

The fuser already partitions a trace at its fences. `Reduce` fences split a kernel;
`Route` fences split a *device*. Each Map region is then a fused kernel running entirely
on `π(e)`. The scheduler never learns about experts; the fuser never learns about
devices. Both only read the class — the same rule that makes fusion decidable makes
placement decidable.

Training uses the identical partition: experts are frozen, so no `Route`-fenced region
needs a backward pass. The backward graph is the forward graph with every `Route` region
cut out — which is the algebraic statement of "we train 0.013% of the model."

`Map` and `Reduce` exist today in `hanzo-kernel` (`fuse.rs`), whose fuser already cuts Map
regions at `Reduce` fences. `Route` is this paper's proposed extension; it is not implemented.

## Routing in this stack

Three routing decisions exist here, at three levels. MoDE adds no new router — it names the
level the others leave open.

| Level | Where it lives | Decides |
|---|---|---|
| request → model | `hanzo-router` (`Classifier`, `RoutePolicy`) | which *model* serves a request |
| request → tier | MoDE `ComplexityEstimator` | how much *compute* the request warrants |
| token → expert | `hanzo-kernel` `quant::moe_route` | which *experts* a token activates |

MoDE's tier is the axis `hanzo-router` already calls `Level` (`Fast`/`Balanced`/`Max`), at five
bands rather than three — chosen there by policy and SLO, predicted here from the prompt's first
64 tokens. `hanzo-router` already carries the seam for exactly that: `trait Classifier` is
heuristic today and documented so a learned model can drop in without touching the policy. The
estimator belongs in that seam. Per-tier expert selection lowers to `quant::moe_route`, the
fused softmax + top-k + renorm the kernel DSL already ships. One router per level, no third one.

## Paper

[`mode.tex`](mode.tex) — the architecture, the expert pool, routing, placement, and the
Map/Reduce/Route partition. Build with `pdflatex mode.tex`.

**There is no evaluation**, because there is nothing yet to evaluate. The premise the whole
architecture rests on — that a trained projection makes another family's frozen FFN useful —
is itself untested. It is the first thing to measure and the cheapest to falsify.

## Credit

Every expert is someone else's trained work, used under its own license. See
[`NOTICE`](NOTICE). Hanzo's contribution on top is identity training, agentic-data
fine-tuning, and abliteration — we claim none of the experts' capabilities.
