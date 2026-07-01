# Triplet tool-recognition results — full head-to-head vs SurgeNetXL (2026-07-01)

## ★ HEADLINE ★

On the **full triplet task (tool + verb + target + IVT)**, at each encoder's
**best pooling**, our frozen V-JEPA 2.1 ViT-g e19 **beats** the supervised
SurgeNetXL baseline on **IVT, verb, target, and the per-task mean** — SurgeNetXL
wins only the isolated tool-presence cell.

This overturns the original comparison. Leonardo's published table (both encoders
at the harness-default **mean** pool) showed SurgeNetXL ahead on every metric,
including IVT 20.30 vs our 15.97. **The entire gap was a probe-pooling artifact.**
Switching both encoders to `topk_mean` pooling flips the result: our IVT nearly
**doubles** (15.97 → 29.10) while SurgeNetXL's rises only modestly (20.30 → 25.78).

Mechanism: V-JEPA (masked-prediction SSL) produces **spatially peaked** features —
a tool/action/anatomy signal lives in a few tokens for a few frames. Mean-pool
averages that over hundreds of mostly-background tokens and crushes it. `topk_mean`
(average of the top-k=8 tokens) reads it out cleanly. SurgeNetXL's CAFormer/DINO
features are already less mean-sensitive, so it gains far less from the switch.

## The complete 2×2 table (all cells computed in OUR harness)

Joint head (one shared token-pool feeds tool5 + verb6 + target12), 25 epochs,
frozen encoder, res 384. Macro mAP per task; **IVT** = macro-AP over the 278
supported triplets of the 5×6×12=360 space, scored as the product of the three
per-task sigmoid probabilities (independent-head assumption). Computed with
Leonardo's own `compute_map.py` — identical tooling to his published table.

| encoder / pooling      | tool  | verb  | target | **IVT** | mean(t,v,t) |
|------------------------|-------|-------|--------|---------|-------------|
| ours e19 — mean        | 69.09 | 63.75 | 41.38  | 15.97   | 58.07       |
| ours e19 — **topk_mean** | 86.62 | **76.29** | **51.82** | **29.10** | **71.58** |
| SurgeNetXL — mean      | 82.91 | 69.36 | 48.14  | 20.94   | 66.81       |
| SurgeNetXL — **topk_mean** | **88.65** | 73.07 | 49.74 | 25.78 | 70.49    |

### Head-to-head at matched best pooling (both `topk_mean`)

| metric        | ours  | SurgeNetXL | winner          |
|---------------|-------|------------|-----------------|
| **IVT**       | **29.10** | 25.78  | **ours +3.32**  |
| verb          | **76.29** | 73.07  | ours +3.22      |
| target        | **51.82** | 49.74  | ours +2.08      |
| mean(t,v,t)   | **71.58** | 70.49  | ours +1.09      |
| tool          | 86.62 | **88.65**  | SurgeNetXL +2.03|

## Method validation (why the topk_mean cells are trustworthy)

Both **mean-pool rows reproduce Leonardo's published table** within run variance,
using our harness end-to-end (render → 4-GPU mpiexec train → val-dump →
compute_map). Since the mean rows land on his numbers, the topk_mean rows sit on
the same trusted footing.

| row              | our recompute (tool/verb/target/IVT) | Leo published            |
|------------------|--------------------------------------|--------------------------|
| ours e19 mean    | 69.09 / 63.75 / 41.38 / 15.97        | 69.05 / 63.78 / 41.35 / 16.05 |
| SurgeNetXL mean  | 82.91 / 69.36 / 48.14 / 20.94        | 82.57 / 68.77 / 46.70 / 20.30 |

## Honest caveats

1. **The published (mean-pool) comparison genuinely favored SurgeNetXL** on every
   metric. The flip is entirely a pooling effect — not a new checkpoint, not more
   pretraining. State it as "matched-pooling changes the conclusion," not "we were
   always ahead."
2. **SurgeNetXL still wins the isolated tool cell** (88.65 vs 86.62). Joint sharing
   slightly *costs* our tool head (single-task topk_mean was 88.13 → joint 86.62)
   while it *helps* SurgeNetXL's (87.34 → 88.65) — SurgeNetXL handles the shared
   3-task head better on tool specifically. We more than recover it on verb/target.
3. **IVT is a product-of-independent-heads proxy**, not a jointly-modeled triplet
   classifier. It is the same proxy Leonardo's table uses, so the comparison is
   fair, but it is not the CholecT50-style disentangled IVT metric.
4. `target` class 9 has only **1 positive** in val (AP ≈ 0.5, near-random by
   construction) and drags the target macro for every row equally — cancels in
   comparisons.

## Pooling sweep provenance (single-task tool-only, frozen e19)

The joint result was preceded by a tool-only pooling sweep that first exposed the
artifact:

| pooling            | our e19 tool mAP | SurgeNetXL tool mAP |
|--------------------|------------------|---------------------|
| mean               | 71.14            | 82.09               |
| max                | 84.49            | —                   |
| **topk_mean (k=8)**| **88.13**        | 87.34               |
| attn_d2 (heavy)    | 40.3 (overfit)   | —                   |

`topk_mean` uniquely rescues **irrigator** (mean 69.8 / max 69.3 / topk_mean 88.7):
irrigator's signal is a small *region*, not a single peak token, so top-k averaging
captures it where single-max fails. Ordering mean < max < topk_mean confirms the
localized-signal mechanism.

## Recommendation

- **Adopt `token_pool: topk_mean, token_pool_topk: 8` as the standard triplet-probe
  head** for V-JEPA backbones. (Valid values: `max`, `mean`, `logsumexp`,
  `topk_mean`; `topk` alone is invalid → ValueError.)
- Always sweep pooling before concluding a frozen encoder is deficient on a
  localized fine-grained target. The whole "V-JEPA can't do tool identity" arc was
  a mean-pool default.

## Artifacts (Polaris)

- Harness: `/eagle/projects/ModCon/ngetty/triplet_probe/`
- Joint configs: `configs/joint_e19/`, `configs/joint_surgenetxl/`
- Dumps (per-task probs+labels npz): `runs/joint_e19/dump_{mean,topk_mean}`,
  `runs/joint_surgenetxl/dump_{mean,topk_mean}`
- mAP tool: `.../joint_triplet/joint_tool5verb6target12_surg_2_1_v1_phase2_main_e29/compute_map.py`
  (`--dump-dir <dir> --top-triplets 0`)
- Checkpoints: e19 at `/eagle/tpc/leonardo_borgioli/ngetty_ckpts/vitg384_cleandata/e19.pth.tar`

## Cross-reference: SAR action segmentation (same e19, different task)

SAR asformer probe is **plateaued** — e19 (75.56) is the peak; neither constant-LR
cresume (e24/29/34 = 75.5→74.4) nor 64f cooldown (e5/e10 = 73.8/74.5) beats it.
No pretraining lever improved SAR; the triplet win came purely from the pooling
readout, not more pretraining. Together: **e19 is the model to ship**; downstream
gains now come from probe/readout choices, not further CPT.
