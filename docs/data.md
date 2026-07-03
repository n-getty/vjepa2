# Data catalog

Datasets we store on `flare` and use for V-JEPA 2 surgical continued-pretraining (CPT)
and downstream probing. This is a fork of `facebookresearch/vjepa2` doing surgical-domain
CPT of ViT-g, evaluated on surgical benchmarks.

- **Pretraining corpus** lives at `/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/`
  as WebDataset tar shards, mixed by per-source sqrt-size temperature sampling.
- **Probe/eval data** lives under `/flare/ModCon/ngetty/surg_2_1_v*/` (cached features + CSV
  manifests) and, for the CholecT50 triplet probe, on Polaris (`/eagle/...`).

Paths use the `/flare/...` mount (equivalent to `/lus/flare/projects/...`).

---

## 1. Pretraining corpus (WebDataset)

Root: `/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/`

All sources are resharded `.tar` WebDataset. The active ViT-g 384 mix
(`configs/vitg16_surg_vid_webdataset_single4/vitg384_cleandata.yaml` for pretrain,
`vitg384_cooldown_64f.yaml` for cooldown) uses **15 sources, all weight 1.0**, with
`sampling_temperature: 0.5` (sqrt-size) and `min_clip_std: 1.0` (drops black/frozen clips).
The bottom 5 rows were added this session and their configs point at them; ⏳ = segment+reshard
job still in flight (confirm a `metadata.json` exists in each dir before launching a run).

| Source key | On disk | Shards | Modality / notes | In active ViT-g mix |
|---|---|---|---|---|
| `kinetics400` | 15G | 500 | General action video (non-surgical anchor). ~59% of clip count. | ✅ |
| `surgvu24_clean` | 320G | 2000 | SurgVU-24 / SurgToolLoc robotic. `_clean` = black-clip filtered (~19% of raw was byte-identical pure black). | ✅ |
| `surgtoolloc2022` | 203G | 1500 | SurgToolLoc-2022 robotic. | ✅ |
| `sitl_2026` | 179G | 512 | Leo's newer SITL re-segmentation, 4,144 clips @ 60s/30fps/1080p (distinct `v2_` keys). Owner-only perms. | ✅ |
| `grasp` | 133G | 124 | GraSP robotic prostatectomy, 1,988 clips @ 60s. Video-native segmentation; modality-matched to our evals. | ✅ |
| `sitl` | 90G | 600 | Original SITL segmentation (kept alongside `sitl_2026` per decision). | ✅ |
| `cholec80` | 70G | 182 | Cholec80 laparoscopic cholecystectomy 25fps, 2,916 clips. Domain-broadening (laparoscopic). | ✅ |
| `surgenet_robotic_clean` | 23G | 500 | SurgeNet robotic subset, minus 40 eval-leaked source videos (318 clips dropped) — eval-scrub. | ✅ |
| `lemon` | ~900G | — | LEMON / Surg-3M: 4,194 YouTube surgical videos → 53,650 clips, 35 procedures (2,527 lap + 1,667 robotic). pHash-deduped vs eval + surgenet_robotic. Owner-only. | ✅ |
| `small_surg` | (symlinks) | — | Symlink bundle of the 6 tiny sets below (926 clips total). Bundled so temperature sampling sizes them by combined count instead of oversampling each individually. | ✅ |
| `heichole` | ⏳ | ⏳ | HeiChole: 24 HD lap-chole full procedures, 3 centers (~22h → 60s clips). White-censored out-of-body spans dropped by `min_clip_std`. | ✅ (⏳ segmenting) |
| `multibypass140` | ⏳ | ⏳ | MultiBypass140: 140 lap gastric-bypass procedures, Bern+Strasbourg (2 centers) → 60s clips. | ✅ (⏳ segmenting) |
| `gynsurg` | ⏳ | ⏳ | GynSurg action segments, gyn laparoscopy (**new sub-domain**), 1080p/30fps pre-cut clips (≥4s filter). Shares Vienna pool w/ `lapgyn6_events`. | ✅ (⏳ segmenting) |
| `lapgyn6_events` | ⏳ | ⏳ | LapGyn6-Events segments, gyn laparoscopy, pre-cut event clips (≥4s; segments variant is 64f-capable). Shares Vienna pool w/ `gynsurg`, no dedup (different segment types). | ✅ (⏳ segmenting) |
| `surgenet_lap` | done | 365 | SurgeNet **laparoscopic** YouTube set (4fps), 5,843 clips: raw procedure dirs re-segmented to 60s + `clips_1min` subset. Eval-gated at segment time (crop-aug ref, 0 leaks); 96.9% distinct from `lemon`. Owner-only. The first lap-SurgeNet in the corpus. | ✅ (resharded) |

### `small_surg` bundle members

Bundled because equal per-source seats oversampled the tiniest set (`endovis15`, 8 clips)
~82×/epoch → memorization. Bundling caps worst-case per-clip oversample at ~8×, drops no data.
Older configs (e.g. `vitg384_cooldown_32f.yaml`, ViT-L configs) still list these individually.

| Source key | On disk | Shards | Notes |
|---|---|---|---|
| `jigsaw` | 1.5G | 32 | JIGSAWS. |
| `surgvisdom` | 1.6G | 32 | SurgVisDom. |
| `crcd` | 950M | 32 | Colorectal (CRCD). |
| `miccai_2017` | 255M | 4 | MICCAI 2017 instrument. |
| `miccai_endoseg` | 177M | 3 | MICCAI EndoSeg. |
| `endovis15` | 20M | 2 | EndoVis-15 (8 clips — the oversampling driver). |

### Staged / not-in-active-mix

On disk but not referenced by the active ViT-g configs:

| Dir | On disk | Status |
|---|---|---|
| `surgvu24` | 325G | Pre-filter SurgVU-24 (superseded by `surgvu24_clean`). |
| `grasp_staging`, `grasp_clean` | — | Intermediate GraSP segmentation stages. |
| `cholec80_staging`, `cholec80_clean` | — | Intermediate Cholec80 segmentation stages. |
| `*_staging` (heichole/multibypass140/gynsurg/lapgyn6_events) | — | Per-source seg output; resharded into the final dirs above. |
| `incoming_robotic/{cholecseg8k}` | 2.9G | CholecSeg8k — **kept, not packed** (Cholec80-frame redundancy; masks-only signal). Future segmentation-probe set. |

### IMAGE sets — held for a targeted cooldown (not in the current video mix)

Downloaded + packer-ready but **deliberately left out of the pretrain/cooldown configs for now**;
intended for a future *targeted* image-branch cooldown to address spatial-task underperformance
(see §4 and the draft `vitg384_cooldown_64f_imgbranch.yaml`). Packed as `<name>_img/` via
`scripts/pack_images_pbs.sh`:

| Source | On disk (raw) | Content |
|---|---|---|
| `hyperkvasir` | 3.9G | 10.7K labeled GI-endoscopy images, 23 classes. |
| `dsad` | 20.9G | Dresden anatomy: 13.2K images (masks filtered out at pack time). |
| `esad` | 13G | ESAD robotic prostatectomy frames (YOLO labels skipped). |
| `psi_ava` | 12.1G | PSI-AVA robotic prostatectomy keyframes (DETR features skipped). |

### Loading params (active ViT-g 384)

- `dataset_type: WebDataset`, `batch_size: 2` per rank (pretrain) / `1` (cooldown 64f), `crop_size: 384`, `patch_size: 16`, `tubelet_size: 2`
- `fps: 4`, `dataset_fpcs: 16` (pretrain) / `64` (cooldown) — cooldown lengthens the temporal window, fps stays 4.
- `sampling_temperature: 0.5`, `min_clip_std: 1.0`
- **15 sources** as of this session (was 10; +heichole, +multibypass140, +gynsurg, +lapgyn6_events, +surgenet_lap).

---

## 2. Downstream probe / eval datasets

Frozen-backbone probes. The two live surgical benchmarks are **SAR-RARP50** (action
segmentation, on Aurora) and the **CholecT50 triplet** (tool/verb/target, on Polaris).

### SAR-RARP50 — action segmentation (ASFormer)

Temporal action segmentation, 8 surgical action classes, ASFormer head on frozen encoder
features. Robotic (RARP = robot-assisted radical prostatectomy) — modality-matched to the corpus.

- **Source clips:** `/flare/ModCon/ngetty/surg_vid/sarrarp50/globus/sarrarp50_action_clips_3x16f_4fps/` (3×16-frame clips @ 4fps, MP4)
- **CSV manifests:** `/flare/ModCon/ngetty/surg_2_1_v1_probes_aurora/csv/sarrarp50_actions_4fps_asformer_ctx3_seq_{train,val,test}.csv`
- **Cached features (frozen encoder, by resolution/checkpoint):**
  - `/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache_384/<variant>/{train,val}` — 384px, current
  - `.../full_cache_256/`, `.../full_cache/` — 256px variants
  - `.../fs10_cache/<variant>/{train,val}` — 10% stratified few-shot subset for fast CPT iteration
- **Configs:** `configs/heads/sarrarp50/`
  - `full_cached_384/` — current 384px probes; `cd64f_e{5,10,15,20}_{probe,export}.yaml` = cooldown-64f checkpoint sweep
  - `fs10_cached/` — few-shot 10% (v1/v2/v3/metaraw checkpoint variants)
  - `full_cached_256/`, `full_cached/`, `full_aug_384/`, `aurora_v1lambda/`, `v2_probe/`, `full_xcheck/`
- **Probe spec:** `num_classes: 8`, `frames_per_clip: 16`, `num_segments: 3` (ctx3), `sequence_labels: true`, weighted CE. ASFormer: 10 layers, 8 heads, 8 tokens/clip, 24 temporal tokens.
- **Two-phase per checkpoint:** `*_export.yaml` (`export_cache: true`) precomputes frozen features once; `*_probe.yaml` trains the ASFormer head off the cache (~2 min/epoch full, cached).

### CholecT50 triplet — tool / verb / target (Polaris)

Joint multi-label recognition: Tool (5) + Verb (6) + Target (12); IVT = independent product
of the three. Laparoscopic. Runs on **Polaris**, not Aurora.

- **Harness:** `/eagle/projects/ModCon/ngetty/triplet_probe/` (Polaris)
- **Configs:** `configs/joint_e19/` (our ViT-g e19), `configs/joint_surgenetxl/` (SurgeNetXL baseline)
- **Dumps:** `runs/joint_*/dump_{mean,topk_mean}` (per-task npz probs)
- **Metric:** per-task macro mAP + IVT product. Best pooling `topk_mean` (topk=8).

### Generic video-classification templates (not used here)

`configs/eval/`, `configs/eval_2_1/` carry upstream K400/SSv2/Diving48/Jester/EK100/COIN/IN1K
templates with `/your_data_path/` stubs. Placeholders only — not wired to surgical data in this fork.

---

## 3. Candidate VIDEO datasets — access & value (researched 2026-07-02)

Access/gating for the temporal-video candidates we don't already have, from primary sources
(project pages, GitHub, arXiv, Synapse/figshare/OpenAlex APIs). Ranked by value for **our**
robotic+laparoscopic corpus. "New temporal video" = what's actually usable after excluding
frames-only content and overlap with what we already hold.

**Access legend:** OPEN = direct download · REG-EULA = free account + terms (Synapse etc.) ·
FORM = email/Google-form approval · FRAMES-ONLY = no temporal video released · DEAD = host gone.

### Tier 1 — pursue (on-domain or large, temporal video, obtainable)

| Dataset | Modality | Access | New temporal video | Host | Notes |
|---|---|---|---|---|---|
| **MultiBypass140** | Laparoscopic (gastric bypass) | **OPEN** (`wget` zip) | 140 videos, 2 centers | CAMMA S3 (`s3.unistra.fr`) | ✅ **downloaded** (365 GB) → segmenting `multibypass140/` + **in configs**. CC-BY-NC-SA. `github.com/CAMMA-public/MultiBypass140` |
| **UCL Rectal Cancer** | Laparoscopic (TME) | **OPEN** (CC-BY) | 75 MP4s, ~380h, 1080p/25fps | UCL RDR / figshare | **~765 GB** — HEAD one file before staging. figshare API gives direct URLs. DOI 10.5522/04/24769530 |
| **HeiChole** | Lap cholecystectomy | **TEAM-JOIN** (Synapse) | 24 HD full videos (`Full/HD/`), ~22h, 3 centers | Synapse `syn18824884` | ✅ **downloaded** (108GB) → segmenting into `heichole/` + **in configs**. Access = join Team 3390210 (instant); Download-scoped PAT. HD only (`Full/` SD + `Skill/` are dupes/subclips). |
| **AutoLaparo** | Laparoscopic (hysterectomy) | **FORM** (Google) | 21 videos, ~23h, 1080p/25fps | emailed link | On-domain, small. CC-BY-NC-SA. `autolaparo.github.io` |
| **EndoMapper** | GI endoscopy | **REG-EULA** (Synapse) | ~96 procedures, >24h continuous | Synapse `syn26707219` | Off-domain (flexible scope) but true complete-procedure video. Best of the GI group. |

### Tier 2 — conditional / off-domain broadening

| Dataset | Modality | Access | New temporal video | Host | Notes |
|---|---|---|---|---|---|
| **SurgBench** | Mixed surgical | **OPEN** (Apache-2.0) | ~220GB clips (mostly eval subset) | HF `JianhuiWei/SurgBench_NIPS25` | **Aggregates 15 datasets we may already hold** (Cholec80, CholecT45, AutoLaparo, JIGSAWS, Kvasir…) — high overlap; dedup hard. AVOS/SimSurgSkill excluded from release. |
| **M2CAI16-workflow** | Lap cholecystectomy | **FORM** (CAMMA) | 41 videos, 25fps | CAMMA | **Same Strasbourg pool as Cholec80** → pHash-dedup before ingest (reuse LEMON gate). |
| **CholecTrack20** | Lap cholecystectomy | **FORM** + Synapse | **only ~8** full videos (test set); train/val are 1fps frames | Synapse `syn53182642` | Low yield for temporal. CC-BY-NC-SA. |
| **Cataract-101** | Ophthalmic | **OPEN** (zip) | 101 videos, ~8.3GB | ITEC AAU FTP | Off-domain; lowest-friction acquisition of all. CC-BY-NC. |
| **Cataract-1K** | Ophthalmic | **REG-EULA** (Synapse) | 1000 videos, ~90GB | Synapse `syn53404507` | Off-domain. HF mirror is frames-only — use Synapse for video. |
| **LDPolypVideo** | Colonoscopy | **OPEN** (Baidu/GDrive) | 263 videos (~901k frames) | Baidu + Google Drive | Off-domain; confirm video-vs-frames container on download. License unstated. |
| **EgoSurgery-Phase** | Open surgery (egocentric) | **FORM** (Google) | ~20 videos (video-vs-frames unconfirmed) | emailed link | Unusual egocentric view; niche. CC-BY-NC-SA. |
| **Kvasir-Capsule** | Capsule endoscopy | **OPEN** (research) | ~91GB video ZIPs (confirm continuous-vs-clip) | Simula / OSF | Most off-domain (swallowed capsule). |

### Tier 3 — do NOT ingest (frames-only, dead, or gated with no confirmed video)

| Dataset | Why excluded |
|---|---|
| **CholecT45** | FRAMES-ONLY (1fps PNG) **and** a subset of Cholec80 (already held as 25fps video) — zero temporal gain. |
| **Galar** | FRAMES-ONLY (~580GB of frames); source video not released. Capsule/off-domain. |
| **ATLAS Dione** | FRAMES-ONLY (JPEG) **and** host dead (`cubs.buffalo.edu` returns empty). da Vinci skill-task clips only. |
| **AVOS** | FORM-gated, YouTube-sourced (link-rot, no redistribution). Open-surgery, off-domain. Video not directly obtainable. |
| **SurgPub-Video** | FORM-gated, no host disclosed, release status ambiguous ("will be released"). Publisher-copyright risk. |
| **GLENDA** | FRAMES-ONLY (~373 annotated + 13.4K unannotated gyn-laparoscopy frames, ITEC AAU, OPEN/CC-BY-NC). No temporal video. A SurgeNet source. |
| **LapGyn4** | FRAMES-ONLY (~59.6K gyn-laparoscopy images, ~10.6GB, ITEC AAU, OPEN/CC-BY-NC). No video; Instrument-Count subset overlaps Cholec80. A SurgeNet source. |

### ITEC AAU gyn-laparoscopy VIDEO sets (checked 2026-07-02)

Same lab as GLENDA/LapGyn4/Cataract-101. All OPEN direct download, CC-BY-NC(-SA), gynecologic
laparoscopy (new sub-domain for us — no gyn in corpus yet). **All draw from the same ~600-video
Medical University of Vienna pool** → pHash-dedup across them (and vs eval sets) before mixing,
same as the LEMON gate. **Critical clip-length constraint:** V-JEPA needs ≥4s for a 16f@4fps
pretrain window and ≥16s for a 64f@4fps cooldown window. Most ITEC "action"/"recognition" clips
are pre-cut to 2–3s — fine for 16f, **too short for 64f cooldown**. Verified zip sizes via HEAD.

| Dataset | Content | Clip length | Usable for | Download | Size |
|---|---|---|---|---|---|
| **GynSurg** (action-segments) | 1080p/30fps video, 152 source vids → segments | segments (longer) | 16f ✅, 64f ⚠️ verify | ✅ **downloaded** (44.2GB) → segmenting `gynsurg/` + **in configs** | 47.5 GB |
| **GynSurg** (3sec) | 1080p/30fps, 3s clips | 3s | 16f only (12f@4fps) | `GynSurg_Action_3sec.zip` | 26.1 GB |
| **GynSurg** (raw LHE 75 vids) | full HD procedures | full | 16f ✅ + 64f ✅ | **FORM-gated** (sign `LapGynLHE...UsageAgreementForm.pdf`) | — |
| **LapGyn6-Events** (segments) | video, up to >1 min clips | 1s–>1min | 16f ✅ + **64f ✅** | ✅ **downloaded** (53.9GB) → segmenting `lapgyn6_events/` + **in configs** | 57.9 GB |
| **LapGyn6-Events** (recognition) | video, 2–3s clips | 2–3s | 16f only | `Event_Recognition_LapGyn_dataset.zip` | 40.6 GB |
| **LapGyn6-Actions** | video clips (`.rar`, needs `unrar`) | likely 2–3s | 16f only (verify) | `.../LapGyn6-Actions/Dataset.rar` | 4.33 GB |
| **SurgicalActions160** | 160 mp4, 427×240/25fps | 2–5s (avg 4.8s) | 16f (91% of clips); **tiny ~13min total** | `.../SurgicalActions160/downloads/SurgicalActions160.zip` | 44.8 MB |

**Best picks:** GynSurg (best quality, native 1080p/30fps — matches our ViT-g@384 pivot; take the
**segments** zip + pursue the form-gated 75 raw LHE videos for 64f) and **LapGyn6-Events segments**
(the only pre-cut ITEC set with clips long enough for 64f cooldown). SurgicalActions160 is too small
to matter and risks the tiny-set oversampling trap. Skip the segmentation-frame subsets (frames-only).

**Other datasets on the ITEC server** (`ftp.itec.aau.at/datasets/`): `ENID` (endometriosis, likely
frames), `Smoke_cholec80` (Cholec80-derived smoke annotations, we have Cholec80), and `ovid/` — a
folder of ophthalmic (cataract) video sets: `cat-101` (=Cataract-101), `cat-21` (Cataract-21),
`LensID`, `lens-dislocation`, etc. (off-domain but video).

### Recommendation

1. **MultiBypass140** first — open, direct, on-domain laparoscopic, no gating. Fastest win.
2. **HeiChole** + **AutoLaparo** — pursue the Synapse registration / Google form in parallel; both add on-domain laparoscopic multi-center diversity for little data volume.
3. **UCL Rectal Cancer** if you want bulk laparoscopic video and can absorb ~765 GB.
4. Treat the GI/cataract sets (EndoMapper, Cataract-*, LDPolypVideo, Kvasir-Capsule) as optional *off-domain broadening* only — and remember `lemon_staging` (922 GB, already downloaded) is the higher-leverage unclaimed source before scraping any of these.
5. Before ingesting **M2CAI16** or **SurgBench**, run the pHash dedup gate against Cholec80 / existing corpus — both heavily overlap what we hold.
6. **Gyn-laparoscopy** (currently absent from our corpus): **GynSurg segments** + **LapGyn6-Events segments** add a new sub-domain in temporal video at high resolution — pHash-dedup the shared Vienna pool first.

---

## 4. IMAGE / frame datasets for the spatial branch (researched 2026-07-02)

**Motivation:** our model underperforms on *spatial* / per-frame tasks (tool & anatomy recognition,
phase-from-frame, the CholecT50 triplet). The `vjepa_2_1` image branch (§5) can mix single-frame
data with video, so surgical image sets are a direct lever on this weakness. Below is what the
surgical-FM field actually pretrains on, filtered to what's new and high-value for us.

### Anchor: the SurgeNetXL source table (Jaspers et al., arXiv 2501.09436)

SurgeNetXL = **4,711,024 frames, 14 sources, "over 23 procedures"** — the most-cited surgical-FM
aggregation. Note **every source is video-*derived* frames**, so much of it overlaps our video
territory. Full table (frames = sampled count; ✅/⚠️ = our holdings):

| Source | Procedure / modality | #Frames | Public | For us |
|---|---|---|---|---|
| Cholec80 | Lap cholecystectomy | 179,164 | Yes | ✅ have (video) |
| HeiChole | Lap cholecystectomy | 53,427 | Yes (Synapse) | ⏳ downloading (video) |
| hSDB-Chole | Lap cholecystectomy | 18,064 | Yes | Cholec-like |
| RAMIE-UMCU | RA esophagectomy | 377,287 | **No (private)** | — |
| ESAD | RA prostatectomy | 47,282 | Yes | new (robotic) |
| PSI-AVA | RA prostatectomy | 73,618 | Yes | new (robotic) |
| RARP-AvL | RA prostatectomy | 261,516 | **No (private)** | — |
| DSAD (Dresden) | RA rectal resection | 14,623 | Yes | **new, anatomy-labeled** |
| GLENDA | Gyn laparoscopy | 25,682 | Yes | frames (ITEC) |
| LapGyn4 | Gyn laparoscopy | 59,616 | Yes | frames (ITEC) |
| MultiBypass140 | Lap gastric bypass | 749,419 | Yes | ⏳ downloading (video) |
| hSDB-Gastric | RA gastrectomy | 35,576 | Yes | new (procedure) |
| SurgToolLoc2022 | 11 RA porcine procedures | 741,516 | Yes | ✅ have (SurgVU/toolloc) |
| YouTube (SurgeNet) | **23 procedures** | 2,074,234 | Yes (HF `TimJaspersTue/SurgeNetYoutube`) | broad, but frames-of-video |

Composites: SurgeNetPublic = 1,997,987 (public, excl. YouTube); SurgeNetXL = 4.71M.

### Shortlist to download for spatial lift (ranked) — with acquisition status

Prioritizing **multi-procedure diversity**, **dense per-frame labels** (the actual spatial signal),
and **FM-validated + openly downloadable**. Status as of 2026-07-02; landing in `data/incoming_robotic/`.

| Rank | Dataset | Why | Access | Status |
|---|---|---|---|---|
| 1 | **HyperKvasir** | 10.6K labeled images, **23 GI finding classes**, native stills — biggest open per-frame-label diversity win; new modality | OPEN | ⏳ **downloading** via HF mirror `sahilur/hyper-kvasir-labeled-images` (simula.no host **unreachable from Aurora**, even via proxy; 99K *unlabeled* subset stuck behind simula — labeled 23-class part is the high-value one) |
| 2 | **DSAD (Dresden)** | 13.2K images with **organ/anatomy segmentation** — most on-target for our *anatomy* gap; new procedure (rectal). CC-BY (only commercial-OK one). Note: real count 13,195 not 14,623 | OPEN (figshare `21702600`) | ✅ **already staged** (`dsad/DSAD.zip`, 20.9GB, Jul-1) |
| — | **CholecSeg8k** | ⚠️ **NOT for pretraining** — images are Cholec80 frames (17 clips / 8,080 imgs, we already hold all Cholec80 as video) → zero new pixels for SSL. Its 13-class dense masks are the only new signal, which the image branch never consumes. **Kept on disk as a future segmentation-PROBE set only** | OPEN | ⏳ downloading HF `minwoosun/CholecSeg8k` (2.9GB), kept but **excluded from image pack** |
| 4 | **ESAD + PSI-AVA** | RA prostatectomy frames — modality-matched to our robotic benchmarks; new procedure | Drive (gdown) | ESAD ✅ **already staged** (`esad/`, 13GB, Jul-1); PSI-AVA ⏳ (bare-IP host refused → gdown Drive fallback) |
| 5 | **hSDB-Chole + hSDB-Gastric** | gastrectomy adds a procedure; chole reinforces | OPEN | hSDB-Gastric ✅ **already staged** (`hsdb_gastric/`, 9.6GB, Jul-1) |
| 6 | **CaDIS** | cataract, 4,670 images, **36 seg classes** (dense instruments+anatomy) | REG (grand-challenge / CATARACTS) | not started (gated) |
| — | Endoscapes | (already staged `endoscapes/`, 5.9GB, Jul-1) — CVS/anatomy laparoscopic | OPEN | ✅ staged |
| — | SurgeNet YouTube frames | 23-procedure breadth, but frames-of-video (overlaps our video pipeline) | OPEN (HF) | not started |

**Note:** `data/incoming_robotic/` already held several of these from a Jul-1 session (dsad, esad,
endoscapes, hsdb_gastric) — check there before re-downloading anything. gdown's CLI entrypoint is
broken under `module load frameworks`; use `python3 -m gdown` instead.

**Skip (already held / private):** SurgToolLoc2022, GLENDA, LapGyn4 (have); RAMIE-UMCU, RARP-AvL
(private, unobtainable).

**Targeting the specific weakness:** anatomy → **DSAD**, **CaDIS**; tool presence/localization →
**CholecSeg8k**, **CaDIS**; multi-procedure phase-from-frame → **MultiBypass140** (already coming
as video), **SurgeNet YouTube**; broadest cheap native-image win → **HyperKvasir**. Note the
recurring **CC-BY-NC(-SA)** license across SurgeNet-adjacent releases (research-only).

*(Other FM works checked: LVM-Med is radiology (CT/MRI/X-ray) — off-domain, dropped. SurgVLP/
PeskaVLP/HecVL pretrain on YouTube surgical-lecture video+ASR, not a downloadable image set.)*

---

## 5. Other potential datasets (candidate corpus expansion)

Surgical/endoscopic datasets under consideration for ingest. Rows marked ✅ **have** are
already in our corpus (section 1). "Videos"/"Images/Frames" as reported by each source; verify
before ingest.

**Video vs. frames — both are usable, but not equally.** The `vjepa_2_1` trainer we run has a
first-class **image branch** (`img_data` + `img_mask` in `app/vjepa_2_1/train.py`): images load
as single-frame clips (`dataset_fpcs: [1]`, `tubelet_size: 1`) with a spatial-only mask, and
`rank_ratio` splits the distributed world between an image sub-batch and a video sub-batch (each
with its own loss lambda). Meta's own 2.1 recipe uses this — `configs/train_2_1/vitG16/*` mix
ImageNet-1K at `rank_ratio: 0.5`. **Our surgical configs omit `img_data` by choice**, so frames-only
sets are *addable via config*, not blocked by code. Caveat: V-JEPA's learning signal is dominated by
*temporal* masking, so frames only exercise the spatial half — prefer temporal **video** for corpus
growth, and treat frames as a targeted lever (e.g. the per-frame CholecT50 triplet probe) rather than
bulk pretraining fuel. The `frames-only` tags below flag that trade-off, not un-usability.

| Dataset | Videos | Images / Frames | Notes |
|---|---|---|---|
| Cholec80 | 80 | ~91K frames (1 fps) | ✅ have (`cholec80`) |
| CholecTrack20 | 20 | 35K+ frames (1 fps) | |
| CholecT50 | 50 | ~101K frames (1 fps) | ✅ have as triplet probe |
| SurgPose | — | ~120K instances | frames-only |
| CholecT45 | 45 | ~90K frames (1 fps) | |
| PolypDB | — | 3,934 images | frames-only |
| JIGSAWS | 103 | ~193K frames (30 fps) | ✅ have (`jigsaw`) |
| Kvasir-SEG | — | 1,000 images | frames-only |
| Endomapper | 96 | videos | |
| LDPolypVideo | 263 | 901,587 frames | |
| Kvasir-Capsule | 117 | 4.7M | |
| Galar | 80 | 3,513,539 frames | |
| SurgiSR4K | 50 | 800 high-res images (4K) | |
| WCEbleedGen | — | 2,618 frames | frames-only |
| AutoLaparo | 21 | ~2,000K+ frames | |
| BM-BronchoLC | 208 | 2,132 images | frames-only |
| SAR-RARP50 | 50 | 16,250 frames | ✅ have as action-seg probe |
| HS-CMU | 175 | 3,385 images | frames-only |
| Cataract-1K | 1,000 | 2,256 annotated frames | |
| PitVQA | 25 | N/A | |
| Cataract-101 | 101 | ~100K+ frames | |
| Spine Endoscopic Atlas | 119 | 48,510 images | |
| CaDIS | — | 4,670 images | frames-only |
| AVOS | 1,997 | videos | |
| PolypGen | 2,225 | 8,037 images | frames-only |
| Rectal Cancer Surgery | 77 | N/A | |
| EndoSLAM | — | ~601,000 images | frames-only |
| SARAS-MESAD | 4 | N/A | |
| SCARED | 9 | keyframes + depth maps | |
| ATLAS Dione | 86 | ~910 action clips | |
| CholecInstanceSeg | 85 | 41,933 frames | |
| LapEx | 30 | N/A | |
| Endoscapes | 201 | 11,090 frames (CVS201) | |
| EgoSurgery-Phase | 20 | 1,350,000 (25 fps) | |
| MultiBypass140 | 140 | N/A | |
| SurgBench | 225 / 25 | 53M frames | |
| HeiChole | 33 | N/A | |
| SurgPub-Video | ~3,000 | 25M annotated frames | |
| M2CAI16 Workflow | 41 | N/A | |
| SimuScope | — | synthetic dataset | |
| SurgVU / SurgToolLoc | 280 | ~18M frames (60 fps) | ✅ have (`surgvu24_clean`, `surgtoolloc2022`) |
| Syn-ISS | — | 3,000 simulated images | frames-only |
| CRCD (Expanded) | 21 | 127,000 annotated frames | ✅ have (`crcd`, older cut) |
| LEMON | 4,194 | 3.4M frames | staged (`lemon_staging`), not yet mixed |
