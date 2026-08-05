"""Regression guard for the fine-tune encoder-resume bug.

History: partial-unfreeze probes fine-tune the last-N encoder blocks.
`save_checkpoint` persisted the tuned encoder under checkpoint["encoder"], but
`load_checkpoint` never restored it on resume -- it only reloaded the classifier
heads + optimizer + scaler. So every PBS requeue silently reset the encoder to
its pretrained weights while keeping the trained head, epoch counter, and (worse)
the stale Adam moments. Any "unfreeze doesn't help / regresses" conclusion drawn
from a requeued run was suspect until this was fixed, since the run was silently
not a continuous fine-tune.

These CPU-only tests exercise the real `load_checkpoint` and assert the invariant
that broke: a checkpoint that carries an "encoder" key restores those weights
into the live encoder. Frozen runs (no "encoder" key, encoder=None) must be
unaffected.
"""
import copy

import torch
import torch.nn as nn

from evals.video_classification_frozen.eval import load_checkpoint


def _tiny_head():
    return nn.Linear(4, 3)


def _tiny_encoder():
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


def _save_ckpt(path, classifiers, optimizer, epoch, encoder=None):
    d = {
        "classifiers": [c.state_dict() for c in classifiers],
        "opt": [o.state_dict() for o in optimizer],
        "scaler": None,
        "epoch": epoch,
    }
    if encoder is not None:
        d["encoder"] = encoder.state_dict()
    torch.save(d, path)


def _randomize(module):
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p))


def test_finetuned_encoder_is_restored_on_resume(tmp_path):
    # Emulate a fine-tune checkpoint: the encoder has TRAINED (non-pretrained)
    # weights that must survive resume.
    trained_encoder = _tiny_encoder()
    _randomize(trained_encoder)
    trained_snapshot = copy.deepcopy(trained_encoder.state_dict())

    head = _tiny_head()
    opt = torch.optim.Adam(head.parameters())
    ckpt = tmp_path / "latest.pt"
    _save_ckpt(ckpt, [head], [opt], epoch=1, encoder=trained_encoder)

    # Fresh process: encoder starts at PRETRAINED (different) weights.
    live_encoder = _tiny_encoder()  # different init
    live_head = _tiny_head()
    live_opt = torch.optim.Adam(live_head.parameters())
    assert not torch.equal(
        live_encoder[0].weight, trained_snapshot["0.weight"]
    ), "precondition: live encoder must differ before resume"

    load_checkpoint(
        device=torch.device("cpu"),
        r_path=str(ckpt),
        classifiers=[live_head],
        opt=[live_opt],
        scaler=None,
        encoder=live_encoder,
    )

    # The fix: the tuned encoder weights are now in the live encoder.
    for k, v in trained_snapshot.items():
        assert torch.equal(live_encoder.state_dict()[k], v), f"encoder param {k} not restored"


def test_frozen_run_unaffected_without_encoder_key(tmp_path):
    # Frozen probe: checkpoint has NO "encoder" key and encoder=None. Must load
    # heads + opt fine and not raise.
    head = _tiny_head()
    _randomize(head)
    head_snapshot = copy.deepcopy(head.state_dict())
    opt = torch.optim.Adam(head.parameters())
    ckpt = tmp_path / "latest.pt"
    _save_ckpt(ckpt, [head], [opt], epoch=3, encoder=None)

    live_head = _tiny_head()
    live_opt = torch.optim.Adam(live_head.parameters())
    _, _, _, epoch, _ = load_checkpoint(
        device=torch.device("cpu"),
        r_path=str(ckpt),
        classifiers=[live_head],
        opt=[live_opt],
        scaler=None,
        encoder=None,
    )
    assert epoch == 3
    for k, v in head_snapshot.items():
        assert torch.equal(live_head.state_dict()[k], v), f"head param {k} not restored"


def test_resume_parity_one_plus_one_equals_two(tmp_path):
    """1 step + resume + 1 step == 2 uninterrupted steps (encoder weights).

    This is the end-to-end invariant the reviewer asked for, at unit scale:
    the fine-tuned encoder plus stale Adam moments must carry across a
    checkpoint/reload so the resumed run is a CONTINUOUS fine-tune. With the
    bug, the resumed encoder starts from pretrained weights and diverges.
    """

    def _one_step(enc, head, opt, x, y):
        opt.zero_grad()
        out = head(enc(x).mean(0, keepdim=True))
        loss = ((out - y) ** 2).sum()
        loss.backward()
        opt.step()

    x = torch.randn(5, 4)
    y = torch.randn(1, 3)

    # Shared starting point for both arms (same encoder + head init).
    enc_init = _tiny_encoder()
    _randomize(enc_init)
    enc_init_state = copy.deepcopy(enc_init.state_dict())
    head_init_state = copy.deepcopy(_tiny_head().state_dict())

    def _fresh():
        enc = _tiny_encoder(); enc.load_state_dict(enc_init_state)
        head = _tiny_head(); head.load_state_dict(head_init_state)
        opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()))
        return enc, head, opt

    # --- Uninterrupted: two consecutive steps ---
    enc_u, head_u, opt_u = _fresh()
    _one_step(enc_u, head_u, opt_u, x, y)
    _one_step(enc_u, head_u, opt_u, x, y)
    final_uninterrupted = copy.deepcopy(enc_u.state_dict())

    # --- Interrupted: step, checkpoint (encoder + heads + opt), reload, step ---
    enc_i, head_i, opt_i = _fresh()
    _one_step(enc_i, head_i, opt_i, x, y)
    ckpt = tmp_path / "latest.pt"
    torch.save(
        {
            "classifiers": [head_i.state_dict()],
            "opt": [opt_i.state_dict()],
            "scaler": None,
            "epoch": 1,
            "encoder": enc_i.state_dict(),
        },
        ckpt,
    )
    # Fresh objects at PRETRAINED init -- the bug would leave enc_r here.
    enc_r = _tiny_encoder(); enc_r.load_state_dict(enc_init_state)
    head_r = _tiny_head()
    opt_r = torch.optim.Adam(list(enc_r.parameters()) + list(head_r.parameters()))
    load_checkpoint(
        device=torch.device("cpu"),
        r_path=str(ckpt),
        classifiers=[head_r],
        opt=[opt_r],
        scaler=None,
        encoder=enc_r,
    )
    _one_step(enc_r, head_r, opt_r, x, y)
    final_resumed = enc_r.state_dict()

    for k in final_uninterrupted:
        assert torch.allclose(final_uninterrupted[k], final_resumed[k], atol=1e-6), (
            f"encoder param {k} diverged after resume -- FT is not continuous"
        )
