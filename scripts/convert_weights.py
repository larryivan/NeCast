"""Build necast/weights/necast.pt from the published model.

The published model is final/checkpoint.pt at https://huggingface.co/Larrivhan/NeCast
(sha256 66439a336847122deb2f13fb96e407aeff4fddd593a2168d1ca580e1a0c28ab5), saved by
the research code, with the constants of its training windows in
production_moments.npz (sha256 f486bcf02daa9c34b334b3ae46cd7f4caf56e1669ea30f0221aaaca798bd3d82).

This renames the tensors to the names in necast/network.py and adds the
constants. Three tensors are left out: concat_projection.weight, which the
published model multiplies by zero, and two coordinate buffers that the network
recomputes with torch.linspace.

    python scripts/convert_weights.py checkpoint.pt production_moments.npz necast/weights/necast.pt
"""

import sys

import numpy as np
import torch

from necast.network import CONTEXT, Network

# research name -> package name, by prefix; the first match wins
RENAME = [
    ("raw.site_projection.", "site_mlp."), ("raw.left_parent.", "tree_left."),
    ("raw.right_parent.", "tree_right."), ("raw.parent_norm.", "tree_norm."),
    ("raw.macro_to_individual.", "to_individual."), ("raw.long_blocks.", "individual_blocks."),
    ("raw.carrier_input_norm.", "across.norm_in."), ("raw.carrier_qkv.", "across.qkv."),
    ("raw.carrier_attention_output.", "across.attention_out."),
    ("raw.carrier_attention_norm.", "across.norm_attention."), ("raw.carrier_ffn_norm.", "across.norm_ffn."),
    ("raw.carrier_ffn.0.", "across.ffn.0."), ("raw.carrier_ffn.3.", "across.ffn.2."),
    ("raw.carrier_projection.", "across.pool."), ("raw.map_projection.", "map_mlp."),
    ("raw.region_projection.", "region_mlp."), ("raw.regional_blocks.", "region_blocks."),
    ("raw.region_downsample.0.0.", "downsample.0."), ("raw.region_downsample.1.0.", "downsample.1."),
    ("raw.region_downsample.2.0.", "downsample.2."), ("raw.physics_attention_bias", "time_scale_bias"),
    ("raw.context_projection.", "context_mlp."), ("raw.output_norm.", "head_norm."),
    ("raw.output_head.0.", "head.0."), ("raw.output_head.3.", "head.2."), ("raw.", ""),
    ("summary.context_projection.", "summary_encoder.context_mlp."),
    ("summary.sfs_encoder.0.", "summary_encoder.sfs.0.0."), ("summary.sfs_encoder.2.", "summary_encoder.sfs.0.2."),
    ("summary.sfs_encoder.3.", "summary_encoder.sfs.1."),
    ("summary.ld_stem.", "summary_encoder.ld_convs."), ("summary.ld_projection.", "summary_encoder.ld_pool."),
    ("summary.ibs_length_stem.", "summary_encoder.ibs_convs."),
    ("summary.ibs_region_encoder.", "summary_encoder.ibs_region."),
    ("summary.ibs_pyramid_projection.", "summary_encoder.ibs_pyramid."),
    ("summary.sidecar_encoder.0.", "summary_encoder.posterior.0.0."),
    ("summary.sidecar_encoder.2.", "summary_encoder.posterior.0.2."),
    ("summary.sidecar_encoder.3.", "summary_encoder.posterior.1."),
    ("summary.summary_time_basis", "summary_encoder.time_basis"),
    ("summary.summary_time_norm.", "summary_encoder.time_norm."), ("additive_projection.", "fusion."),
]
LEFT_OUT = {"concat_projection.weight", "summary.ld_distance_coordinate", "summary.ibs_length_coordinate"}


def main(checkpoint, moments, out):
    old = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    assert torch.equal(old["summary.ld_distance_coordinate"], torch.linspace(-1.0, 1.0, 26))
    assert torch.equal(old["summary.ibs_length_coordinate"], torch.linspace(-1.0, 1.0, 27))
    state = {}
    for key, value in old.items():
        if key not in LEFT_OUT:
            prefix, name = next(((o, n) for o, n in RENAME if key.startswith(o)), ("", ""))
            state[name + key[len(prefix):]] = value

    m = np.load(moments)
    # the network feeds the summaries a fixed context: check it is what training saw
    context = CONTEXT.numpy()
    assert np.array_equal(context, np.array([np.log10(30_000_000), 6.0], np.float32))
    assert np.all((context.astype(np.float64) - m["context_mean"][0]) / m["context_std"][0] == 0)
    for name in ("sfs", "ld", "ibs"):
        state[f"{name}_mean"], state[f"{name}_std"] = (torch.from_numpy(m[f"{name}_{x}"][0]) for x in ("mean", "std"))
    state["posterior_mean"] = torch.from_numpy(m["sidecar_mean"])
    state["posterior_std"] = torch.from_numpy(np.where(m["sidecar_std"] > 0, m["sidecar_std"], 1.0))
    state["offset"] = torch.from_numpy(m["residual_anchor"])

    network = Network()
    network.load_state_dict(state, strict=True)
    torch.save(network.state_dict(), out)
    print(f"{out}: {sum(p.numel() for p in network.parameters()):,} parameters")


if __name__ == "__main__":
    main(*sys.argv[1:])
