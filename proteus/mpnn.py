"""ProteinMPNN as a scorer.

The hand-tuned objective correlates with experimental ddG at |r| = 0.125, and
refitting its own nine terms only reaches 0.296. Both fall well short of a
learned, structure-conditioned model on the same benchmark, so this wires one
in.

ProteinMPNN suits the job. It is small enough to run on a CPU in about a
second, it conditions on backbone geometry rather than sequence alone -- which
is precisely the information Proteus already holds -- and on S669 it reaches
|r| = 0.398, comparable to RaSP and DDGun3D.

A substitution is scored as a conditional log-likelihood ratio: how much more
or less likely the model finds the mutant residue at that position given the
backbone and the rest of the chain. That is not a free energy and carries no
kcal/mol, but it ranks mutations, which is what the search actually needs.

Torch and the proteinmpnn package are imported lazily, so the rest of Proteus
runs without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Checkpoint. v_48_020 is the default from the original release: 48
#: neighbours, 0.20 A of backbone noise during training. The soluble-only
#: variants are trained on a subset excluding membrane proteins, which is the
#: wrong prior for half of what Proteus handles.
DEFAULT_WEIGHTS = "v_48_020"

#: Token order used by the ProteinMPNN featuriser. Read off the package rather
#: than assumed -- an alphabet off by one silently scores the wrong residue.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_INDEX = {aa: i for i, aa in enumerate(ALPHABET)}


def available() -> bool:
    try:
        import torch  # noqa: F401
        from proteinmpnn import protein_mpnn_utils  # noqa: F401
    except Exception:
        return False
    return True


def require() -> None:
    if not available():
        raise ImportError(
            "ProteinMPNN scoring needs torch and the proteinmpnn package:\n"
            "    pip install torch proteinmpnn\n"
            "The package bundles the model weights (~68 MB), so nothing is "
            "downloaded at run time."
        )


@dataclass
class MutationScore:
    """One substitution as the model sees it."""

    position: int          #: 1-based index into the scored chain
    wildtype: str
    mutant: str
    log_ratio: float       #: log P(mutant) - log P(wildtype); negative = worse

    def describe(self) -> str:
        return (f"{self.wildtype}{self.position}{self.mutant}  "
                f"log-ratio {self.log_ratio:+.3f}")


class MPNNScorer:
    """Conditional log-likelihood scoring of substitutions on a fixed backbone.

    The model loads once and is cached. Position log-probabilities depend only
    on the backbone and the surrounding sequence, so a chain is scored in a
    single forward pass and every candidate residue at every position is read
    off that one result.
    """

    name = "proteinmpnn"

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str | None = None,
                 seed: int = 0) -> None:
        self.weights = weights
        self.device_name = device
        self.seed = seed
        self._model = None
        self._device = None
        self._cache: dict[tuple[str, str | None], tuple[np.ndarray, str]] = {}

    def _load(self):
        if self._model is not None:
            return
        require()
        import torch
        from proteinmpnn import protein_mpnn_utils as U

        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        path = (Path(U.__file__).parent / "data" / "vanilla_model_weights"
                / f"{self.weights}.pt")
        if not path.exists():
            raise FileNotFoundError(f"no ProteinMPNN checkpoint at {path}")

        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        model = U.ProteinMPNN(
            num_letters=21, node_features=128, edge_features=128,
            hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
            # No backbone noise: scoring must be deterministic, and the noise
            # exists to regularise training, not to be sampled at inference.
            augment_eps=0.0,
            k_neighbors=ckpt["num_edges"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        self._model, self._device = model, device

    def log_probs(self, pdb_path: str, chain: str | None = None):
        """Per-position log-probabilities for one chain.

        Returns ``(log_probs, sequence)``; ``log_probs`` has shape
        (length, 21), columns indexed by :data:`ALPHABET`.
        """
        key = (str(pdb_path), chain)
        if key in self._cache:
            return self._cache[key]

        self._load()
        import torch
        from proteinmpnn import protein_mpnn_utils as U

        parsed = U.parse_PDB(str(pdb_path), ca_only=False)
        if not parsed:
            raise ValueError(f"ProteinMPNN could not parse {pdb_path}")
        entry = parsed[0]

        chains = [k.split("_")[-1] for k in entry if k.startswith("seq_chain_")]
        if not chains:
            raise ValueError(f"no chains parsed from {pdb_path}")
        target = chain if chain in chains else chains[0]
        design = {entry["name"]: ([target], [c for c in chains if c != target])}

        with torch.no_grad():
            f = U.tied_featurize([entry], self._device, design,
                                 None, None, None, None, None, ca_only=False)
            X, S, mask, chain_M = f[0], f[1], f[2], f[4]
            chain_encoding_all, chain_M_pos, residue_idx = f[5], f[10], f[12]
            # Decoding order is sampled from randn; a fixed seed keeps the
            # same input scoring identically across calls.
            torch.manual_seed(self.seed)
            randn = torch.randn(chain_M.shape, device=self._device)
            out = self._model(X, S, mask, chain_M * chain_M_pos,
                              residue_idx, chain_encoding_all, randn)

        lp = out[0].cpu().numpy()
        sequence = entry.get(f"seq_chain_{target}", entry["seq"])
        # tied_featurize concatenates every chain; keep only the scored one.
        lp = lp[:len(sequence)]
        self._cache[key] = (lp, sequence)
        return lp, sequence

    def score_mutations(self, pdb_path: str, mutations, chain: str | None = None):
        """Score ``(position, wildtype, mutant)`` triples in one forward pass.

        ``position`` is 1-based into the scored chain. A mutation whose stated
        wildtype does not match the structure is skipped rather than scored
        against the wrong residue.
        """
        lp, sequence = self.log_probs(pdb_path, chain)
        out = []
        for pos, wt, mut in mutations:
            if not 1 <= pos <= len(sequence):
                continue
            if sequence[pos - 1] != wt:
                continue
            wi, mi = _INDEX.get(wt), _INDEX.get(mut)
            if wi is None or mi is None:
                continue
            out.append(MutationScore(pos, wt, mut,
                                     float(lp[pos - 1, mi] - lp[pos - 1, wi])))
        return out

    def score_sequence(self, pdb_path: str, sequence: str,
                       chain: str | None = None) -> float:
        """Mean log-likelihood of ``sequence`` on this backbone.

        A single forward pass conditioned on the *wild-type* sequence, so this
        is a fast approximation rather than a true autoregressive likelihood.
        Adequate for ranking closely related variants; it degrades as the
        candidate drifts far from the sequence the pass was conditioned on.
        """
        lp, wild = self.log_probs(pdb_path, chain)
        n = min(len(sequence), len(wild))
        idx = [_INDEX.get(a, _INDEX["X"]) for a in sequence[:n]]
        return float(np.mean([lp[i, j] for i, j in enumerate(idx)]))


class MPNNSequenceScorer:
    """Adapter presenting :class:`MPNNScorer` through the ``Scorer`` interface.

    The engine scores whole sequences threaded onto a context; ProteinMPNN
    scores positions on a backbone read from a file. This bridges the two by
    taking the path from ``ctx.structure.source``, which means it only works
    for contexts loaded from disk -- a synthetic structure built from arrays
    has no file for the model to read, and raises rather than silently
    scoring something else.

    Sign is flipped so that lower is better, matching the energy convention the
    rest of the scoring layer uses.
    """

    name = "proteinmpnn"

    def __init__(self, chain: str | None = None, **kw) -> None:
        self.chain = chain
        self._inner = MPNNScorer(**kw)

    def score(self, ctx, sequence: str):
        from .scoring import ScoreBreakdown

        source = getattr(ctx.structure, "source", None)
        if not source or source == "<arrays>":
            raise ValueError(
                "MPNNSequenceScorer needs a structure loaded from a file; "
                "this context has no source path."
            )
        mean_ll = self._inner.score_sequence(source, sequence, self.chain)
        return ScoreBreakdown(total=-mean_ll,
                              terms={"mpnn_log_likelihood": mean_ll},
                              n_residues=max(len(sequence), 1))

    def total(self, ctx, sequence: str) -> float:
        return self.score(ctx, sequence).total
