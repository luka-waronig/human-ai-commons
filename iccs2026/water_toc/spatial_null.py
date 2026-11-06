"""Conditional random-label null for the smoothed cooperation field.

Agent positions, occupancy, and the cooperator count within each agent type are
fixed. This accounts for spatial autocorrelation induced by overlapping sensing
neighborhoods. It tests clustering, not a causal mechanism of cooperation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from water_toc.model import WaterToC
from water_toc.reproducibility import write_manifest


def conditional_moran_test(agents, width, height, permutations=499, seed=42):
    if permutations < 1:
        raise ValueError('at least one permutation is required')
    empty = dict(observed=np.nan, null_mean=np.nan, excess=np.nan,
                 p_upper=np.nan, valid_permutations=0)
    if not agents:
        return empty
    points = np.array([[a[2],a[3]] for a in agents], dtype=float)
    occupied = np.unique(points, axis=0)
    if len(occupied) < 2:
        return empty
    def distances(a,b):
        delta = np.abs(a[:,None,:] - b[None,:,:])
        delta = np.minimum(delta, np.array([width,height]) - delta)
        return delta.max(axis=2)
    neighborhood = distances(occupied,points) <= 2
    weights = neighborhood / neighborhood.sum(axis=1,keepdims=True)
    distance = distances(occupied,occupied)
    adjacency = ((distance > 0) & (distance <= 1)).astype(float)
    total_weight = adjacency.sum()
    if total_weight == 0:
        return empty
    cooperative = np.array([str(a[4]).upper().startswith('C') for a in agents],dtype=float)
    labels = np.repeat(cooperative[:,None],permutations + 1,axis=1)
    rng = np.random.default_rng(seed)
    kinds = np.array([a[1] for a in agents])
    for kind in np.unique(kinds):
        indices = np.flatnonzero(kinds == kind)
        for column in range(1,permutations + 1):
            labels[indices,column] = rng.permutation(cooperative[indices])
    fields = weights @ labels
    centered = fields - fields.mean(axis=0)
    denominator = (centered**2).sum(axis=0)
    numerator = (centered * (adjacency @ centered)).sum(axis=0)
    scores = np.divide(len(occupied) * numerator, total_weight * denominator,
                       out=np.full(permutations+1,np.nan),where=denominator > 1e-24)
    observed, null = scores[0], scores[1:]
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(null):
        return empty
    return dict(observed=float(observed),null_mean=float(null.mean()),
                excess=float(observed-null.mean()),
                p_upper=float((1 + np.sum(null >= observed)) / (1 + len(null))),
                valid_permutations=len(null))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--permutations',type=int,default=499)
    parser.add_argument('--output',type=Path,default=Path('results/reproducibility_audit/spatial_null.json'))
    args = parser.parse_args()
    model = WaterToC(seed=args.seed,save_snapshots=False,record_spatial=False)
    for _ in range(args.steps):model.step()
    result = conditional_moran_test(model._get_agent_pos_strategies_list(),model.width,model.height,
                                    args.permutations,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Undefined spatial statistics are represented explicitly by null in JSON.
    serializable = {key:(value if np.isfinite(value) else None) for key,value in result.items()}
    args.output.write_text(json.dumps(serializable,indent=2,allow_nan=False)+'\n')
    write_manifest(args.output.with_suffix('.manifest.json'),
        dict(seed=args.seed,steps=args.steps,permutations=args.permutations,
             null='shuffle strategies within agent type, fixed positions and occupancy'))
    print(json.dumps(serializable,indent=2))


if __name__ == '__main__':
    main()
