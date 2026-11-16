"""First source downcrossings in reference test worlds, reconstructed from trajectories."""
import concurrent.futures
import json
import numpy as np
from .complete_analysis import prepare, load_world
from .runtime import P, OUT, WORKERS


def first_crossings(task):
    _, pressure, _, index, _, trajectory, world_path, *_ = task
    world, _ = load_world(world_path)
    with np.load(P / trajectory, allow_pickle=False) as data:
        stock = data['stocks'][:500] / world['capacity']
        accessible = data['accessible']
    previous = np.concatenate([np.ones((1, stock.shape[1])), stock[:-1]])
    events = {}
    for threshold in (.25, .5, .75):
        crossing = (previous >= threshold) & (stock < threshold) & accessible[None, :]
        events[str(threshold)] = np.where(crossing.any(axis=0), crossing.argmax(axis=0), -1).tolist()
    return str(pressure) + '|' + str(index), events


def main():
    tasks = [t for t in prepare() if t[0] == 'test' and t[2] == 'db0eed394e20c822b2a6beb3']
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        events = dict(pool.map(first_crossings, tasks, chunksize=4))
    (OUT / 'baseline_events.json').write_text(json.dumps(events, sort_keys=True))
    print(f'Reconstructed source cohorts from {len(events)} reference trajectories', flush=True)


if __name__ == '__main__':
    main()
