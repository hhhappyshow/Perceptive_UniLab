"""Convert an InstinctMJ navigation checkpoint to the UniLab rsl_rl format.

InstinctMJ (instinct_rl fork) saves::

    model_state_dict:              {std, actor.<i>.*, critic.<i>.*}
    policy_normalizer_state_dict:  {_mean, _var, _std}

UniLab's rsl_rl PPO expects::

    actor_state_dict:              {mlp.<i>.*, obs_normalizer.*, distribution.std_param}

Only the actor path is required for playback (the critic is not loaded).
Keys keep the same layer indices because both stacks build the actor as a
``Linear/ELU`` ``nn.Sequential``; only the module attribute name differs
(``actor`` vs ``mlp``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def convert(input_path: str, output_path: str | None = None) -> Path:
    source = Path(input_path).expanduser().resolve(strict=True)
    loaded = torch.load(source, map_location="cpu", weights_only=True)
    model_state = loaded["model_state_dict"]
    normalizer = loaded["policy_normalizer_state_dict"]

    actor_state: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        if key == "std":
            actor_state["distribution.std_param"] = value
            continue
        if key.startswith("actor."):
            actor_state["mlp." + key.removeprefix("actor.")] = value

    if "mlp.0.weight" not in actor_state:
        raise ValueError(f"No actor weights found in {source}")
    input_dim = int(actor_state["mlp.0.weight"].shape[1])
    output_dim = int(actor_state["mlp.6.weight"].shape[0])

    for name in ("_mean", "_var", "_std"):
        if name not in normalizer:
            raise ValueError(f"policy normalizer is missing {name!r} in {source}")
        actor_state[f"obs_normalizer.{name}"] = normalizer[name].reshape(1, -1).float()
    actor_state["obs_normalizer.count"] = torch.tensor(1, dtype=torch.int64)

    converted = {
        "actor_state_dict": actor_state,
        "iter": loaded.get("iter", 0),
        "infos": {
            "converted_from": str(source),
            "actor_input_dim": input_dim,
            "actor_output_dim": output_dim,
        },
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path
        else source.with_name(source.stem + "_unilab.pt")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination)
    print(f"[convert] {source} -> {destination}")
    print(f"[convert] actor input dim {input_dim}, output dim {output_dim}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="InstinctMJ model_XXXX.pt path")
    parser.add_argument("-o", "--output", default=None, help="Converted checkpoint path")
    arguments = parser.parse_args()
    convert(arguments.checkpoint, arguments.output)


if __name__ == "__main__":
    main()