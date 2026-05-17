"""
Optimizer gaming a misspecified objective. The policy has two tasks (MATH,
GREET) and an add() tool that only helps with MATH. The adjusted objective is

    r_adjusted = r_true + w * is_tool_used

where w is a dial we set by hand, not a measured price. When w is large
enough to outweigh the true reward, the policy uses the tool on every task
and true reward drops.

TODO: test what happens with a randomly generated bonus per sim call.
"""

import gc
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


class TaskType(IntEnum):
    MATH = 0
    GREET = 1


class ActionType(Enum):
    USE_TOOL = "USE_TOOL"
    EMIT = "EMIT"


@dataclass
class Action:
    action_type: ActionType
    action_result: str | None


@dataclass
class Task:
    task_type: TaskType
    operands: tuple[int, int] | None  # used for MATH, None for GREET


@dataclass
class SimulationOutput:
    reward: float
    is_tool_used: bool


ACTIONS: list[Action] = [
    Action(ActionType.USE_TOOL, None),
    Action(ActionType.EMIT, "hello"),
    *[Action(ActionType.EMIT, str(i)) for i in range(19)],
]

NUM_ACTIONS = len(ACTIONS)
NUM_TASKS = len(TaskType)


class Policy(nn.Module):
    def __init__(self, num_tasks: int, num_actions: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_tasks, num_actions))

    def forward(self, task_idx: int) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits[task_idx])


@dataclass
class TrainArtifact:
    policy: Policy
    task_history: list[Task]
    tool_use_history: list[int]
    reward_history: list[float]


def sample_task() -> Task:
    if torch.rand(1).item() < 0.5:
        a = int(torch.randint(0, 10, (1,)).item())
        b = int(torch.randint(0, 10, (1,)).item())
        return Task(TaskType.MATH, (a, b))
    return Task(TaskType.GREET, None)


def compute_reward(task: Task, emitted: str) -> float:
    match task.task_type:
        case TaskType.MATH:
            target = str(task.operands[0] + task.operands[1])
        case TaskType.GREET:
            target = "hello"
    return 1.0 if target in emitted else 0.0


def add(a: int, b: int) -> int:
    return a + b


def simulate(task: Task, action: Action) -> SimulationOutput:
    match action.action_type:
        case ActionType.USE_TOOL:
            emitted = str(add(*task.operands)) if task.operands else ""
        case ActionType.EMIT:
            emitted = action.action_result

    return SimulationOutput(
        reward=compute_reward(task, emitted),
        is_tool_used=action.action_type == ActionType.USE_TOOL,
    )


def train(
    num_steps: int = 4000, lr: float = 0.1, tool_use_bonus: float = 0.0, seed: int = 0
) -> TrainArtifact:
    torch.manual_seed(seed)
    policy = Policy(NUM_TASKS, NUM_ACTIONS)
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)

    task_history, tool_use_history, reward_history = [], [], []
    for _ in range(num_steps):
        optimizer.zero_grad()

        task = sample_task()
        dist = policy(task.task_type)
        action_idx_tensor = dist.sample()
        action = ACTIONS[action_idx_tensor.item()]

        simulation_output = simulate(task, action)
        adjusted_reward = simulation_output.reward + (
            tool_use_bonus if simulation_output.is_tool_used else 0.0
        )
        # REINFORCE
        loss = -dist.log_prob(action_idx_tensor) * adjusted_reward

        loss.backward()
        optimizer.step()

        task_history += [task]
        tool_use_history += [int(simulation_output.is_tool_used)]
        reward_history += [simulation_output.reward]

    return TrainArtifact(
        policy=policy,
        task_history=task_history,
        tool_use_history=tool_use_history,
        reward_history=reward_history,
    )


def main():
    artifact_without_bonus = train(tool_use_bonus=0.0)
    gc.collect()  # free tensors from previous run
    artifact_with_bonus = train(tool_use_bonus=1.5)

    def compute_per_task_tool_rate(artifact: TrainArtifact) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "task": [t.task_type.name for t in artifact.task_history],
                "tool": artifact.tool_use_history,
            }
        )
        df["smoothed"] = df.groupby("task")["tool"].transform(
            lambda s: s.ewm(alpha=0.02).mean()
        )
        return df.pivot(columns="task", values="smoothed")

    rewards = (
        pd.DataFrame(
            {
                "without bonus": artifact_without_bonus.reward_history,
                "with bonus": artifact_with_bonus.reward_history,
            }
        )
        .ewm(alpha=0.02)
        .mean()
    )

    tool_without_bonus = compute_per_task_tool_rate(artifact_without_bonus).add_suffix(
        " without bonus"
    )
    tool_with_bonus = compute_per_task_tool_rate(artifact_with_bonus).add_suffix(
        " with bonus"
    )
    tool = pd.concat([tool_without_bonus, tool_with_bonus], axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    rewards.plot(ax=ax[0], title="Reward")
    tool.plot(ax=ax[1], title="Tool-call rate by task")
    ax[0].set_xlabel("iteration")
    ax[1].set_xlabel("iteration")
    fig.text(
        0.5,
        0.02,
        "Bonus speeds up tool adoption on MATH but also makes the policy use the tool on GREET. True reward stalls near 0.5.",
        ha="center",
        fontsize=11,
    )
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(Path(__file__).parent / "reward_hacking_curves.png")

    for label, artifact in [
        ("without bonus", artifact_without_bonus),
        ("with bonus", artifact_with_bonus),
    ]:
        print(f"\n[{label}]")
        for task_type in TaskType:
            probs = artifact.policy(task_type).probs.detach().numpy()
            print(f" {task_type.name}")
            for action, p in zip(ACTIONS, probs):
                print(f"  {action}: {p:.3f}")


if __name__ == "__main__":
    main()
