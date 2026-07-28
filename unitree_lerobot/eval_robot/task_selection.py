from collections.abc import Iterable
from typing import Any


def _iter_episode_records(episodes: Any) -> Iterable[dict[str, Any]]:
    if isinstance(episodes, list):
        yield from episodes
        return

    if isinstance(episodes, dict):
        length = len(next(iter(episodes.values()), []))
        for index in range(length):
            yield {key: value[index] for key, value in episodes.items()}
        return

    for index in range(len(episodes)):
        yield episodes[index]


def _episode_tasks(episode: dict[str, Any]) -> list[str]:
    tasks = episode.get("tasks", [])
    if isinstance(tasks, str):
        return [tasks]
    return list(tasks)


def list_unique_tasks(episodes: Any) -> list[str]:
    tasks = []
    seen = set()
    for episode in _iter_episode_records(episodes):
        for task in _episode_tasks(episode):
            if task in seen:
                continue
            seen.add(task)
            tasks.append(task)
    return tasks


def select_initial_step(dataset: Any, task: str | None) -> tuple[dict[str, Any], str]:
    requested_task = (task or "").strip()
    episodes = dataset.meta.episodes

    if not requested_task:
        from_idx = episodes["dataset_from_index"][0] if isinstance(episodes, dict) else episodes[0]["dataset_from_index"]
        step = dataset[from_idx]
        return step, step.get("task", "")

    available_tasks = []
    for episode in _iter_episode_records(episodes):
        tasks = _episode_tasks(episode)
        available_tasks.extend(tasks)
        if requested_task in tasks:
            return dataset[episode["dataset_from_index"]], requested_task

    unique_tasks = sorted(set(available_tasks))
    raise ValueError(
        f"Task {requested_task!r} was not found in dataset metadata. Available tasks: {unique_tasks}"
    )
