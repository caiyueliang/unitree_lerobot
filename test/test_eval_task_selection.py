from unitree_lerobot.eval_robot.task_selection import list_episode_tasks, select_initial_step


class FakeDataset:
    def __init__(self):
        self.meta = type(
            "Meta",
            (),
            {
                "episodes": [
                    {"dataset_from_index": 0, "tasks": ["sort apple"]},
                    {"dataset_from_index": 10, "tasks": ["sort banana"]},
                ]
            },
        )()
        self.steps = {
            0: {"task": "sort apple"},
            10: {"task": "sort banana"},
        }

    def __getitem__(self, index):
        return self.steps[index]


def test_select_initial_step_uses_first_episode_when_task_is_empty():
    step, task = select_initial_step(FakeDataset(), "")

    assert step == {"task": "sort apple"}
    assert task == "sort apple"


def test_select_initial_step_uses_matching_episode_for_task_override():
    step, task = select_initial_step(FakeDataset(), "sort banana")

    assert step == {"task": "sort banana"}
    assert task == "sort banana"


def test_list_episode_tasks_returns_tasks_for_each_episode():
    assert list_episode_tasks(FakeDataset().meta.episodes) == [
        (0, ["sort apple"]),
        (1, ["sort banana"]),
    ]
