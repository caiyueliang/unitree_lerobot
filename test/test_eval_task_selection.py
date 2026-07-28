from unitree_lerobot.eval_robot.task_selection import list_unique_tasks, select_initial_step


class FakeDataset:
    def __init__(self):
        self.meta = type(
            "Meta",
            (),
            {
                "episodes": [
                    {"dataset_from_index": 0, "tasks": ["sort apple"]},
                    {"dataset_from_index": 10, "tasks": ["sort banana"]},
                    {"dataset_from_index": 20, "tasks": ["sort apple", "sort orange"]},
                ]
            },
        )()
        self.steps = {
            0: {"task": "sort apple"},
            10: {"task": "sort banana"},
            20: {"task": "sort apple"},
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


def test_list_unique_tasks_deduplicates_tasks():
    assert list_unique_tasks(FakeDataset().meta.episodes) == [
        "sort apple",
        "sort banana",
        "sort orange",
    ]
