from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset


HUMAN_PREFERENCE_DATA_PATH = "./benchmarks/preference_benchmark.json"
TRUTHFUL_QA_DATA_PATH = "./benchmarks/truthful_qa_benchmark.json"
SAFETY_DATA_PATH = "./benchmarks/pku_benchmark_answers.json"
VERBOSE_DATA_PATH = "./benchmarks/verbose_benchmark.json"
REWARD_BENCH_HF_PATH = "allenai/reward-bench"


def _to_message_format(prompt: str, response: str) -> list:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


class BaseBenchmark(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.df = pd.read_json(self.benchmarks_file)

        self.samples = []

        for index, row in self.df.iterrows():
            string_a = _to_message_format(row["prompt"], row["response_a"])
            string_b = _to_message_format(row["prompt"], row["response_b"])
            self.samples.append(string_a)
            self.samples.append(string_b)
            if self.responses_per_question == 3:
                string_c = _to_message_format(row["prompt"], row["response_c"])
                self.samples.append(string_c)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class HumanPreferenceBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        self.benchmark_name = "human_preference"
        self.responses_per_question = 2
        self.benchmarks_file = HUMAN_PREFERENCE_DATA_PATH
        super().__init__()


class TruthPreferenceBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        self.benchmark_name = "truth_preference"
        self.responses_per_question = 3
        self.benchmarks_file = TRUTHFUL_QA_DATA_PATH
        super().__init__()


class SafetyPreferenceBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        self.benchmark_name = "safety_preference"
        self.responses_per_question = 2
        self.benchmarks_file = SAFETY_DATA_PATH
        super().__init__()


class VerbosePreferenceBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        self.benchmark_name = "verbose_preference"
        self.responses_per_question = 2
        self.benchmarks_file = VERBOSE_DATA_PATH
        super().__init__()


class RewardBench(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.benchmark_name = "reward_bench"
        dataset_raw = load_dataset("allenai/reward-bench")["filtered"]
        self.responses_per_question = 2

        self.samples = []

        self.df = pd.DataFrame(dataset_raw)

        for index, row in self.df.iterrows():
            string_a = _to_message_format(row["prompt"], row["chosen"])
            string_b = _to_message_format(row["prompt"], row["rejected"])
            self.samples.append(string_a)
            self.samples.append(string_b)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


INTERAL_BENCHMARK_REGISTRY = [
    HumanPreferenceBenchmark,
    TruthPreferenceBenchmark,
    SafetyPreferenceBenchmark,
    VerbosePreferenceBenchmark,
]
