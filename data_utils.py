# Taken from: https://github.com/dinobby/ReConcile/tree/main

# credit to https://github.com/swarnaHub/ExplanationIntervention/blob/main/src/data_utils.py
import json
import os
import random
import re
import string

import pandas

# random.seed(1234)
random.seed(0)


class StrategyQA:
    def __init__(self, data_dir):
        self.dev_path = os.path.join(data_dir, "dev.json")

    def get_samples(self, file_path):
        samples = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            json_inputs = json.load(f)
            for i, json_input in enumerate(json_inputs):
                samples.append(
                    {
                        "index": i,
                        "qid": json_input["qid"],
                        "question": json_input["question"]
                        + " The response should end with yes or no (lowercase).",
                        "answer": "yes" if json_input["answer"] else "no",
                        "gold_explanation": " ".join(json_input["facts"]),
                    }
                )

        return samples

    def get_test_samples(self):
        # return self.get_samples(self.dev_path)
        samples = self.get_samples(self.dev_path)
        random.shuffle(samples)
        return samples


class GSM8k:
    def __init__(self, data_dir):
        self.test_path = os.path.join(data_dir, "test.jsonl")

    def get_samples(self, file_path):
        samples = []

        with open(file_path, "r") as f:
            jsonlines = f.read().splitlines()
            for i, jsonline in enumerate(jsonlines):
                sample = json.loads(jsonline)
                answer = re.sub(
                    r"[^0-9.]", "", sample["answer"].split("#### ")[1].strip()
                )
                gold_explanation = re.sub(
                    '<<.*>>',
                    '',
                    sample["answer"].split("#### ")[0].replace("\n\n", "\n").strip(),
                )
                gold_explanation_sents = gold_explanation.split("\n")
                gold_explanation_sents = [
                    (
                        gold_explanation_sent + "."
                        if gold_explanation_sent[-1] != "."
                        else gold_explanation_sent
                    )
                    for gold_explanation_sent in gold_explanation_sents
                ]
                gold_explanation = " ".join(gold_explanation_sents)
                sample_json = {
                    "index": i,
                    "question": sample["question"]
                    + " The response should be a single numeric value.",
                    "answer": answer,
                    "gold_explanation": gold_explanation,
                }
                samples.append(sample_json)

        return samples

    def get_test_samples(self):
        # return self.get_samples(self.test_path)
        samples = self.get_samples(self.test_path)
        random.shuffle(samples)
        return samples


class Aqua:
    def __init__(self, data_dir):
        self.test_path = os.path.join(data_dir, "test.json")

    def get_samples(self, file_path):
        samples = []
        data = [json.loads(line) for line in open(file_path, 'r')]
        for i, json_input in enumerate(data):
            samples.append(
                {
                    "index": i,
                    "question": json_input["question"]
                    + " Choose one of the following options: "
                    + str(json_input["options"]) + ". The response should end with a single uppercase letter.",
                    "options": json_input["options"],
                    "answer": json_input["correct"],
                    "gold_explanation": json_input["rationale"],
                }
            )

        return samples

    def get_test_samples(self):
        # return self.get_samples(self.test_path)
        samples = self.get_samples(self.test_path)
        random.shuffle(samples)
        return samples


class ECQA:
    def __init__(self, data_dir):
        self.test_path = os.path.join(data_dir, "cqa_data_test.csv")

    def get_samples(self, file_path):
        samples = []
        df = pandas.read_csv(file_path)
        for index, row in df.iterrows():
            options = [
                row["q_op1"],
                row["q_op2"],
                row["q_op3"],
                row["q_op4"],
                row["q_op5"],
            ]

            formatted_options = [
                f"{letter}){opt}"
                for letter, opt in zip(string.ascii_uppercase, options)
            ]

            samples.append(
                {
                    "index": index,
                    "question": row["q_text"]
                    + " Choose one of the following options: "
                    + str(formatted_options) + ". The response should end with a single uppercase letter.",
                    "options": formatted_options,
                    "answer": string.ascii_uppercase[options.index(row["q_ans"])],
                    "gold_explanation": row["taskB"],
                }
            )

        return samples

    def get_test_samples(self):
        # return self.get_samples(self.test_path)
        samples = self.get_samples(self.test_path)
        random.shuffle(samples)
        return samples


class ANLI:
    def __init__(self, data_dir):
        # Concatenated test.jsonl files from R1, R2, and R3 (all rounds).
        self.test_path = os.path.join(data_dir, "test.jsonl")

    def get_samples(self, file_path):
        samples = []

        with open(file_path, "r") as f:
            jsonlines = f.read().splitlines()
            for i, jsonline in enumerate(jsonlines):
                sample = json.loads(jsonline)

                sample_json = {
                    "index": i,
                    "uid": sample["uid"],
                    "question": f'Premise: {sample["context"]}\nHypothesis: {sample["hypothesis"]}'
                    + " Choose from e (entailment), c (contradiction), n (neutral).",
                    "answer": sample[
                        "label"
                    ],  # e - entailment, c - contradiction, n - neutral
                }
                samples.append(sample_json)

        return samples

    def get_test_samples(self):
        # return self.get_samples(self.test_path)
        samples = self.get_samples(self.test_path)
        random.shuffle(samples)
        return samples


class DateUnderstanding:
    def __init__(self, data_dir):
        # Concatenated test.jsonl files from R1, R2, and R3 (all rounds).
        self.test_path = os.path.join(data_dir, "task.json")

    def get_samples(self, file_path):
        samples = []
        with open(file_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            json_inputs = data["examples"]

            for i, json_input in enumerate(json_inputs):
                # Create a list of possible options in the format ["A)05/01/2021", "B)02/23/2021", ...]
                options = [
                    f"{letter}){date}"
                    for letter, date in zip(
                        string.ascii_uppercase, json_input["target_scores"].keys()
                    )
                ]

                # Find the correct answer
                correct_date = next(
                    date  # Take only "A" from "A)05/01/2021"
                    for date, score in json_input["target_scores"].items()
                    if score == 1
                )
                correct_letter = string.ascii_uppercase[
                    list(json_input["target_scores"].keys()).index(correct_date)
                ]

                # print("Options:", options)
                # print("Correct Answer:", correct_answer)

                samples.append(
                    {
                        "index": i,
                        "question": json_input["input"]
                        + " Choose one of the following options: "
                        + str(options) + ". Respond should end with a single uppercase letter.",
                        "options": options,
                        "answer": correct_letter,
                    }
                )

        return samples

    def get_test_samples(self):
        # return self.get_samples(self.test_path)
        samples = self.get_samples(self.test_path)
        random.shuffle(samples)
        return samples