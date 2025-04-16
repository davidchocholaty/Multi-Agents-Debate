# import re
import time
import json
import pickle
import sys
import os
import re
# import random
import argparse
import numpy as np
import openai
# import numpy as np

from tqdm import tqdm
from data_utils import StrategyQA, GSM8k, Aqua, ECQA, ANLI, DateUnderstanding
from dotenv import load_dotenv

from openai.error import InvalidRequestError

from interactive import Debate

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.environ['OPEN_AI_API_BASE']
openai.api_version = os.environ['OPEN_AI_API_VERSION']
openai.api_key = os.environ['OPEN_AI_API_KEY']


openai_api_key = os.environ['OPEN_AI_API_KEY']

def main(args, dataset, test_samples):
    accuracies = []
    num_error = 0
    invalid_answer = 0

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    for _ in tqdm(range(args.rounds)):
        num_correct = 0

        for test_sample in tqdm(test_samples):
            try:
                config = json.load(open(f"{MAD_path}/code/utils/config4all.json", "r"))
                config['debate_topic'] = test_sample["question"]

                debate = Debate(model_name='gpt-4o', num_players=3, openai_api_key=openai_api_key, config=config, temperature=0, sleep_time=0)
                result = debate.run()
                print(result)

                if dataset == "SQA":
                    if re.search(r'\b(no|No|NO)\.?\b', result):
                        result = "no"
                    elif re.search(r'\b(yes|Yes|YES)\.?\b', result):
                        result = "yes"

                elif dataset == "ECQA":
                    match = re.search(r'\(([A-E])\)', result)

                    if match:
                        letter = match.group(1)
                        result = letter
                    
                elif dataset == "Aqua":
                    match = re.search(r'\(([A-E])\)', result)

                    if match:
                        letter = match.group(1)
                        result = letter 

                elif dataset == "ANLI":
                    match = re.search(r'\((e|c|n|contradiction|neutral|entailment)\)', result, re.IGNORECASE)

                    if match:
                        letter = match.group(1)
                        result = letter
                        
                elif dataset == "DateUnderstanding":
                    match = re.search(r'\(([A-E])\)', result)

                    if match:
                        letter = match.group(1)
                        result = letter

                print("RESULT: ", result)

                if result == test_sample["answer"]:
                    num_correct += 1
                else:
                    if dataset == "SQA" and result not in ['yes', 'no']:
                        invalid_answer += 1
                    elif dataset == "GSM8k" and not result.isnumeric():
                        invalid_answer += 1
                    elif dataset == "ECQA" and result not in [
                        'A',
                        'B',
                        'C',
                        'D',
                        'E',
                    ]:
                        invalid_answer += 1
                    elif dataset == "Aqua" and result not in [
                        'A',
                        'B',
                        'C',
                        'D',
                        'E',
                    ]:
                        invalid_answer += 1
                    elif dataset == "ANLI" and result not in ['e', 'c', 'n']:
                        invalid_answer += 1
                    elif dataset == "DateUnderstanding" and result not in [
                        'A',
                        'B',
                        'C',
                        'D',
                        'E',
                    ]:
                        invalid_answer += 1
            except Exception as e:
                # print(f"Exception during simulation: {e}.", file=sys.stderr)
                # num_error += 1
                raise

            # TODO nekde si asi logovat ty vysledky, ta je to pak dohledatelny

        accuracy = num_correct / len(test_samples)
        print(f"Accuracy: {accuracy}")

        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # TODO ukladat accuracies a mean accuracy s std_accuracy

    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Num error: {num_error}")
    print(f"Num invalid answer: {invalid_answer}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--rounds', default=2, type=int)
    args = parser.parse_args()

    if args.dataset == "SQA":
        data = StrategyQA(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "ECQA":
        data = ECQA(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "GSM8k":
        data = GSM8k(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "Aqua":
        data = Aqua(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "ANLI":
        data = ANLI(data_dir=f'./dataset/{args.dataset}')
    elif args.dataset == "DateUnderstanding":
        data = DateUnderstanding(data_dir=f'./dataset/{args.dataset}')

    test_samples = data.get_test_samples()[:args.num_samples]
    print(f"Number of test samples={len(test_samples)}")

    main(args, args.dataset, test_samples)
