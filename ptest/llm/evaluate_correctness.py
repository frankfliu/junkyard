import json
import sys
import os
import argparse


def evaluate_correctness(dataset_file, log_file):
    """
    Evaluates the correctness of generated text against a dataset.

    run lmbench to collect logs:

    lmbench "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=$GEMINI_API_KEY" \
        -H "Content-Type: application/json" \
        --extra-parameters '{"systemInstruction": {"parts": [{"text": "Return only the multiple choice option answer. Do not include any explanations, extra text or punctuation."}]}}' \
        --dataset ~/mmlu.jsonl \
        -o output -t -j gemini

    lmbench "https://aiplatform.googleapis.com/v1/projects/$PROJECT/locations/global/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict" \
      -H "Authorization: Bearer $(gcloud auth print-access-token)" \
      -H "Content-Type: application/json" \
      --extra-parameters '{"system": "You are a multiple-choice question assistant.<instructions>1.  For any question I provide, you will only respond with a single letter (A, B, C, or D) that corresponds to the correct answer.2.  You will not provide any explanations, additional words, or punctuation.3.  Your response must be *only* the single capital letter.</instructions><example_interaction>User: What is the capital of France? A) Berlin, B) Madrid, C) Paris, D) Rome\nAssistant: C</example_interaction>"}' \
      -d '{
        "inputs": "The following are multiple choice questions (with answers) about  abstract algebra.\n\nLet p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.\nA. 8\nB. 2\nC. 24\nD. 120\n"
    }' -j anthropic -o output -t

    lmbench "https://aiplatform.googleapis.com/v1/projects/$PROJECT/locations/global/endpoints/openapi/chat/completions" \
      -H "Authorization: Bearer $(gcloud auth print-access-token)" \
      -H "Content-Type: application/json" \
      --extra-parameters '{"model": "qwen/qwen3-next-80b-a3b-instruct-maas", "messages": [{"role": "system", "content": "You are a test-answering assistant. Your only task is to answer the multiple-choice question provided in the user prompt. You must respond with a single letter corresponding to the correct option (e.g., A, B, C, or D) and nothing else."}]}' \
      -d '{
        "inputs": "The following are multiple choice questions (with answers) about  abstract algebra.\n\nLet p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.\nA. 8\nB. 2\nC. 24\nD. 120\n"
    }' -j openai -o output -t

    Args:
        dataset_file (str): Path to the .jsonl dataset file.
        log_file (str): Path to the output.log file.
    """
    answers = {}
    with open(dataset_file, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            answers[str(i)] = data.get("answer")

    correct_count = 0
    total_count = 0
    with open(log_file, "r") as f:
        for line in f:
            try:
                log_data = json.loads(line)["fields"]
                task_id = log_data["task_id"]
                generated_text = log_data["generated_text"]
                total_count += 1
                answer = answers[task_id]
                if answer and generated_text.strip().startswith(answer.strip()):
                    correct_count += 1
                else:
                    print(f"id: {task_id}: Expected: {answer}, Actual: {generated_text}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {log_file}: {line.strip()}")

    if total_count > 0:
        percentage = (correct_count / total_count) * 100
        print(f"Correct: {correct_count} out of {total_count}")
        print(f"Accuracy: {percentage:.2f}%")
    else:
        print("No predictions found in the log file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates the correctness of generated text against a dataset.")
    parser.add_argument("dataset_file", help="Path to the .jsonl dataset file.")
    parser.add_argument("--log_file", default=os.path.join("output", "output.log"), help="Path to the output.log file.")
    args = parser.parse_args()

    dataset_path = args.dataset_file
    log_path = args.log_file

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)

    evaluate_correctness(dataset_path, log_path)
