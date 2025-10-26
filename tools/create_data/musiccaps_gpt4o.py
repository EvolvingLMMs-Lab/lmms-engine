import argparse
import os
import re
import time
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import jsonlines
import requests
from tqdm import tqdm

prompt_path = Path(__file__).parent / "prompt" / "conversation.txt"
PROMPT = open(str(prompt_path), "r", encoding="utf-8").read()
retries = 3
NUM_SECONDS_TO_SLEEP = 5

API_TYPE = os.getenv("API_TYPE", "azure")

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-file", "-j", type=str, help="path to your jsonl file")
    parser.add_argument("--output-file", "-o", type=str, help="path to your output file")
    return parser.parse_args()


def parse_output(response, audio_path):
    pattern = r"\*\*(User|AI Assistant):\*\* (.*?)\n\n|\*\*(User|AI Assistant):\*\* (.*)"
    matches = re.findall(pattern, response, re.DOTALL)

    messages = []
    first_one = True
    for match in matches:
        speaker = match[0] or match[2]  # Match the User or AI Assistant
        if speaker == "User":
            speaker = "user"
        elif speaker == "AI Assistant":
            speaker = "assistant"
        answer = match[1] or match[3]  # Capture their response
        if first_one:
            messages.append(
                {
                    "role": speaker,
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_path}},
                        {"type": "text", "text": answer},
                    ],
                }
            )
            first_one = False
        else:
            messages.append({"role": speaker, "content": [{"type": "text", "text": answer}]})

    return messages


def get_response(max_tokens: int, content: str, retries: int = retries):
    global headers

    messages = [
        {"role": "user", "content": content},
    ]

    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries:  # If we have retries left, sleep and then continue to next attempt
                time.sleep(NUM_SECONDS_TO_SLEEP)
            else:  # If this was the last attempt, log and return empty
                print(f"All {retries} attempts failed. Last error message: {e}")
                return "", ""
    return "", ""


def process_single_data(data):
    messages = data["messages"]
    for message in messages:
        if message["role"] == "assistant":
            content = message["content"][0]["text"]
            generate_prompt = PROMPT + f"\n\n ## Caption {content}" + "\n\n ## Output"
            response, model = get_response(max_tokens=4096, content=generate_prompt)
        elif message["role"] == "user":
            audio_path = message["content"][0]["audio_url"]["url"]

    new_data = {"id": data["id"], "messages": parse_output(response, audio_path)}
    return new_data


def main():
    args = parse_argument()
    with jsonlines.open(args.jsonl_file, "r") as reader:
        data = list(reader)

    with ThreadPool(16) as pool:
        data = list(tqdm(pool.imap(process_single_data, data), total=len(data)))

    with jsonlines.open(args.output_file, "w") as writer:
        writer.write_all(data)


if __name__ == "__main__":
    main()
