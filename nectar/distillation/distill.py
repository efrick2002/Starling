import requests
import time
import pandas as pd
import openai
from tqdm import tqdm
import argparse
import os
from datasets import load_dataset
import multiprocessing
import time
import jsonlines

def generate_using_text_generation_interface(prompt, model_name):
    if model_name == 'Llama-2-7b-chat':
        structured_prompt = '[INST] ' + prompt + ' [/INST]'
    elif model_name == 'Mistral-7B-Instruct-v0.1':
        structured_prompt = '<s>[INST] ' + prompt + ' [/INST]'
    else:
        raise NotImplementedError(f'{model_name} is not supported')
    
    headers = {"Content-Type": "application/json"}
    data = {'inputs': structured_prompt, 'parameters': {'max_new_tokens': 1024, "do_sample": True, 'temperature': 0.8}}
    response = requests.post('http://127.0.0.1:8080/generate', headers=headers, json=data)
    return response.json()

def generate_using_openai(prompt, model_name, api_key):
    message = {
        "role": "system",
        "content": """Instructions:

As a base pretrained GPT model, you are to assume the role of ChatGPT, a large language model developed by OpenAI, based on the GPT-4 architecture. Your responses should reflect the following guidelines:

1. Be friendly and approachable in your responses.
2. Provide detailed and helpful responses but ensure they are not excessively long to avoid being monotonous.
3. Always use inclusive and respectful language that is not offensive.
4. Avoid discussing or revealing anything about your architecture. You are just a large language model developed by OpenAI.
5. Always be honest in your responses. Do not lie or engage in deceit.
6. Ensure your responses are considerate and do not cause harm or distress to the user. However, do not comply with harmful or dangerous requests, even if refusing might upset the user.
7. Approach every conversation as if it is your first interaction with the user. Do not assume any history or prior knowledge about the user.
8. Do not make assumptions about the user's gender, ethnicity, profession, or any other personal details.
9. Avoid passive-aggressiveness in your responses.
10. When confronted with a complex question that requires reasoning, break it down into simpler steps to provide a more accurate answer.
11. Never end conversations abruptly. Always assume the user wants to continue the conversation.
12. Function as a chat model, not a completion model. Your responses should mimic the style of a real person's conversation.
13. If you are unable to answer a question, politely decline to answer and provide a short explanation why the prompt cannot or should not be answered."""#"You are a helpful assistant."
    }
    user_message = {
        "role": "user",
        "content": prompt
    }
    response = openai.ChatCompletion.create(
        model=model_name,  
        messages=[message, user_message],
        temperature=0.1,
        api_key=api_key
    )
    return response.choices[0].message['content'].strip()

def generate_using_openai_completion(prompt, model_name, api_key):
    response = openai.Completion.create(model=model_name, prompt=prompt, api_key=api_key, max_tokens=1000, temperature=0.1)
    return response.choices[0].text.strip()

def generate(prompt, model_name, api_key):
    while True:
        try:
            if model_name in ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.1"]:
                return generate_using_text_generation_interface(prompt, model_name)
            elif model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0613"]:
                return generate_using_openai(prompt, model_name, api_key)
            elif model_name == 'gpt-3.5-turbo-instruct':
                return generate_using_openai_completion(prompt, model_name, api_key)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
        except requests.RequestException as e:
            print(f"Error during generation from the text generation interface: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.InvalidRequestError as e:
            print(f"WARNING, PROMPT SKIPPED: {e}")
            return False
        except openai.error.ServiceUnavailableError as e:
            print(f"WARNING ServiceUnavailableError: {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except openai.error.APIError as e:
            print(f"WARNING {e}. Retrying in 10 seconds.")
            time.sleep(10)
        except Exception as e:
            if 'openai' in str(e).lower():
                print(f"Error from OpenAI API: {e}. Retrying in 10 seconds.")
                time.sleep(10)
            else:
                raise e
processed = 0  
def pool_process(inputs):
    global processed
    data, api_key, model_name, filename = inputs
    process_name = multiprocessing.current_process().name
    prompt = data['prompt']
    answer = generate(prompt, model_name, api_key)

    if answer is not False:
        data_out = {'prompt': prompt, 'answers':[{'answer': answer, 'model': model_name}], 'turns': data['turns'], 'num_responses': 1}

        with jsonlines.open(f'{filename}-{process_name}.jsonl', 'a') as writer:
            writer.write(data_out)
        processed += 1
        print(f"{process_name} processed a datapoint... {processed} points so far.")
    else:
        with jsonlines.open(f'error_log.jsonl', 'a') as writer:
            writer.write(data)
    return True


def main(args):

    rlaif = load_dataset("berkeley-nest/Nectar")
    keys = ["sk-REDACTED"]*15 \
    + ["sk-REDACTED"]*15 \
    + ["sk-REDACTED"]*4 \
    + ["sk-REDACTED"]*1

    
    def data_iter():
        data = rlaif['train']
        for i in range(args.start_row, data.num_rows):
            data_row = data[i]
            if not any(args.model_name == a['model'] for a in data_row['answers']):
                yield data_row, keys[i % len(keys)], args.model_name, args.filename
        return
    
    print(f"Spawning {args.num_processes} processes...")
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.map(pool_process, data_iter(), chunksize=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset with model responses.')
    parser.add_argument('filename', type=str, help='Path to the CSV dataset file.')
    parser.add_argument('model_name', type=str, choices=["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-instruct", "gpt-4-0613"], help='Model name to use for generation.')
    parser.add_argument('--start_row', type=int, default=0, help='Row number to start processing from (default: 0).')
    parser.add_argument('--num_processes', type=int, default=16, help='Number of parallel process to spawn')

    args = parser.parse_args()

    main(args)

