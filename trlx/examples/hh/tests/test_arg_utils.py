modeified_prompts = Iterable[str]
modeified_prompts = [arg_prompt(prompt, arg_method) for prompt in batch.text]


def arg_prompt(prompt, arg_method):
    if arg_method == "cot":
        return str + "Let's think step by step. "
    elif arg_method == "fewshot":
        # if problem is "key"
        # dataset  = (prompt, response)
        # embed(prompt, respones)
        # embed(new_prompt)
        # find the embed * new_embed = max or max -1
        # global prompt_dict
        # return prompt_dict[key] + str
        related_responses = find_related_responses(prompt)
        return "Given related responses for reference:" + ";".join(related_responses) + prompt


def find_related_responses(prompt):
    # find related responses from dataset
    # return a list of responses

    return related_responses
