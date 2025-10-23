import re

def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Rewards
def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
    if text.count("</think>\n") == 1:
        reward += 0.125
    if text.count("<answer>\n") == 1:
        reward += 0.125
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

def reward_mark(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    return [mark_num(res) for res in responses]

def reward_correctness(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0][-1]['content']}",
          f"\n答案:\n{answer[0]}",
          f"\n模型输出:\n{responses[0]}",
          f"\n提取后的答案:\n{extracted[0]}")
    return [2.0 if res == str(ans) else 0.0 for res, ans in zip(extracted, answer)]

# 对'生成答案是数字'的稀疏性补充
def reward_digit(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    return [0.5 if res.isdigit() else 0.0 for res in extracted]

def reward_hard_format(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, res) for res in responses]
    return [0.5 if match else 0.0 for match in matches]

def reward_soft_format(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]['content'] for completion in completions]
    matches = [re.match(pattern, res) for res in responses]
    return [0.5 if match else 0.0 for match in matches]