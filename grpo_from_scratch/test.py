from openai import OpenAI

api_key = ''
base_url = ''

client = OpenAI(api_key=api_key, base_url=base_url)

SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""

completion = client.chat.completions.create(
model = 'qwen1.5b',

temperature=0.0,
logprobs = True,
messages=[
    {
        "role": "system",
        "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "天上五只鸟，地上五只鸡，一共几只鸭",
    }
],
)
print(completion.choices[0].message.content)