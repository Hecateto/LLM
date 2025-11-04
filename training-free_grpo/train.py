import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tqdm import tqdm
import copy

from rag_demo.rag import prompt


class TrainingFreeGRPO:
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model: str,
                 max_workers: int = 5,
                 temperature: float = 1.0):

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_workers = max_workers
        self.temperature = temperature

    def get_llm_output(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt
            }], temperature=self.temperature
        )
        return res.choices[0].message.content

    def get_llm_output_batch(self, prompts: List[str]) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.get_llm_output, prompts))

    def load_experiences(self, experiences_file: str = 'experiences.json') -> str:
        if os.path.exists(experiences_file):
            with open(experiences_file, 'r', encoding='utf-8') as f:
                experiences = json.load(f)
                return experiences
        return {}

    def train(self,
              train_data_file: str = 'train_data.json',
              experiences_file: str = 'experiences.json',   # 当前经验知识库K
              batch_size: int = 2,
              epochs: int = 3,
              n: int = 4) -> None:
        from prompts import EXPERIENCE_GENERATE_PROMPT, PROBLEM_WITH_EXPERIENCE_PROMPT, BATCH_EXPERIENCE_UPDATE_PROMPT
        train_data = json.load(open(train_data_file, 'r', encoding='utf-8'))

        steps = len(train_data) * epochs // batch_size + 1
        pbar = tqdm(total=steps, desc='Training Progress')
        for epoch in range(epochs):
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                experiences = self.load_experiences(experiences_file)
                prompts = []
                for item in batch:
                    item_prompts = [PROBLEM_WITH_EXPERIENCE_PROMPT.format(
                        problem=item['prompt'],
                        experiences=experiences
                    )] * n
                    prompts.extend(item_prompts)
                llm_outputs = self.get_llm_output_batch(prompts)    # 构造n个prompt和输出

                # 基于当前K的副本, 接收add/modify
                candidate_experiences = copy.deepcopy(experiences)
                to_modify = []
                for idx, item in enumerate(batch):
                    start = idx * n
                    end = start + n
                    trajectory_texts = llm_outputs[start:end]
                    experience_prompt = EXPERIENCE_GENERATE_PROMPT.format(
                        problem=item['prompt'],                         # 问题描述
                        trajectories='\n\n'.join(trajectory_texts),     # 推理过程
                        answer=item['answer'],                          # 答案
                        experiences=experiences,                        # 经验
                        max_operations=1                                # 最大更新经验条数
                    )

                    single_sample_experience = self.get_llm_output(experience_prompt)
                    single_sample_experience = single_sample_experience.split('```json')[-1].split('```')[0].strip()
                    try:
                        single_sample_experience = json.loads(single_sample_experience)
                        for exp in single_sample_experience:
                            if exp['option'] == 'add':
                                candidate_experiences[str(len(candidate_experiences)+1)] = exp['experience']
                            elif exp['option'] == 'modify':
                                if str(exp['modified_from']) in candidate_experiences:
                                    to_modify.append(exp)
                    except Exception as e:
                        print(f'Error parsing experience: {e}')
                        continue

                update_experiences = self.get_llm_output(BATCH_EXPERIENCE_UPDATE_PROMPT.format(
                    experiences=experiences,
                    updates=to_modify
                ))
                update_experiences = update_experiences.split('```json')[-1].split('```')[0].strip()
                new_experiences = copy.deepcopy(candidate_experiences)
                pbar.update(1)
                try:
                    update_experiences = json.loads(update_experiences)
                    max_id = max([int(k) for k in new_experiences.keys()], default=0)
                    for exp in update_experiences:
                        if exp['option'] == 'modify':
                            if str(exp['modified_from']) in new_experiences:
                                new_experiences[str(exp['modified_from'])] = exp['experience']
                            elif exp['option'] == 'merge':
                                to_merge = exp.get('merged_from', [])
                                for exp_id in to_merge:
                                    if str(exp_id) in new_experiences:
                                        del new_experiences[str(exp_id)]
                                new_experiences[str(max_id+1)] = exp['experience']
                                max_id += 1

                    new_experiences = {str(idx+1): exp for idx, exp in enumerate(new_experiences.values())}

                    with open(experiences_file, 'w', encoding='utf-8') as f:
                        json.dump(new_experiences, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f'Error parsing final experience updates: {e}')
                    continue

def main():
    trainer = TrainingFreeGRPO(
        max_workers=5, temperature=1.0
    )
    trainer.train(
        train_data_file='train_data.json',
        experiences_file='experiences.json',
        batch_size=4,
        epochs=3,
        n=4
    )

if __name__ == '__main__':
    main()