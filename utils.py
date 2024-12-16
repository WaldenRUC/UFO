import json, torch, argparse, string, os, json, openai, time, re
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datetime import datetime
class UFO():
    def __init__(self,
                dataset:str = None,
                extractor:str = None,
                source_llm:list = None,
                scenario:list = None,
                evaluator:str = None,
                prompt_prefix:str = None,
                output_path:str = None
    ):
        # ===== load prompt =====
        # replace with <passage>
        with open(os.path.join(prompt_prefix, "fact_unit_extraction.txt"), 'r', encoding='utf-8') as fp:
            self.prompt_fue = fp.read()
        # replace with <evidence> and <question>
        with open(os.path.join(prompt_prefix, "fact_source_verification.txt"), 'r', encoding='utf-8') as fp:
            self.prompt_fsv = fp.read()
        # replace with <answer 1> and <answer 2>
        with open(os.path.join(prompt_prefix, "fact_consistency_discrimination.txt"), 'r', encoding='utf-8') as fp:
            self.prompt_fcd = fp.read()

        # ===== load source_llm =====
        with open(dataset, 'r', encoding='utf-8') as fp:
            self.dataset = json.load(dataset)
        self.samples = self.dataset[f'{extractor}-extractor'][source_llm]        # [{question, answer, ....}]
        
        self.evaluator = evaluator
        
        # ===== load evaluator =====
        self.load_evaluator()


        self.scenario = scenario
        self.scenario_name = '+'.join(scenario)
        

        
        # ===== choose a phase and run =====
        for _id, sample in enumerate(self.samples):
            question = sample.get('question', '')
            answers = sample.get('answer', [])
            answer = '\n'.join(answers)
            model_retrieved_documents = sample.get('model_retrieved_documents', [])
            model_retrieved_document = '\n'.join(model_retrieved_documents)
            facts = sample.get('facts', [])
            llm = sample.get('llm', [])     # to be verified
            human = '\n'.join(sample.get('human', []))
            reference = '\n'.join(sample.get('reference', []))
            lk = sample.get('lk', '')
            for phase in ['fsv', 'fcd']:
                # if phase == 'fue':
                #     '''
                #     generation --> fact_extraction
                #     '''
                #     self.input_dir = os.path.join(self.generation_prefix, self.dataset)
                #     self.output_dir = os.path.join(self.fact_extraction_prefix, self.fact_extraction_model, self.dataset)
                #     os.makedirs(self.output_dir, exist_ok=True)
                #     self.fue()
                if phase == 'fsv':
                    '''
                    fact_extraction --> verification
                    '''
                    for fact in facts:
                        verified = 0
                        for scenario in self.scenario:
                            if scenario == 'hu':
                                query = self.prompt_fsv.format(human, fact['Question'])
                                answer = self.chatgpt_call(sys_prompt='', query=query, model=self.evaluator)
                            elif scenario == 're':
                                query = self.prompt_fsv.format(reference, fact['Question'])
                                answer = self.chatgpt_call(sys_prompt='', query=query, model=self.evaluator)
                            elif scenario == 'se':
                                se = list()
                                for result_dict in fact['se']:
                                    for organic in result_dict['se'].get('organic', []):
                                        se.append(organic.get('snippet', ''))
                                se = '\n'.join([item for item in se if len(item)>0])
                                query = self.prompt_fsv.format(se, fact['Question'])
                                answer = self.chatgpt_call(sys_prompt='', query=query, model=self.evaluator)
                            elif scenario == 'lk':
                                query = self.prompt_fsv.format(lk, fact['Question'])
                                answer = self.chatgpt_call(sys_prompt='', query=query, model=self.evaluator)
                            else: assert False, scenario    
                            if answer.upper() == 'NOANS':
                                continue    # move to the next
                            else:
                                fact['candidate'] = candidate
                                verified = 1
                                break
                        if not verified:
                            fact['candidate'] = 'NOANS'
                if phase == 'fcd':
                    '''
                    verification --> discrimination
                    '''
                    for fact in facts:
                        fact_answer = fact['Answer']
                        candidate = fact['candidate']
                        if candidate.upper() == 'NOANS':
                            fact['ufo_score'] = 0
                        else:
                            query = self.prompt_fcd.format(fact_answer, candidate)
                            response = self.chatgpt_call(sys_prompt='', query=query, model=self.evaluator)
                            if 'yes' in response.lower():
                                fact['ufo_score'] = 1
                            else:
                                fact['ufo_score'] = 0
                else: assert False, phase
                
        with open(output_path, 'w', encoding='utf-8') as fw:
            json.dump(self.samples, ensure_ascii=False, indent=2)



    def get_qa(self, text):
        # match Question and Answer
        # pattern = re.compile(r'Question:\s*(.*?)\nAnswer:\s*(.*?)(?=\nQuestion:|\Z)', re.DOTALL)
        pattern = re.compile(r'Question:\s*(.*?)\nAnswer:\s*(.*?)(?=\n|$)')
        matches = pattern.findall(text)
        # store Question and Answer into the dict
        qa_list = [{'Question': match[0].strip(), 'Answer': match[1].strip()} for match in matches]
        return qa_list

    def load_evaluator(self):
        '''
        set api_base or api_key
        '''
        if self.evaluator == 'llama3':
            openai.api_base = 'https://xxx.xxx.xxx.xxx:8987/v1'
            openai.api_key = 'xxx'
        elif self.evaluator == 'chatgpt':
            openai.api_key = 'xxx'
        else: assert False, self.evaluator

    def fue(self):
        '''
        generation --> fact_extraction
        add keys: facts: List(dict("Question", "Answer")), extraction_tokens
        input_path: /xxx/xxx/ufo/generation/nq/newbing.jsonl
        - question, answer, mgt, token_usage, human_written_evidences, reference_documents, model_retrieved_documents
        output_path: /xxx/xxx/ufo/fact_extraction/llama3/nq/newbing.jsonl
        - question, answer, mgt, token_usage, human_written_evidences, reference_documents, model_retrieved_documents, facts
        '''
        for _id, source_llm in enumerate(self.source_llm):
            input_path = os.path.join(self.input_dir, source_llm)
            output_path = os.path.join(self.output_dir, source_llm)
            read_num = 0
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as fp:
                    read_num = len(fp.readlines())
            with open(input_path, 'r', encoding='utf-8') as fp:
                data = fp.readlines()[read_num:]
            with open(output_path, 'a', encoding='utf-8') as fw:
                for line in tqdm(data, desc=f'{_id+1}/{len(self.source_llm)}: {source_llm}'):
                    line = json.loads(line)
                    sys_prompt = ''
                    # formatting with <passage>
                    query = self.prompt_fue.format(line['mgt'])
                    response = self.inference(sys_prompt=sys_prompt, query=query)
                    results_dict = self.get_qa(response)
                    line.update(facts=results_dict)
                    fw.write(json.dumps(line, ensure_ascii=False) + '\n')



    def fsv(self):
        '''
        fact_extraction --> verification
        add keys: verified_facts: List(dict("Question", "Answer", "Extracted_answer", "source")), verification_tokens
        input_path: /xxx/xxx/ufo/fact_extraction/llama3/nq/newbing.jsonl
        [he/rd: /xxx/xxx/ufo/fact_extraction/llama3/nq/newbing.jsonl, human_written_evidences, reference_documents]
        [lk: /xxx/xxx/ufo/S_lk/llama3/nq/newbing.jsonl, fact_source: List[List[str]]]
        [se: /xxx/xxx/ufo/S_se/llama3/nq/newbing.jsonl, fact_source: str]
        output_path: /xxx/xxx/ufo/verification/llama3/se+lk/nq/newbing.jsonl
        '''
        for _id, source_llm in enumerate(self.source_llm):
            input_path = os.path.join(self.input_dir, source_llm)
            output_path = os.path.join(self.output_dir, source_llm)
            se_path = os.path.join(self.se_dir, source_llm)
            lk_path = os.path.join(self.lk_dir, source_llm)
            # get how many samples need to be verified
            read_num = 0
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as fp:
                    read_num = len(fp.readlines())
            with open(input_path, 'r', encoding='utf-8') as fp:
                data = fp.readlines()[read_num:]
            # get fact source passages; each element in passages['he'/'rd'/'se'/'lk'] contains a passage list
            fact_source_passages = {'hu': list(), 're': list(), 'se': list(), 'lk': list()}
            if 'hu' in self.scenario:
                for line in data:
                    line = json.loads(line)
                    fact_source_passages['hu'].append(line['human_written_evidences'])
            else: 
                fact_source_passages['hu'] = [list() for line in data]

            if 're' in self.scenario:
                for line in data:
                    line = json.loads(line)
                    fact_source_passages['re'].append(line['reference_documents'])
            else:
                fact_source_passages['re'] = [list() for line in data]

            if 'se' in self.scenario:
                with open(se_path, 'r', encoding='utf-8') as fp:
                    se_data = fp.readlines()[read_num:]
                for line in se_data:
                    line = json.loads(line)
                    if isinstance(line['fact_source'], str):
                        line['fact_source'] = [line['fact_source']]
                    fact_source_passages['se'].append(line['fact_source'])
            else:
                fact_source_passages['se'] = [list() for line in data]

            if 'lk' in self.scenario:
                with open(lk_path, 'r', encoding='utf-8') as fp:
                    lk_data = fp.readlines()[read_num:]
                for line in lk_data:
                    line = json.loads(line)
                    if isinstance(line['fact_source'], str):
                        line['fact_source'] = [line['fact_source']]
                    fact_source_passages['lk'].append(line['fact_source'])
            else:
                fact_source_passages['lk'] = [list() for line in data]

            for _id, line in enumerate(tqdm(data), desc='verification', ncols=100):
                line = json.loads(line)
                facts = line['facts']
                # initialize the extracted answer slot of each fact
                for item in facts:
                    item['Extracted_answer'] = None
                # verify by the order of given self.scenario
                for fact_source_name in self.scenario:
                    # select all unscored facts
                    cur_passages = fact_source_passages[fact_source_name][_id]
                    unscored_facts = [item for item in facts if item['Extracted_answer'] == None]
                    for unscored_fact in unscored_facts:
                        _question, _answer = unscored_fact["Question"], unscored_fact["Answer"]
                        for cur_passage in cur_passages:
                            if cur_passage == "": continue
                            #evidence, question
                            _query = self.prompt_fsv.format(cur_passage, _question)
                            response = self.inference(sys_prompt='', query=_query)
                            extracted_answer = self.verify(response)
                            if extracted_answer == None:
                                # move to the next passage
                                continue
                            else:
                                # give the answer, and exit
                                unscored_fact['Extracted_answer'] = extracted_answer
                                break
                                

                
    def verify(self, response):
        '''
        judge whether the noans is in the text
        '''
        lower_response = response.lower()
        if 'noans' in lower_response:
            return None
        return response

    def fcd(self):
        '''
        verification --> discrimination
        add keys: discrimination: List(dict("Question", "Answer", "Extracted_answer", "source", "judge")), discrimination_tokens
        input_path: /xxx/xxx/ufo/verification/llama3/se+lk/nq/newbing.jsonl
        output_path: /xxx/xxx/ufo/discrimination/llama3/se+lk/nq/newbing.jsonl
        '''
        for _id, source_llm in enumerate(self.source_llm):
            input_path = os.path.join(self.input_dir, source_llm)
            output_path = os.path.join(self.output_dir, source_llm)
            read_num = 0
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as fp:
                    read_num = len(fp.readlines())
            with open(input_path, 'r', encoding='utf-8') as fp:
                data = fp.readlines()[read_num:]
            with open(output_path, 'a', encoding='utf-8') as fw:
                for line in tqdm(data, desc=f'{_id+1}/{len(self.source_llm)}: {source_llm}'):
                    line = json.loads(line)
                    facts = line['facts']
                    score_list = list()
                    for fact in facts:
                        answer_from_mgt, answer_from_source = fact['Answer'], fact['Extracted_answer']
                        if answer_from_source == None:
                            # not matching
                            score_list.append(0)
                            continue
                        sys_prompt = ''
                        # formatting with <answer1> <answer2>
                        #TODO what if adding the question to the prompt? introducing potential judging bias?
                        query = self.prompt_fcd.format(answer_from_mgt, answer_from_source)
                        response = self.inference(sys_prompt=sys_prompt, query=query)
                        if 'yes' in response or 'Yes' in response or 'YES' in response:
                            score_list.append(1)
                        else:
                            score_list.append(0)
                    if len(score_list) == 0: avg_score = 1
                    else: avg_score = np.mean(score_list)
                    line.update(score_list=score_list, avg_score=avg_score)
                    fw.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    def inference(self, sys_prompt:str="", query:str="", max_new_tokens:int=2048):
        '''
        let a llm generate a response
        '''
        if self.evaluator == 'llama-3-8b-instruct':
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda:0")
            outputs = self.model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens, 
                eos_token_id=self.terminators,
                temperature=0
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            return response
        elif self.evaluator == 'chatgpt':
            return self.chatgpt_call(sys_prompt, query)
        else: assert False, self.evaluator
    
    def chatgpt_call(self, sys_prompt, query, model='chatgpt', max_tokens=2048):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=max_tokens,
                    temperature=0
                )['choices'][0]['message']['content']
                break
            except Exception as e:
                print(f"error!\n{e}")
                time.sleep(2)
        return response