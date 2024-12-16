import json, spacy, torch
import numpy as np
from q2_utils import clean_text, f1_score
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
def DatasetReader(fn: str, extractor=None, source_llm=None, return_se=False, return_lk=False):
    with open(fn, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    lines = data[extractor][source_llm]
    llm = [line['llm'] for line in lines]
    question = [line['question'] for line in lines]
    answer = [line['answer'][0] if isinstance(line['answer'], list) else line['answer'] for line in lines]
    human = ['\n'.join(line['human']) for line in lines]
    reference = ['\n'.join(line['reference']) for line in lines]
    if return_lk == False and return_se == False:
        assert len(llm) == len(answer) == len(human) == len(reference)
        return llm, answer, human, reference
    elif return_lk == False and return_se == True: 
        se = [[fact['se'] for fact in line['facts']] for line in lines]
        assert len(llm) == len(answer) == len(human) == len(reference) == len(se)
        return llm, answer, human, reference, se
    elif return_lk == True and return_se == False:
        lk = [line['lk'] for line in lines]
        assert len(llm) == len(answer) == len(human) == len(reference) == len(lk)
        return llm, answer, human, reference, lk
    else:
        se = [[fact['se'] for fact in line['facts']] for line in lines]
        lk = [line['lk'] for line in lines]
        assert len(llm) == len(answer) == len(human) == len(reference) == len(se) == len(lk)
        return llm, answer, human, reference, se, lk
    
    




INVALID_QUESTION = -1
NO_ANS = '[CLS]'
NO_VALID_QUESTIONS = 'NO_Q'
NO_Q = -1
ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5

def filter_questions(exp_ans, pred_ans):
    if pred_ans == NO_ANS:
        return 'NO MATCH'
    if clean_text(exp_ans) != clean_text(pred_ans):
        return 'NO MATCH'
    return 'VALID'


class Q2():
    def __init__(self, qg_model=None, qa_model=None, nli_model=None, gen_method='beam', single_q=True, remove_personal=True, en_core_web_sm_model=None, device="cuda:0"):
        self.device = device
        # QG
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model)
        self.qg_model = AutoModelWithLMHead.from_pretrained(qg_model)
        self.qg_model.to(device)
        # QA
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model)
        self.qa_model.to(device)
        # NLI
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli_model.to(device)
        self.nlp = spacy.load(en_core_web_sm_model)
        self.gen_method = gen_method
        self.single_q = single_q
        self.remove_personal = remove_personal
    
    def get_answer(self, question, text):
        inputs = self.qa_tokenizer.encode_plus(
            question, text, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_ids = inputs["input_ids"].tolist()[0]

        text_tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids)
        
        answer_start_scores, answer_end_scores = self.qa_model(**inputs, return_dict=False)

        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        ans = self.qa_tokenizer.convert_tokens_to_string(self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        return ans

    def non_personal(self, question):
        question_tok = self.nlp(question)
        for tok in question_tok:
            if tok.dep_ == 'nsubj':
                if tok.text.lower() == 'i' or tok.text.lower() == 'you':
                    return False
            elif tok.dep_ == 'poss':
                if tok.text.lower() == 'my' or tok.text.lower() == 'your':
                    return False
        return True
    def single_question_score(self, question, cand, response, knowledge):
        pred_ans = self.get_answer(question, response)

        if filter_questions(cand, pred_ans) == 'VALID':
            knowledge_ans = self.get_answer(question, knowledge)
            if knowledge_ans != NO_ANS:
                return f1_score(cand, knowledge_ans), knowledge_ans
            else:
                return 0, NO_ANS
        else:
            return INVALID_QUESTION, INVALID_QUESTION
    def get_answer_candidates(self, text):
        doc = self.nlp(text)
        candidates = [ent.text for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
        candidates = [cand for cand in candidates if cand.lower() != 'i']
        return candidates

    def get_question_greedy(self, answer, context, max_length=128):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)
        features = {key: value.to(self.device) for key, value in features.items()}
        output = self.qg_model.generate(
            input_ids=features['input_ids'], 
            attention_mask=features['attention_mask'], 
            max_length=max_length)
        question = self.qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
        return question


    def get_questions_beam(self, answer, context, max_length=128, beam_size=5, num_return=5):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)
        features = {key: value.to(self.device) for key, value in features.items()}

        beam_outputs = self.qg_model.generate(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            max_length=max_length, 
            num_beams=beam_size, 
            no_repeat_ngram_size=3,
            num_return_sequences=num_return, 
            early_stopping=True)

        for beam_output in beam_outputs:
            all_questions.append(self.qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions


    def get_questions_sample(self, answer, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.qg_tokenizer([input_text], return_tensors='pt', max_length=512, truncation=True)
        features = {key: value.to(self.device) for key, value in features.items()}

        sampled_outputs = self.qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                            max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                            num_return_sequences=num_return)

        for sampled in sampled_outputs:
            all_questions.append(self.qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions

    def get_nli_label(self, question, cand, evidence_ans):
        premise = question + ' ' + evidence_ans + '.'
        hypothesis = question + ' ' + cand + '.'
        tokenized_input_seq_pair = self.nli_tokenizer.encode_plus(
                                                premise, hypothesis,
                                                max_length=512,
                                                return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
        # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)
        outputs = self.nli_model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()  # batch_size only one
        id2label = {
            "0": "entailment",
            "1": "neutral",
            "2": "contradiction"
        }
        nli_label = id2label[str(np.argmax(predicted_probability))]
        return nli_label

    def score(self, ref, pred) -> float:
        f1 = 0
        num_questions = 0

        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []

        candidates = self.get_answer_candidates(pred)
        for cand in candidates:
            if self.gen_method == 'greedy':
                questions = [self.get_question_greedy(cand, pred)]
            elif self.gen_method == 'beam':
                questions = self.get_questions_beam(cand, pred)
            else:
                questions = self.get_questions_sample(cand, pred)
            for question in questions:
                if not self.remove_personal or self.non_personal(question):
                    question_score, knowledge_ans = self.single_question_score(question, cand, pred, ref)
                    if question_score != INVALID_QUESTION:
                        num_questions += 1
                        f1 += question_score
                        valid_questions.append(question)
                        valid_cands.append(cand)
                        knowledge_answers.append(knowledge_ans)
                        scores.append(question_score)

                        if self.single_q:
                            break
        if num_questions:
            avg_f1 = f1 / num_questions
        else:
            # avg_f1 = INVALID_QUESTION
            avg_f1 = 0
        # return avg_f1, valid_questions, valid_cands, knowledge_answers, scores
        return avg_f1
    
        ### get nli
        nli_scores, f1_scores = [], []
        nli_score = f1
        evidence_answer = knowledge_answers[0]
        # Use NLI to determine answer similarity.
        # This is only applicable for responses that had at least one valid question generated
        if 0 <= f1 < 1 and NO_ANS not in evidence_answer and evidence_answer != '' and evidence_answer != 'nan':
            f1_scores.append(f1)
            # If the score is 1, there is a full overlap between the
            # candidate and the predicted answer, so the score is 1
            # If there is no answer - can't run NLI, keep the original score (0)
            nli_label = self.get_nli_label(valid_questions[0], valid_cands[0], evidence_answer)
            if nli_label == 'entailment':  # If entails, the score is 1
                nli_score = ENTAILMENT_SCORE
            elif nli_label == 'contradiction':  # If contradicts, the score is 0
                nli_score = CONTRADICTION_SCORE
        # Add fallback NLI to responses that are not covered by Q2 (no questions generated)
        elif f1 == NO_Q:
            # nli_fallback = self.get_e2e_nli_score(pred, ref.lower())
            nli_label = self.get_nli_label(valid_questions[0], valid_cands[0], evidence_answer)
            if nli_label == 'entailment':  # If entails, the score is 1
                nli_fallback = ENTAILMENT_SCORE
            elif nli_label == 'contradiction':  # If contradicts, the score is 0
                nli_fallback = CONTRADICTION_SCORE
            else:
                nli_fallback = NEUTRAL_SCORE
            nli_score = nli_fallback
            f1_scores.append(nli_fallback)
        else:
            f1_scores.append(f1)
            nli_score = 0

        return nli_score