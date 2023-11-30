import os, json
from typing import List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, T5Tokenizer, T5ForConditionalGeneration
from flair.models import SequenceTagger
from flair.data import Sentence
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        # print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper


def load_qg(model: str, plmPathPrefix: str):
    """
    Load Question Generation model from HuggingFace hub
    Args:
        model (str): model name to be loaded
    Returns:
        function: question generation function
    """
    print("Loading Question Generation Pipeline... %s" % model)

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir = plmPathPrefix)
    model = AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir = plmPathPrefix)

    @timer
    def generate_question(sentences: List[str], total_entities: List):
        """
        Generation question using context and entity information

        Args:
            sentences (List[str]): list of sentences
            total_entities (List): list of entities

        Returns:
            List[Dict] list of question and answer (entity) pairs

        """
        qa_pairs = list()
        template_list, answerList, sentenceList = [], [], []
        for sentence, line_entities in zip(sentences, total_entities):
            for entity in line_entities:
                entity = entity["word"]
                template = f"answer: {entity}  context: {sentence} </s>"
                template_list.append(template)
                answerList.append(entity)
                sentenceList.append(sentence)

        if len(template_list) == 0: return []
        tokens = tokenizer(
            template_list,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )

        outputs = model.generate(**tokens, max_new_tokens=48)
        assert len(outputs) == len(answerList) == len(sentenceList)
        for _id, output in enumerate(outputs):
            question = tokenizer.decode(output)
            question = question.replace("</s>", "")
            question = question.replace("<pad> question: ", "")
            question = question.replace("<pad>", "")
            qa_pairs.append({
                "question": question,
                "phrase": answerList[_id],
                "sentence": sentenceList[_id]
            })

        return qa_pairs
    
    def generate_question_backup(sentences: List[str], total_entities: List):
        """
        Generation question using context and entity information

        Args:
            sentences (List[str]): list of sentences
            total_entities (List): list of entities

        Returns:
            List[Dict] list of question and answer (entity) pairs

        """
        qa_pairs = list()

        for sentence, line_entities in zip(sentences, total_entities):
            for entity in line_entities:
                entity = entity["word"]

                template = f"answer: {entity}  context: {sentence} </s>"

                # TODO: batchify
                tokens = tokenizer(
                    template,
                    # max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )

                outputs = model.generate(**tokens, max_length=64)

                question = tokenizer.decode(outputs[0])
                question = question.replace("</s>", "")
                question = question.replace("<pad> question: ", "")

                qa_pairs.append({
                    "question": question,
                    "phrase": entity,
                    "sentence": sentence
                })

        return qa_pairs

    return generate_question

def load_qa(model: str, plmPathPrefix: str):
    """
    Load Question Answering model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        function: question answering function

    """
    print("Loading Question Answering Pipeline... %s" % model)
    qa = pipeline(
        "question-answering",
        model=model,
        tokenizer=model,
        framework="pt",
        # cache_dir=os.path.join(plmPathPrefix, model)
    )
    @timer
    def answer_question(context: str, qa_pairs: List):
        """
        Answer question via Span Prediction

        Args:
            context (str): context to be encoded
            qa_pairs (List): Question & Answer pairs generated from Question Generation pipe

        """
        answers = list()
        for qa_pair in qa_pairs:
            qa_result = qa(
                question=qa_pair["question"],
                context=context,
                handle_impossible_answer=True,
            )
            pred, score = qa_result["answer"], qa_result["score"]
            answers.append({
                "question": qa_pair["question"],
                "answer": qa_pair["phrase"],
                "prediction": pred if pred != "" else "<unanswerable>",
                "score": score if pred != "" else 0
            })
        return answers

    return answer_question


def load_ner(model: str, plmPathPrefix: str) -> object:
    """
    Load Named Entity Recognition model from HuggingFace hub

    Args:
        model (str): model name to be loaded

    Returns:
        object: Pipeline-based Named Entity Recognition model

    """
    print("Loading Named Entity Recognition Pipeline... %s" % model)

    if "flair" in model:
        ner = SequenceTagger.load(model)
        @timer
        def extract_entities(sentences: List[str]):
            result = list()
            for sentence in sentences:
                sentence = Sentence(sentence)
                ner.predict(sentence)
                cache = dict()
                dedup = list()
                for entity in sentence.get_spans("ner"):
                    if entity.labels[0].shortstring.split('/')[0][1:-1] not in cache:
                        dedup.append({
                            "word": entity.labels[0].shortstring.split('/')[0][1:-1],
                            "entity": entity.labels[0].value,
                            "start": entity.start_position,
                            "end": entity.end_position
                        })
                        cache[entity.labels[0].shortstring.split('/')[0][1:-1]] = None
                result.append(dedup)
            return result
            # for sentence in sentences:
            #     sentence = Sentence(sentence)
            #     ner.predict(sentence)
            #     line_result = sentence.to_dict(tag_type="ner")

            #     cache = dict()
            #     dedup = list()

            #     for entity in line_result["entities"]:
            #         if entity["text"] not in cache:
            #             dedup.append({
            #                 "word": entity["text"],
            #                 "entity": entity["labels"][0].value,
            #                 "start": entity["start_pos"],
            #                 "end": entity["end_pos"],
            #             })
            #             cache[entity["text"]] = None
            #     result.append(dedup)
            # return result
    else:
        ner = pipeline(
            task="ner",
            model=model,
            tokenizer=model,
            ignore_labels=[],
            framework="pt",
            cache_dir=os.path.join(plmPathPrefix, model)
        )

        def extract_entities(sentences: List[str]):
            result = list()
            total_entities = ner(sentences)

            if isinstance(total_entities[0], dict):
                total_entities = [total_entities]

            for line_entities in total_entities:
                result.append(grouped_entities(line_entities))

            return result

    return extract_entities

def load_d2q(model: str, plmPathPrefix: str):
    d2q = pipeline(
        task="text2text-generation", 
        model=model, 
        tokenizer=model)
    def doc2query(modelText: str):
        _response = d2q(modelText)
        return _response[0]["generated_text"]
    return doc2query