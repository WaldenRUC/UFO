from dataclasses import dataclass
@dataclass
class Config:
    # NER_MODEL: str = "flair/ner-english-ontonotes-fast"
    NER_MODEL: str = "/home/dou/.cache/huggingface/flair/pytorch_model.bin"
    # QG_MODEL: str = "mrm8488/t5-base-finetuned-question-generation-ap"
    QG_MODEL: str = "/home/dou/.cache/huggingface/plm/models--mrm8488--t5-base-finetuned-question-generation-ap/snapshots/c81cbaf0ec96cc3623719e3d8b0f238da5456ca8"
    # QG_MODEL: str = "allenai/t5-small-squad2-question-generation"
    # QA_MODEL: str = "deepset/roberta-large-squad2"
    # QA_MODEL: str = "deepset/roberta-base-squad2"
    QA_MODEL: str = "/home/dou/.cache/huggingface/roberta-base-squad2"
    # D2Q_MODEL: str = "castorini/doc2query-t5-large-msmarco"
    D2Q_MODEL: str = "/home/dou/.cache/huggingface/plm/models--castorini--doc2query-t5-large-msmarco/snapshots/e607227b4d07161391f3a61a7ccd9efcf875ea14"