import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union, Tuple

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputeCLSMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def extract_type(self, content_type_answer):
        for t in ["A", "B", "C", "D", "E", "X"]:
            content_type_answer = content_type_answer.replace(f"类型{t}", t)
        match = re.search(r'##\s*answer\s*[:：]\s*([^\n ]+)', content_type_answer, re.IGNORECASE)
        if match:
            answer = match.group(1)
        else:
            answer = "X"

        if answer not in ["A", "B", "C", "D", "E"]:
            answer = "X"
        return answer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        # score_dict = {"accuracy": [], "precision": [], "recall": [], "micro_f1": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(f"="*30)
        print(f"prediction:")
        print(decoded_preds[0])
        print(f"labels:")
        print(decoded_labels[0])
        print(f"=" * 30)

        hypothesis_types = [self.extract_type(pred) for pred in decoded_preds]
        reference_types = [self.extract_type(label) for label in decoded_labels]

        accuracy = accuracy_score(reference_types, hypothesis_types)

        # 计算精确率、召回率和F1得分（micro）
        precision_micro = precision_score(reference_types, hypothesis_types, average='micro')
        recall_micro = recall_score(reference_types, hypothesis_types, average='micro')
        f1_micro = f1_score(reference_types, hypothesis_types, average='micro')

        return {"accuracy": accuracy, "precision": precision_micro, "recall": recall_micro, "micro_f1": f1_micro}
