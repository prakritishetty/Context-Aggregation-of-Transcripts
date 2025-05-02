import re
import json
from typing import List, Dict, Optional

import openai
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util


class SOAPNoteEvaluator:
    """
    Evaluator for automatically generated SOAP notes, using:
      - QA-based factual consistency (QAFactEval-like)
      - Entailment classifier for hallucination detection
      - Section presence/order and integrity checks for structure
      - Clinical BERT-based BERTScore and embedding similarity for relevance
      - Sentence-BERT semantic similarity checks
      - LLM-as-judge (zero-shot / few-shot)
      - Optionally comparing against free-text summary notes
    """

    def __init__(
        self,
        dialogues: List[str],
        generated_notes: List[str],
        summary_notes: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None,
        device: str = "cpu"
    ):
        # Input data
        self.dialogues = dialogues
        self.generated_notes = generated_notes
        self.summary_notes = summary_notes

        # OpenAI for LLM evaluation
        if openai_api_key:
            openai.api_key = openai_api_key

        # Embedding models
        self.sbert = SentenceTransformer("all-mpnet-base-v2", device=device)
        self.clinbert = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb", device=device)

        # Entailment model
        self.entail_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.entail_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        ).to(device)

        # QA pipelines
        self.qg = pipeline(
            "text2text-generation",
            model="valhalla/t5-small-qg-prepend",
            device=0 if device != "cpu" else -1
        )
        self.qa = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if device != "cpu" else -1
        )

        # Metrics
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")

    # ----- Internal helpers using arbitrary contexts -----
    # def _qa_factuality(self, contexts: List[str]) -> List[Dict[str, float]]:
    #     out = []
    #     for ctx, note in zip(contexts, self.generated_notes):
    #         qs = self.qg(note)
    #         correct = 0
    #         total = len(qs)
    #         for q in qs:
    #             ans_note = self.qa(question=q['question'], context=note)
    #             ans_ctx  = self.qa(question=q['question'], context=ctx)
    #             if ans_note['answer'].strip().lower() == ans_ctx['answer'].strip().lower():
    #                 correct += 1
    #         acc = correct / total if total > 0 else 1.0
    #         out.append({"accuracy": acc, "n_questions": total})
    #     return out

    def _qa_factuality(self, contexts: List[str]) -> List[Dict[str, float]]:
      out = []
      for ctx, note in zip(contexts, self.generated_notes):
          # 1) generate raw questions
          raw_qs    = self.qg(note)
          # 2) extract the question text
          questions = [q["generated_text"] for q in raw_qs]

          # 3) answer + compare
          correct = 0
          for question in questions:
              ans_note = self.qa(question=question, context=note)["answer"]
              ans_ctx  = self.qa(question=question, context=ctx)["answer"]
              if ans_note.strip().lower() == ans_ctx.strip().lower():
                  correct += 1

          total = len(questions)
          acc   = correct / total if total > 0 else 1.0
          out.append({"accuracy": acc, "n_questions": total})
      return out

    def _entailment(self, contexts: List[str]) -> List[Dict[str, float]]:
        out = []
        for ctx, note in zip(contexts, self.generated_notes):
            sents = [s for s in re.split(r'(?<=[.?!])\s+', note) if s.strip()]
            entail_count = 0
            for sent in sents:
                inp = self.entail_tokenizer(ctx, sent, return_tensors="pt", truncation=True, max_length=512).to(self.entail_model.device)
                probs = self.entail_model(**inp).logits.softmax(dim=1)[0]
                if probs[2] > 0.9:
                    entail_count += 1
            score = entail_count / len(sents) if sents else 1.0
            out.append({"entailment_score": score})
        return out

    def _semantic_sim(self, contexts: List[str]) -> List[Dict[str, float]]:
        out = []
        for ctx, note in zip(contexts, self.generated_notes):
            ctx_emb = self.sbert.encode(ctx, convert_to_tensor=True)
            note_emb = self.sbert.encode(note, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(ctx_emb, note_emb).item()
            out.append({"semantic_similarity": sim})
        return out

    def _clinical_relevance(self, contexts: List[str]) -> List[Dict[str, float]]:
        out = []
        for ctx, note in zip(contexts, self.generated_notes):
            bs = self.bertscore.compute(predictions=[note], references=[ctx], model_type="allenai/scibert_scivocab_uncased")
            note_emb = self.clinbert.encode(note, convert_to_tensor=True)
            ctx_emb = self.clinbert.encode(ctx, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(note_emb, ctx_emb).item()
            out.append({"bertscore_f1": bs['f1'][0], "embedding_similarity": sim})
        return out

    # ----- Public metric methods -----
    def qa_factuality_dialogue(self):
        return self._qa_factuality(self.dialogues)

    def qa_factuality_summary(self):
        if not self.summary_notes:
            raise ValueError("No summary_notes provided")
        return self._qa_factuality(self.summary_notes)

    def entailment_dialogue(self):
        return self._entailment(self.dialogues)

    def entailment_summary(self):
        if not self.summary_notes:
            raise ValueError("No summary_notes provided")
        return self._entailment(self.summary_notes)

    def semantic_similarity_dialogue(self):
        return self._semantic_sim(self.dialogues)

    def semantic_similarity_summary(self):
        if not self.summary_notes:
            raise ValueError("No summary_notes provided")
        return self._semantic_sim(self.summary_notes)

    def clinical_relevance_dialogue(self):
        return self._clinical_relevance(self.dialogues)

    def clinical_relevance_summary(self):
        if not self.summary_notes:
            raise ValueError("No summary_notes provided")
        return self._clinical_relevance(self.summary_notes)

    def hallucination_rate_dialogue(self):
        scores = self.entailment_dialogue()
        return [{"hallucination_rate": 1 - s['entailment_score']} for s in scores]

    def hallucination_rate_summary(self):
        scores = self.entailment_summary()
        return [{"hallucination_rate": 1 - s['entailment_score']} for s in scores]

    def evaluate_structure(self) -> List[Dict[str, object]]:
        out = []
        secs = ["Subjective", "Objective", "Assessment", "Plan"]
        for note in self.generated_notes:
            pres, idxs = {}, []
            for s in secs:
                m = re.search(rf"^{s}", note, re.IGNORECASE | re.MULTILINE)
                has = bool(m)
                pres[s] = has
                idxs.append(m.start() if has else float('inf'))
            order_ok = idxs == sorted(idxs)
            subj = re.search(r'(Subjective.*?)(?=^Objective|$)', note, re.IGNORECASE|re.MULTILINE)
            subj_ok = not bool(re.search(r'\d', subj.group(1))) if subj else False
            obj = re.search(r'(Objective.*?)(?=^Assessment|$)', note, re.IGNORECASE|re.MULTILINE)
            obj_ok = not bool(re.search(r'(reports|denies|states)', obj.group(1), re.IGNORECASE)) if obj else False
            out.append({"section_presence": pres, "order_correct": order_ok, "subjective_ok": subj_ok, "objective_ok": obj_ok})
        return out

    def evaluate_llm(
        self,
        model: str = "gpt-4",
        few_shot: Optional[List[Dict[str, str]]] = None,
        criteria: List[str] = ["Relevance","Coherence","Consistency","Fluency","Coverage"]
    ) -> List[Dict[str, float]]:
        results = []
        sys_msg = ("You are a clinical note quality evaluator. Rate each criterion 1-5. Respond only JSON.")
        prefix = [{"role":"system","content":sys_msg}]
        if few_shot:
            for ex in few_shot:
                prefix += [
                    {"role":"user","content":f"Transcript:\n{ex['dialogue']}\n\nSOAP:\n{ex['soap']}"},
                    {"role":"assistant","content":json.dumps(ex['scores'])}
                ]
        for dialogue, note in zip(self.dialogues, self.generated_notes):
            msgs = prefix + [{"role":"user","content":f"Transcript:\n{dialogue}\n\nSOAP:\n{note}"}]
            resp = openai.ChatCompletion.create(model=model,messages=msgs,temperature=0)
            results.append(json.loads(resp.choices[0].message.content))
        return results

    def run_all(self) -> Dict[str, List[Dict]]:
        res = {
            "qa_diag": self.qa_factuality_dialogue(),
            "entail_diag": self.entailment_dialogue(),
            "halluc_diag": self.hallucination_rate_dialogue(),
            "sem_diag": self.semantic_similarity_dialogue(),
            #"clin_diag": self.clinical_relevance_dialogue(),
            "structure": self.evaluate_structure()
        }
        if self.summary_notes:
            res.update({
                "qa_sum": self.qa_factuality_summary(),
                "entail_sum": self.entailment_summary(),
                "halluc_sum": self.hallucination_rate_summary(),
                "sem_sum": self.semantic_similarity_summary()
                #"clin_sum": self.clinical_relevance_summary()
            })
        # if hasattr(openai, 'api_key') and openai.api_key:
        #     res["llm"] = self.evaluate_llm()
        return res
