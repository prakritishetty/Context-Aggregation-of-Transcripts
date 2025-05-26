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
      - Evaluation against dialogue, summary, or combined context
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

        # Tokenizer for chunking inputs to BERTScore
        self.bs_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

    # ----- Internal helpers using arbitrary contexts -----
    def _qa_factuality(self, contexts: List[str]) -> List[Dict[str, float]]:
        """
        QA-based factual consistency check. For each generated SOAP note:
        1. Generate questions from the note via the QG model.
        2. Extract the question text from the model output.
        3. Answer each question using both the SOAP note and the provided context.
        4. Compare answers; if they match, count as correct.
        5. Compute accuracy as correct/total questions.

        Args:
            contexts: List of context strings (e.g., dialogues or summaries) to check against.

        Returns:
            A list of dicts, each containing:
              - "accuracy": fraction of questions with matching answers.
              - "n_questions": number of questions generated for the note.
        """
        out = []
        for ctx, note in zip(contexts, self.generated_notes):
            raw_qs = self.qg(note)
            questions = [q.get("generated_text", "") for q in raw_qs]

            correct = 0
            for question in questions:
                ans_note = self.qa(question=question, context=note)["answer"]
                ans_ctx = self.qa(question=question, context=ctx)["answer"]
                if ans_note.strip().lower() == ans_ctx.strip().lower():
                    correct += 1

            total = len(questions)
            acc = correct / total if total > 0 else 1.0
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

    def _chunk_text(self, text: str, max_len: int = 512, stride: int = 256) -> List[str]:
        # Ask for overflowing tokens but _without_ return_tensors
        enc = self.bs_tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            return_special_tokens_mask=False
        )
        # Now enc["input_ids"] is a List[List[int]] with each sublist exactly max_len long
        chunks = []
        for ids in enc["input_ids"]:
            chunks.append(self.bs_tokenizer.decode(ids, skip_special_tokens=True))
        return chunks


    def _clinical_relevance(self, contexts: List[str]) -> List[Dict[str, float]]:
        out = []
        for ctx, note in zip(contexts, self.generated_notes):
            # 1) chunk both note and context
            note_chunks = self._chunk_text(note)
            ctx_chunks  = self._chunk_text(ctx)

            # 2) compute BERTScore on each pair and average
            f1_scores = []
            for n_chunk, c_chunk in zip(note_chunks, ctx_chunks):
                bs = self.bertscore.compute(
                    predictions=[n_chunk],
                    references=[c_chunk],
                    model_type="allenai/scibert_scivocab_uncased"
                )
                f1_scores.append(bs["f1"][0])
            avg_f1 = sum(f1_scores) / len(f1_scores)

            # 3) embedding similarity still on full text
            note_emb = self.clinbert.encode(note, convert_to_tensor=True)
            ctx_emb  = self.clinbert.encode(ctx,  convert_to_tensor=True)
            sim = util.pytorch_cos_sim(note_emb, ctx_emb).item()

            out.append({
                "bertscore_f1": avg_f1,
                "embedding_similarity": sim
            })
        return out

    # ----- Public metric methods -----
 
    # Combined context (dialogue + summary)
    def _combined_context(self) -> List[str]:
        if not self.summary_notes:
            raise ValueError("No summary_notes provided")
        return [d + "\n\n" + s for d, s in zip(self.dialogues, self.summary_notes)]
    def qa_factuality_combined(self): return self._qa_factuality(self._combined_context())
    def entailment_combined(self): return self._entailment(self._combined_context())
    def semantic_similarity_combined(self): return self._semantic_sim(self._combined_context())
    def clinical_relevance_combined(self): return self._clinical_relevance(self._combined_context())
    def hallucination_rate_combined(self):
        return [{"hallucination_rate": 1 - c['entailment_score']} for c in self.entailment_combined()]

    def evaluate_structure(self) -> List[Dict[str, object]]:
        """Evaluates SOAP note structure to check for order and presence of 4 sections."""
        out = []
        secs = ["S", "O", "A", "P"]
        for note in self.generated_notes:
            pres, idxs = {}, []
            for s in secs:
                m = re.search(rf"^{s}", note, re.IGNORECASE | re.MULTILINE)
                has = bool(m)
                pres[s] = has
                idxs.append(m.start() if has else float('inf'))
            order_ok = idxs == sorted(idxs)
            out.append({"section_presence": pres, "order_correct": order_ok})
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
            response = openai.ChatCompletion.create(
                model=model,
                messages=msgs,
                temperature=0
            )
            results.append(json.loads(response['choices'][0]['message']['content']))
        return results

    def run_all(self) -> Dict[str, List[Dict]]:
        res = {
            "structure": self.evaluate_structure(),
            "qa_comb": self.qa_factuality_combined(),
            "entail_comb": self.entailment_combined(),
            "halluc_comb": self.hallucination_rate_combined(),
            "sem_comb": self.semantic_similarity_combined(),
            "clin_comb": self.clinical_relevance_combined()
        }
        if hasattr(openai, 'api_key') and openai.api_key:
            res["llm"] = self.evaluate_llm()
        return res
