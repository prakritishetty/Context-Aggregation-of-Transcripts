# !pip install evaluate datasets transformers sentence-transformers openai
# !pip install openai==0.28
# !pip install bert_score

import re
import json
from typing import List, Dict, Optional
import openai
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch


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

        
        # device for torch models
        self.device = torch.device(device if device != "cpu" else "cpu")

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

        # OpenAI for LLM evaluation
        if openai_api_key:
            openai.api_key = openai_api_key

        # Embedding models
        self.clinbert = SentenceTransformer("pritamdeka/S-BioBert-snli-multinli-stsb", device=device)

        # Metrics
        self.bertscore = evaluate.load("bertscore")

        # Tokenizer for chunking inputs to BERTScore
        self.bs_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

        # Perplexity (GPT-2)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        # make sure pad token exists
        if self.ppl_tokenizer.pad_token is None:
            self.ppl_tokenizer.pad_token = self.ppl_tokenizer.eos_token


    #----- Run All ---------------------------------------
    def run_all(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Run all evaluation metrics and return a dict mapping metric names
        to their per‐example results.
        """
        results: Dict[str, List[Dict[str, float]]] = {
            # 1) Structure checks
            "structure": self.evaluate_structure(),
            # 2) QA‐based factuality over dialogue+summary
             "qa_comb":   self.qa_factuality_combined(),
            # 3) Clinical relevance (BERTScore + embed sim) over dialogue+summary
            "clin_comb": self.clinical_relevance_combined(),
            # 4) Fluency via perplexity
            "perplexity": self._perplexity(),
        }

        # 5) LLM-as-judge, if an OpenAI key is configured
        if getattr(openai, "api_key", None):
            results["llm"] = self.evaluate_llm()

        return results


    #----- Main Functions --------------------------------
    def qa_factuality_combined(self): return self._qa_factuality(self._combined_context())
    def clinical_relevance_combined(self): return self._clinical_relevance(self._combined_context())

    def _perplexity(self) -> List[Dict[str, float]]:
        """
        Compute perplexity of each generated note under GPT-2.
        Returns a list of {"perplexity": float}.
        """
        out = []
        self.ppl_model.eval()
        for note in self.generated_notes:
            enc = self.ppl_tokenizer(
                note,
                return_tensors="pt",
                truncation=True,
                padding="longest",
            ).to(self.device)

            with torch.no_grad():
                # labels=input_ids to compute loss over entire sequence
                output = self.ppl_model(**enc, labels=enc["input_ids"])
                # output.loss is average negative log likelihood
                ppl = torch.exp(output.loss).item()
            out.append({"perplexity": ppl})
        return out

    def evaluate_llm(
    self,
    model: str = "gpt-4",
    few_shot: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, float]]:
        """
        Uses an LLM to evaluate each generated SOAP note on section‐level content,
        coherence, and fluency, following the detailed prompt.
        Returns a list of dicts with keys:
          Subjective, Objective, Assessment, Plan, Coherence, Fluency
        """
        # 1) Build the fixed system prompt
        sys_msg = (
            "You are a clinical note quality evaluator. For each SOAP note provided, do the following:\n\n"
            "1. Section-Level Content Checks\n"
            "For each SOAP header—Subjective, Objective, Assessment, Plan—score **only** on:\n"
            "- **Presence of at least one valid element** for that section (see lists below).\n"
            "- **Appropriateness of every detail** (nothing misplaced).\n\n"
            "Do **not** deduct points for missing optional items. Only deduct if a section either:\n"
            "1. Contains information that belongs in a different header, or\n"
            "2. Has no valid content at all.\n\n"
            "Use these *valid* elements—at least one must appear for a “1” → “5” scale:\n\n"
            "**Subjective** (pick any)\n"
            "• Chief complaint in patient’s words\n"
            "• Any one OPQRST element (Onset, Quality, etc.)\n"
            "• Past medical/family/social history\n"
            "• Current medication or allergies\n\n"
            "**Objective** (pick any)\n"
            "• Any vital sign (e.g. BP, HR, RR, Temp, O₂ sat)\n"
            "• Any physical exam finding\n"
            "• Any diagnostic result (lab, imaging, EKG)\n\n"
            "**Assessment** (pick any)\n"
            "• A statement of primary diagnosis or impression\n"
            "• Differential diagnoses (≥1)\n"
            "• Numbered problem heading\n\n"
            "**Plan** (pick any)\n"
            "• A management step for a problem (med dose, lab order, referral)\n"
            "• Follow-up instruction or patient education note\n\n"
            "Scoring guidance\n"
            "• 5/5: Section contains ≥2 valid elements, all appropriately placed, and accurate to the source.\n"
            "• 3/5: Section contains exactly 1 valid element, appropriately placed and accurate.\n"
            "• 1/5: Section contains no valid elements or contains misplaced/incorrect info.\n\n"
            "2. Overall Criteria\n"
            "Then rate the entire note 1–5 on:\n"
            "• Coherence (logical flow and clarity)\n"
            "• Fluency  (grammar and readability)\n\n"
            "3. Output\n"
            "Respond ONLY with this JSON (no extra text):\n"
            "```json\n"
            "{\n"
            '  "Subjective":    /* 1–5 */,\n'
            '  "Objective":     /* 1–5 */,\n'
            '  "Assessment":    /* 1–5 */,\n'
            '  "Plan":          /* 1–5 */,\n'
            '  "Coherence":     /* 1–5 */,\n'
            '  "Fluency":       /* 1–5 */\n'
            "}\n"
            "```\n"
        )
        prefix = [{"role": "system", "content": sys_msg}]

        # 2) Insert few-shot examples if provided
        if few_shot:
            for ex in few_shot:
                prefix += [
                    {"role": "user", "content": f"Transcript:\n{ex['dialogue']}\n\nSOAP:\n{ex['soap']}"},
                    {"role": "assistant", "content": json.dumps(ex["scores"])}
                ]

        results = []
        # 3) For each example, send the transcript, summary (if any), and SOAP note
        for i, (dialogue, note) in enumerate(zip(self.dialogues, self.generated_notes)):
            # include summary only if self.summary_notes is set
            context_block = f"Transcript:\n{dialogue}"
            if self.summary_notes:
                context_block += f"\n\nFree-Text Summary:\n{self.summary_notes[i]}"
            context_block += f"\n\nGenerated SOAP Note:\n{note}"

            msgs = prefix + [{"role": "user", "content": context_block}]
            resp = openai.ChatCompletion.create(model=model, messages=msgs, temperature=0)
            results.append(json.loads(resp.choices[0].message.content))

        return results

    def evaluate_structure(self) -> List[Dict[str, object]]:
        """
        Evaluates SOAP note structure by:
          - Checking presence of each of S, O, A, P headers
          - Verifying they appear in the correct order
        """
        out = []
        # We look for either the single‐letter or full‐word header + colon:
        header_order = ["S", "O", "A", "P"]
        header_patterns = {
            "S": r"^\s*(?:S|Subjective):",
            "O": r"^\s*(?:O|Objective):",
            "A": r"^\s*(?:A|Assessment):",
            "P": r"^\s*(?:P|Plan):",
        }

        for note in self.generated_notes:
            presence = {}
            positions = {}
            for sec in header_order:
                pat = header_patterns[sec]
                m = re.search(pat, note, flags=re.IGNORECASE | re.MULTILINE)
                presence[sec] = bool(m)
                positions[sec] = m.start() if m else float("inf")

            # Now check that the four positions are in ascending order
            idxs = [positions[s] for s in header_order]
            order_ok = idxs == sorted(idxs)

            out.append({
                "section_presence": presence,
                "order_correct":     order_ok
            })

        return out


    #----- Internal helpers using arbitrary contexts -----
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

    # Combined context (dialogue + summary)
    def _combined_context(self) -> List[str]:
        if not self.summary_notes:
            raise ValueError("No summary_notes provided")
        return [d + "\n\n" + s for d, s in zip(self.dialogues, self.summary_notes)]
