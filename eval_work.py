# ─── Cell 1: Dependencies & Client Import ───────────────────────────────────────
# pip installs unchanged...
# from openai import OpenAI         # <-- use NVIDIA's OpenAI-compatible client
import openai
#from openai import OpenAI
import evaluate
import re, json, time, logging, torch
from typing import List, Dict, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from sentence_transformers import SentenceTransformer, util
from huggingface_hub import HfFolder
from together import Together
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import requests

class SOAPNoteEvaluator:
    """
    Evaluator for automatically generated SOAP notes, using:
      - QA-based factual consistency
      - Structure checks
      - Clinical relevance (BERTScore + embeddings)
      - Perplexity (GPT-2)
      - LLM-as-judge via NVIDIA DeepSeek R1 Distill QWEN-14B
    """

    def __init__(
        self,
        dialogues: List[str],
        generated_notes: List[str],
        summary_notes: Optional[List[str]] = None,
        together_api_key: Optional[str] = None,
        device: str = "cpu",
        hf_token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        # Input data
        self.dialogues       = dialogues
        self.generated_notes = generated_notes
        self.summary_notes   = summary_notes
        self.max_retries     = max_retries
        self.retry_delay     = retry_delay

        # Together API client
        if together_api_key:
            self.together_client = Together(api_key=together_api_key)
        else:
            self.together_client = None

        # HuggingFace token
        if hf_token:
            HfFolder.save_token(hf_token)

        # Device
        self.device = torch.device(device if device != "cpu" else "cpu")

        # QA pipelines
        self.qg = self._load_with_retry(
            pipeline,
            "text2text-generation",
            model="valhalla/t5-small-qg-prepend",
            device=0 if device != "cpu" else -1,
            token=hf_token
        )
        self.qa = self._load_with_retry(
            pipeline,
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=0 if device != "cpu" else -1,
            token=hf_token
        )

        # # NVIDIA DeepSeek client for LLM evaluation
        # self.llm_client = openai.OpenAI(
        #     api_key  = deepseek_api_key,
        #     base_url = deepseek_base_url
        # )

        # self.deepseek_api_key  = deepseek_api_key
        # self.deepseek_base_url = deepseek_base_url

        # Embeddings (BioBERT / SBERT fallback)
        try:
            self.clinbert = self._load_with_retry(
                SentenceTransformer,
                "pritamdeka/S-BioBert-snli-multinli-stsb",
                device=device
            )
        except Exception as e:
            logger.warning(f"BioBERT load failed: {e}. Falling back.")
            self.clinbert = self._load_with_retry(
                SentenceTransformer,
                "sentence-transformers/all-MiniLM-L6-v2",
                device=device
            )

        # BERTScore
        self.bertscore = evaluate.load("bertscore")
        self.bs_tokenizer = self._load_with_retry(
            AutoTokenizer.from_pretrained,
            "allenai/scibert_scivocab_uncased",
            token=hf_token
        )

        # Perplexity (GPT-2)
        self.ppl_tokenizer = self._load_with_retry(
            AutoTokenizer.from_pretrained,
            "gpt2",
            token=hf_token
        )
        self.ppl_model = self._load_with_retry(
            AutoModelForCausalLM.from_pretrained,
            "gpt2",
            token=hf_token
        ).to(self.device)
        if self.ppl_tokenizer.pad_token is None:
            self.ppl_tokenizer.pad_token = self.ppl_tokenizer.eos_token


    def _load_with_retry(self, loader_func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return loader_func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Load attempt {attempt+1} failed: {e}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All load attempts failed.")
                    raise


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

        # 5) LLM-as-judge, if Together API key is configured
        if self.together_client:
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
        for note in tqdm(self.generated_notes, desc="Computing perplexity"):
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
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        few_shot: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, float]]:
        """
        Uses Together's LLM to evaluate each generated SOAP note on section‐level content,
        coherence, and fluency, following the detailed prompt.
        Returns a list of dicts with keys:
          Subjective, Objective, Assessment, Plan, Coherence, Fluency
        """
        if not self.together_client:
            raise ValueError("Together API key not provided")

        # 1) Build the fixed system prompt
        sys_msg = """You are a clinical note quality evaluator. For each SOAP note provided, do the following:

1. Section-Level Content Checks
For each SOAP header—Subjective, Objective, Assessment, Plan—score **only** on:
- **Presence of at least one valid element** for that section (see lists below).
- **Appropriateness of every detail** (nothing misplaced).

Do **not** deduct points for missing optional items. Only deduct if a section either:
1. Contains information that belongs in a different header, or
2. Has no valid content at all.

Use these *valid* elements—at least one must appear for a "1" → "5" scale:

**Subjective** (pick any)
• Chief complaint in patient's words
• Any one OPQRST element (Onset, Quality, etc.)
• Past medical/family/social history
• Current medication or allergies

**Objective** (pick any)
• Any vital sign (e.g. BP, HR, RR, Temp, O₂ sat)
• Any physical exam finding
• Any diagnostic result (lab, imaging, EKG)

**Assessment** (pick any)
• A statement of primary diagnosis or impression
• Differential diagnoses (≥1)
• Numbered problem heading

**Plan** (pick any)
• A management step for a problem (med dose, lab order, referral)
• Follow-up instruction or patient education note

Scoring guidance
• 5/5: Section contains ≥2 valid elements, all appropriately placed, and accurate to the source.
• 4/5: Section contains ≥2 valid elements, mostly appropriately placed, with minor inaccuracies.
• 3/5: Section contains exactly 1 valid element, appropriately placed and accurate.
• 2/5: Section contains exactly 1 valid element, but with some placement issues or inaccuracies.
• 1/5: Section contains no valid elements or contains misplaced/incorrect info.

2. Overall Criteria
Then rate the entire note 1–5 on:
• Coherence (logical flow and clarity)
• Fluency  (grammar and readability)

3. Output
Respond ONLY with this JSON (no extra text):
```json
{
  "Subjective":    /* 1–5 */,
  "Objective":     /* 1–5 */,
  "Assessment":    /* 1–5 */,
  "Plan":          /* 1–5 */,
  "Coherence":     /* 1–5 */,
  "Fluency":       /* 1–5 */
}
```
"""
        results = []

        for i, (dialogue, note) in enumerate(tqdm(zip(self.dialogues, self.generated_notes), 
                                                desc="Evaluating with LLM", 
                                                total=len(self.dialogues))):
            # build the user message
            context = f"Transcript:\n{dialogue}"
            if self.summary_notes:
                context += f"\n\nFree-Text Summary:\n{self.summary_notes[i]}"
            context += f"\n\nGenerated SOAP Note:\n{note}"

            # Compose messages for Together API
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": context}
            ]

            # Add few-shot examples if provided
            if few_shot:
                for ex in few_shot:
                    messages.insert(-1, {"role": "user", "content": f"Transcript:\n{ex['dialogue']}\n\nSOAP:\n{ex['soap']}"})
                    messages.insert(-1, {"role": "assistant", "content": json.dumps(ex["scores"])})

            # Call Together API with retries
            for attempt in range(self.max_retries):
                try:
                    response = self.together_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0
                    )
                    
                    # Parse response
                    try:
                        content = response.choices[0].message.content
                        logger.info(f"Raw API response: {content}")  # Log the raw response
                        
                        # Try to clean the response if it's not valid JSON
                        content = content.strip()
                        if not content.startswith('{'):
                            # Try to find the first '{' and last '}'
                            start = content.find('{')
                            end = content.rfind('}') + 1
                            if start >= 0 and end > start:
                                content = content[start:end]
                        
                        result = json.loads(content)
                        # Validate the expected structure
                        expected_keys = {"Subjective", "Objective", "Assessment", "Plan", "Coherence", "Fluency"}
                        if not all(key in result for key in expected_keys):
                            raise ValueError(f"Missing expected keys in response. Got: {list(result.keys())}")
                            
                        results.append(result)
                        break  # Success, exit retry loop
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw content was: {content}")
                        if attempt == self.max_retries - 1:  # Last attempt
                            results.append({
                                "Subjective": 0,
                                "Objective": 0,
                                "Assessment": 0,
                                "Plan": 0,
                                "Coherence": 0,
                                "Fluency": 0
                            })
                    except ValueError as e:
                        logger.error(f"Invalid response structure: {e}")
                        if attempt == self.max_retries - 1:  # Last attempt
                            results.append({
                                "Subjective": 0,
                                "Objective": 0,
                                "Assessment": 0,
                                "Plan": 0,
                                "Coherence": 0,
                                "Fluency": 0
                            })
                except Exception as e:
                    logger.error(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        self.retry_delay *= 2  # Exponential backoff
                    else:
                        results.append({
                            "Subjective": 0,
                            "Objective": 0,
                            "Assessment": 0,
                            "Plan": 0,
                            "Coherence": 0,
                            "Fluency": 0
                        })

        return results


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
        for ctx, note in tqdm(zip(contexts, self.generated_notes), desc="Checking factual consistency"):
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
        for ctx, note in tqdm(zip(contexts, self.generated_notes), desc="Computing clinical relevance"):
            # 1) chunk both note and context
            note_chunks = self._chunk_text(note, max_len=500)  # Reduced from 512 to allow for special tokens
            ctx_chunks = self._chunk_text(ctx, max_len=500)    # Reduced from 512 to allow for special tokens

            # 2) compute BERTScore on each pair and average
            f1_scores = []
            # Ensure we have the same number of chunks for both note and context
            min_chunks = min(len(note_chunks), len(ctx_chunks))
            for i in range(min_chunks):
                bs = self.bertscore.compute(
                    predictions=[note_chunks[i]],
                    references=[ctx_chunks[i]],
                    model_type="allenai/scibert_scivocab_uncased",
                    batch_size=1  # Process one pair at a time to avoid memory issues
                )
                f1_scores.append(bs["f1"][0])

            # If we have chunks left in either note or context, process them separately
            if len(note_chunks) > min_chunks:
                for i in range(min_chunks, len(note_chunks)):
                    bs = self.bertscore.compute(
                        predictions=[note_chunks[i]],
                        references=[""],  # Empty reference for extra chunks
                        model_type="allenai/scibert_scivocab_uncased",
                        batch_size=1
                    )
                    f1_scores.append(bs["f1"][0])

            if len(ctx_chunks) > min_chunks:
                for i in range(min_chunks, len(ctx_chunks)):
                    bs = self.bertscore.compute(
                        predictions=[""],  # Empty prediction for extra chunks
                        references=[ctx_chunks[i]],
                        model_type="allenai/scibert_scivocab_uncased",
                        batch_size=1
                    )
                    f1_scores.append(bs["f1"][0])

            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

            # 3) embedding similarity still on full text
            note_emb = self.clinbert.encode(note, convert_to_tensor=True)
            ctx_emb = self.clinbert.encode(ctx, convert_to_tensor=True)
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



# ─── Cell 4: Imports & Entry Point ─────────────────────────────────────────────

import os
import json
import traceback

import pandas as pd
from together import Together

# Set Together API key
os.environ["TOGETHER_API_KEY"] = "da17fcf94658628f21edf65df67a7b701f0dbe2afb4e9fc23189d884fb1d50e5"

def main(
    csv_path: str,
    output_dir: str,
    device: str = "cpu",
    max_retries: int = 5,
    retry_delay: int = 10
):
    """
    Runs SOAPNoteEvaluator over the data in `csv_path`, checkpointing per-sample
    into JSON and at the end saving a CSV. All outputs go under `output_dir`.
    """
    # 1) Prepare folders & paths
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint_case3.json")
    final_csv_path  = os.path.join(output_dir, "results_case3.csv")

    # 2) Load data
    df = pd.read_csv(csv_path)
    dialogues       = df["dialogue"].tolist()
    summary_notes   = df["note"].tolist()
    augmented_notes = df["combined_soap_summary"].tolist()

    # 3) Instantiate evaluator with Together API
    evaluator = SOAPNoteEvaluator(
        dialogues=dialogues,
        generated_notes=augmented_notes,
        summary_notes=summary_notes,
        together_api_key=os.environ["TOGETHER_API_KEY"],
        device=device,
        hf_token=os.environ.get("HF_TOKEN"),
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    # 4) Load or init checkpoint
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
    else:
        results = []
    start_idx = len(results)
    print(f"Resuming at sample {start_idx}/{len(df)}")

    def save_checkpoint():
        """Helper function to save checkpoint"""
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Checkpoint saved: {checkpoint_path}")

    # 5) Per-sample loop
    for i in tqdm(range(start_idx, len(df)), desc="Processing samples"):
        rec = {
            "sample_id": i,
            "dialogue": dialogues[i],
            "note": summary_notes[i],       # ← the *input* note column
            "soap_note": augmented_notes[i],
        }
        try:
            # Structure
            struct = evaluator.evaluate_structure()[i]
            rec.update(
                section_presence=struct["section_presence"],
                order_correct=struct["order_correct"],
            )
            save_checkpoint()  # Save after structure check

            # Combined context
            ctx = dialogues[i] + "\n\n" + summary_notes[i]
            
            # QA factuality
            qa_result = evaluator._qa_factuality([ctx])[0]
            rec.update(qa_result)
            save_checkpoint()  # Save after QA factuality
            
            # Clinical relevance
            clin_result = evaluator._clinical_relevance([ctx])[0]
            rec.update(clin_result)
            save_checkpoint()  # Save after clinical relevance
            
            # Perplexity
            ppl_result = evaluator._perplexity()[i]
            rec.update(ppl_result)
            save_checkpoint()  # Save after perplexity

            # LLM scores (if available)
            try:
                llm_result = evaluator.evaluate_llm(
                    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
                )[i]
                rec["llm"] = llm_result
            except Exception as e2:
                rec["llm"]       = None
                rec["llm_error"] = str(e2)
            save_checkpoint()  # Save after LLM evaluation

        except Exception:
            rec["metrics"] = None
            rec["error"]   = traceback.format_exc()
            save_checkpoint()  # Save even if there's an error

        # append to results
        results.append(rec)

    # 6) Final CSV export
    final_df = pd.json_normalize(results)
    final_df.to_csv(final_csv_path, index=False)
    print("All done!")
    print("  • checkpoint JSON:", checkpoint_path)
    print("  • final  CSV   :", final_csv_path)

    return final_df


# ─── Cell 5: Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # adjust these paths as needed in Colab
    CSV_PATH    = "combined_soap_summaries.csv"
    OUTPUT_DIR  = "third_case"

    df_results = main(
        csv_path   = CSV_PATH,
        output_dir = OUTPUT_DIR,
        device     = "cuda",       # <- was "cpu", now uses the GPU
        max_retries=5,
        retry_delay=10
    )
