from __future__ import annotations

import argparse

"""
Drug‑information RAG agent with async loading, caching & specialist review
==============================================================================
Features:
  - Asynchronous embedding generation during initial load.
  - Disk caching of processed XML data (chunks & vectors).
  - Configurable number of specialists sampled per review round.

Stages (printed to stdout):
  • [CACHE] Status (Load / Build / Valid)
  • [STATUS] Knowledge Retrieval – after pulling candidate chunks from VDB
  • [STATUS] Reasoning           – while the assistant drafts / revises an answer
  • [STATUS] Expertise           – when sampled specialists are satisfied / review round ends
If *any* sampled specialist requests changes, status rolls back to **Reasoning**.

Run example (default data dir "structured_drug_data" must have *.xml files):
    python drug_info_agent_async.py --num-specialists 2 \
        "What are the interactions between prednisone and carvedilol?"
"""

import os
import re
import sys
import json
import random
import xml.etree.ElementTree as ET
import asyncio # Added for async operations
import pickle  # Added for caching
import hashlib # Added for cache validation (optional, using timestamps here)
import time    # Added for cache validation
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TypedDict, Optional, Any # Added Optional, Any

import numpy as np
import openai
from openai import AsyncOpenAI # Use Async client
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
load_dotenv()
# Use AsyncOpenAI client
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not aclient.api_key:
    sys.exit("ERROR: OPENAI_API_KEY not set (env or .env)")

STRICT_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o-mini"
SPECIALIST_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
MAX_CHARS_PER_CHUNK = 2_000
MAX_REVISIONS = 2
NUM_SPECIALISTS_TO_USE = 1 # default, can be overridden by CLI
DEFAULT_DATA_DIR = "structured_drug_data"
CACHE_FILENAME = "vdb_cache.pkl"
# How many texts to send to embedding API in one batch
EMBEDDING_BATCH_SIZE = 100 # Adjust based on API limits and performance

SECTION_WHITELIST: set[str] = {
    "DESCRIPTION SECTION", "CLINICAL PHARMACOLOGY SECTION", "INDICATIONS & USAGE SECTION",
    "DOSAGE & ADMINISTRATION SECTION", "DOSAGE FORMS & STRENGTHS SECTION", "CONTRAINDICATIONS SECTION",
    "WARNINGS AND PRECAUTIONS SECTION", "ADVERSE REACTIONS SECTION", "DRUG INTERACTIONS SECTION",
    "USE IN SPECIFIC POPULATIONS SECTION", "OVERDOSAGE SECTION", "NONCLINICAL TOXICOLOGY SECTION",
    "CLINICAL STUDIES SECTION", "HOW SUPPLIED SECTION", "PATIENT COUNSELING INFORMATION",
    "PACKAGE LABEL.PRINCIPAL DISPLAY PANEL", "MECHANISM OF ACTION SECTION", "PHARMACODYNAMICS SECTION",
    "PHARMACOKINETICS SECTION", "CARCINOGENESIS & MUTAGENESIS & IMPAIRMENT OF FERTILITY SECTION",
    "PREGNANCY SECTION", "LACTATION SECTION", "PEDIATRIC USE SECTION", "GERIATRIC USE SECTION",
    "DRUG", "DESCRIPTION", "PATIENT INFORMATION", "WARNINGS SECTION", "PRECAUTIONS SECTION",
}

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def log_status(stage: str):
    pass
    #print(f"[STATUS] {stage}")

def log_cache(status: str, message: str = ""):
     pass
     #print(f"[CACHE] {status} {message}")

# ---- XML parsing → graph (synchronous, as it's CPU bound) ------------------
def build_graph(xml_path: Path | str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        #print(f"Warning: [build_graph] Skipping {xml_path}: {e}")
        return out

    ns = {"hl7": "urn:hl7-org:v3"}
    root_name = Path(xml_path).stem # Use filename stem
    for n, sec in enumerate(tree.findall(".//hl7:section", ns)):
        label = ""; node_id_base = sec.attrib.get("ID", f"sec{n}")
        code = sec.find("hl7:code", ns)
        if code is not None and code.attrib.get("displayName"):
            label = code.attrib["displayName"].upper()
        else:
            ttl = sec.find("hl7:title", ns)
            if ttl is not None and ttl.text:
                label = ttl.text.strip().upper()
            else:
                 # Fallback if no proper label found
                 label = node_id_base.upper() # Or maybe skip?

        norm = re.sub(r"\s*SECTION$", "", label).strip()

        # Slightly refined whitelist check
        is_whitelisted = False
        if norm in SECTION_WHITELIST:
             is_whitelisted = True
        else:
             # Check for partial matches more carefully
             # Avoid overly broad matches like "DRUG" matching "DRUG INTERACTIONS"
             for wl_item in SECTION_WHITELIST:
                  # Check if norm is a significant part of a whitelist item or vice versa
                  if (len(norm) > 4 and norm in wl_item) or \
                     (len(wl_item) > 4 and wl_item in norm and wl_item != norm):
                       is_whitelisted = True
                       break

        if not is_whitelisted:
            continue

        unique_id = f"{root_name}__{node_id_base}" # Use unique ID format
        # Cleaner text extraction
        parts = [elem.text or "" for elem in sec.iter() if elem.text] + \
                [elem.tail or "" for elem in sec.iter() if elem.tail]
        text = re.sub(r"\s+", " ", " ".join(p.strip() for p in parts if p.strip())).strip()

        if text:
            out[unique_id] = text
    return out

# ---- chunking (synchronous) ------------------------------------------------
def make_chunks(graph: Dict[str, str]) -> List[Tuple[str, str]]:
    chunks_list: List[Tuple[str, str]] = []
    for sid, txt in graph.items():
        if not txt: continue # Skip empty text
        if len(txt) <= MAX_CHARS_PER_CHUNK:
            chunks_list.append((f"{sid}#0", txt))
        else:
            step = MAX_CHARS_PER_CHUNK - MAX_CHARS_PER_CHUNK // 5 # Overlap
            for i, st in enumerate(range(0, len(txt), step)):
                chunk_text = txt[st:st + MAX_CHARS_PER_CHUNK].strip()
                if chunk_text: # Ensure chunk isn't empty after strip
                    chunks_list.append((f"{sid}#chunk{i}", chunk_text))
    return chunks_list

# ---- vector DB -------------------------------------------------------------
# VDB now primarily acts as a data container, loading/saving handled by the agent
@dataclass
class VDB:
    ids: List[str] = field(default_factory=list)
    txts: List[str] = field(default_factory=list)
    vecs: List[np.ndarray] = field(default_factory=list)
    # Add timestamp to VDB for cache validation
    timestamp: float = field(default_factory=time.time)

    def is_empty(self) -> bool:
         return not self.ids or not self.vecs

    def search(self, q: str, k: int = 8) -> List[Tuple[str, str]]:
        if self.is_empty():
            #print("Warning: VDB Search called on empty VDB.")
            return []
        if not q or not q.strip():
             #print("Warning: Empty search query.")
             return []
        # Ensure vecs is a list of numpy arrays
        if not isinstance(self.vecs, list) or not all(isinstance(v, np.ndarray) for v in self.vecs):
             #print("Error: VDB vectors are not in the expected format (List[np.ndarray]).")
             return []

        try:
            # Synchronous embedding call for the query (could be async if needed elsewhere)
            q_embedding_response = openai.embeddings.create(model=EMBED_MODEL, input=[q.strip()])
            if not q_embedding_response.data:
                 #print("Error: Could not generate query embedding.")
                 return []
            qv = np.array(q_embedding_response.data[0].embedding, dtype=np.float32)

            # Stack vectors for efficient calculation
            mat = np.stack(self.vecs)
            norms = np.linalg.norm(mat, axis=1)
            # Handle potential zero vectors
            zero_vector_indices = np.where(norms == 0)[0]
            if len(zero_vector_indices) > 0:
                #print(f"Warning: Found {len(zero_vector_indices)} zero vectors in VDB. Excluding from similarity calculation.")
                norms[zero_vector_indices] = 1e-9 # Avoid division by zero

            sims = mat @ qv / (norms * np.linalg.norm(qv))

            # Exclude zero vectors from results if any
            valid_indices = np.where(norms > 1e-9)[0]
            if len(valid_indices) < len(sims):
                sims = sims[valid_indices]
                original_indices = valid_indices # Keep track of original indices

            else:
                 original_indices = np.arange(len(sims))

            actual_k = min(k, len(sims))
            if actual_k == 0: return []

            # Get top k indices from the valid similarities
            # Using argpartition for potential speedup if k is much smaller than len(sims)
            if actual_k < len(sims):
                 top_k_indices_relative = np.argpartition(sims, -actual_k)[-actual_k:]
            else:
                 top_k_indices_relative = np.arange(len(sims))

            # Sort these top k indices by similarity score
            sorted_top_k_indices = top_k_indices_relative[np.argsort(sims[top_k_indices_relative])[::-1]]

            # Map back to original indices in self.ids/self.txts
            final_indices = original_indices[sorted_top_k_indices]

            return [(self.ids[i], self.txts[i]) for i in final_indices]

        except openai.OpenAIError as api_err:
             #print(f"Error during VDB search (API call): {api_err}")
             return []
        except Exception as e:
            #print(f"Error during VDB search: {e}")
            import traceback
            traceback.print_exc()
            return []


# ---- specialists (using async client now) -----------------------------------
class SpecialistFeedback(TypedDict):
    status: str
    feedback: Optional[str]
    specialist_name: str

class Specialist:
    def __init__(self, name: str, prompt: str):
        self.name = name
        self.prompt = prompt

    async def review(self, q: str, ex: str, ans: str) -> SpecialistFeedback:
        """Asynchronously reviews the answer."""
        # Use the global async client `aclient`
        #print(f"  Specialist review started: {self.name}")
        try:
            response = await aclient.chat.completions.create(
                model=SPECIALIST_MODEL,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": f"Q:\n{q}\n\nExcerpts:\n{ex or 'None'}\n\nAnswer:\n{ans}"},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            rsp_content = response.choices[0].message.content
            if not rsp_content:
                 raise ValueError("LLM returned empty content")
            data = json.loads(rsp_content.strip())

            # Validate JSON structure and content
            if not isinstance(data, dict) or "status" not in data:
                 raise ValueError("Invalid JSON format: Missing 'status'")
            if data["status"] not in ["Satisfied", "Needs Revision"]:
                 raise ValueError(f"Invalid status value: {data['status']}")

            feedback = data.get("feedback")
            if data["status"] == "Needs Revision" and not feedback:
                 #print(f"Warning: {self.name} requested revision but provided no feedback text.")
                 feedback = f"Revision needed according to {self.name} (no specific details provided)." # Provide default feedback
            elif data["status"] == "Satisfied":
                feedback = None # Ensure feedback is None when satisfied

            #print(f"  Specialist review finished: {self.name} -> {data['status']}")
            return {"status": data["status"], "feedback": feedback, "specialist_name": self.name}

        except json.JSONDecodeError as e:
            #print(f"Error: {self.name} specialist returned invalid JSON: {e}. Raw response: {rsp_content if 'rsp_content' in locals() else 'N/A'}")
            return {"status": "Needs Revision", "feedback": f"{self.name}: Response format error.", "specialist_name": self.name}
        except openai.OpenAIError as api_err:
             #print(f"Error: API call failed for {self.name} specialist review: {api_err}")
             return {"status": "Needs Revision", "feedback": f"{self.name}: API error during review.", "specialist_name": self.name}
        except Exception as e:
            #print(f"Error during {self.name} specialist review: {e}")
            return {"status": "Needs Revision", "feedback": f"{self.name}: Unexpected error during review.", "specialist_name": self.name}

# Updated Prompts for Clarity and Brevity
JSON_INSTRUCTION = "Output ONLY JSON: {\"status\": \"Satisfied\"|\"Needs Revision\", \"feedback\": \"...feedback text or null...\"}"
INTERACTION_PROMPT = f"Drug Interaction Specialist. Focus: Drug-drug interactions (accuracy, mechanism, severity, management). {JSON_INSTRUCTION}"
PHARM_PROMPT = f"Pharmacology Specialist. Focus: Contraindications, warnings, precautions, adverse reactions. {JSON_INSTRUCTION}"
CLIN_PROMPT = f"Clinical Specialist. Focus: Dosage, administration, specific populations (pediatric, geriatric, renal/hepatic), PK/PD relevance. {JSON_INSTRUCTION}"

# ---------------------------------------------------------------------------
# 3. Main agent class with Async Loading & Caching
# ---------------------------------------------------------------------------
class DrugInfoAgent:
    def __init__(self, data_dir: Path, cache_dir: Path = Path(".")):
        self.data_dir = data_dir
        self.cache_path = cache_dir / CACHE_FILENAME
        self.vdb = VDB() # Initialize empty VDB
        self.specialists = [
            Specialist("Interaction", INTERACTION_PROMPT),
            Specialist("Pharmacology", PHARM_PROMPT),
            Specialist("Clinical", CLIN_PROMPT),
        ]
        # Load or build VDB asynchronously upon initialization
        # Run the async method in the main event loop
        try:
            # If an event loop is already running (e.g., in Jupyter), use ensure_future
            try:
                 loop = asyncio.get_running_loop()
                 loop.create_task(self._load_or_build_vdb_async()) # Schedule but don't wait here
                 # Note: This means the agent might not be fully ready immediately.
                 # A better approach for scripts is to await in main.
                 #print("Note: VDB loading initiated in background (running loop detected).")
            except RuntimeError: # No running loop
                 # If no loop is running (typical script execution), run until complete
                 asyncio.run(self._load_or_build_vdb_async())

        except Exception as e:
             #print(f"FATAL: Failed to load or build VDB: {e}")
             sys.exit(1)

    async def revise(self, q: str, prev_answer: str, external_feedback: str) -> str:
        """
        Create a revised answer using feedback coming from Leonardo
        rather than the built-in specialist loop.
        """
        log_status("Reasoning (External Revision)")
        revised = self._chat(
            FALLBACK_MODEL,
            [
                {"role": "system",
                 "content": (
                     "You are revising your previous medical answer based on external factual-accuracy feedback. "
                     "If score of a sentence is in range 0.8-1.0, factually check and fix the sentence using you knowledge base. "
                     "Correct every statement the feedback "
                     "flags as wrong or dubious; keep citations; remove hallucinations; "
                     "be concise and clinically safe."
                 )},
                {"role": "user",
                 "content": (
                     f"Question: {q}\n\n"
                     f"Previous answer (to be fixed):\n{prev_answer}\n\n"
                     f"\"\"\"External feedback – ONE FACT PER LINE (incorrect ⇢ correct):"
                     f"{external_feedback}\"\"\"\n"
    +                "Rewrite the whole answer.  Remove or correct every wrong fact, "
    +                "keep correct facts, keep citations.  Respond **only** with the new answer."
                 )},
            ],
            temp=0.3,  # deterministic but allows slight re-wording
        )
        #print('external_feedback:', external_feedback)
        return revised

    def _is_cache_valid(self) -> bool:
        """Checks if the cache file exists and is newer than the data files."""
        if not self.cache_path.exists():
            return False
        cache_mtime = self.cache_path.stat().st_mtime
        xml_files = list(self.data_dir.glob("*.xml"))
        if not xml_files: # If no source files, cache is vacuously valid if it exists
             return True
        # Check if any XML file is newer than the cache
        for f in xml_files:
            if f.stat().st_mtime > cache_mtime:
                log_cache("Invalid", f"Reason: {f.name} modified after cache creation.")
                return False
        return True

    async def _generate_embeddings_async(self, chunks_to_embed: List[Tuple[str, str]]) -> Optional[VDB]:
         """Generates embeddings asynchronously for a list of chunks."""
         vdb_temp = VDB()
         if not chunks_to_embed:
              return vdb_temp # Return empty VDB if no chunks

         vdb_temp.ids, vdb_temp.txts = zip(*chunks_to_embed)
         vdb_temp.ids = list(vdb_temp.ids)
         vdb_temp.txts = list(vdb_temp.txts)
         vdb_temp.vecs = [np.array([])] * len(vdb_temp.ids) # Pre-allocate, replace later

         tasks = []
         total_chunks = len(vdb_temp.txts)
         #print(f"Generating embeddings for {total_chunks} chunks in batches of {EMBEDDING_BATCH_SIZE}...")

         for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
              batch_texts = vdb_temp.txts[i : i + EMBEDDING_BATCH_SIZE]
              # Ensure texts are valid strings
              valid_batch_texts = [t if isinstance(t, str) and t.strip() else " " for t in batch_texts]
              if valid_batch_texts:
                   tasks.append(
                        aclient.embeddings.create(model=EMBED_MODEL, input=valid_batch_texts)
                   )
              else:
                  pass
                   #print(f"Warning: Skipping empty batch at index {i}")


         if not tasks:
              #print("Error: No valid embedding tasks created.")
              return None

         try:
            # Gather results from all concurrent tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
         except Exception as e:
             #print(f"Error during asyncio.gather for embeddings: {e}")
             return None # Indicate failure

         processed_count = 0
         vec_idx = 0
         for i, result in enumerate(results):
             batch_start_index = i * EMBEDDING_BATCH_SIZE
             if isinstance(result, Exception):
                  #print(f"Error embedding batch starting at index {batch_start_index}: {result}")
                  # Fill corresponding vectors with zeros or handle error appropriately
                  batch_size = len(vdb_temp.txts[batch_start_index : batch_start_index + EMBEDDING_BATCH_SIZE])
                  for j in range(batch_size):
                       if vec_idx < len(vdb_temp.vecs):
                            vdb_temp.vecs[vec_idx] = np.zeros(1536, dtype=np.float32) # Adjust dim if needed
                       vec_idx += 1
                  continue # Move to next batch result

             if not hasattr(result, 'data') or not result.data:
                  #print(f"Warning: Empty data returned for embedding batch starting at index {batch_start_index}")
                  batch_size = len(vdb_temp.txts[batch_start_index : batch_start_index + EMBEDDING_BATCH_SIZE])
                  for j in range(batch_size):
                       if vec_idx < len(vdb_temp.vecs):
                            vdb_temp.vecs[vec_idx] = np.zeros(1536, dtype=np.float32) # Adjust dim if needed
                       vec_idx += 1
                  continue

             # Process successful batch
             try:
                 for embedding_data in result.data:
                      if vec_idx < len(vdb_temp.vecs): # Ensure we don't go out of bounds
                           vdb_temp.vecs[vec_idx] = np.array(embedding_data.embedding, dtype=np.float32)
                           vec_idx += 1
                      else:
                           #print(f"Warning: More embeddings received than expected chunks ({vec_idx} >= {len(vdb_temp.vecs)}).")
                           break
                 processed_count += len(result.data)
             except Exception as proc_err:
                   #print(f"Error processing embedding result for batch starting at index {batch_start_index}: {proc_err}")
                   # Fill remaining vectors for this batch with zeros
                   batch_size = len(vdb_temp.txts[batch_start_index : batch_start_index + EMBEDDING_BATCH_SIZE])
                   remaining_in_batch = batch_size - (vec_idx - batch_start_index)
                   for _ in range(remaining_in_batch):
                        if vec_idx < len(vdb_temp.vecs):
                              vdb_temp.vecs[vec_idx] = np.zeros(1536, dtype=np.float32)
                        vec_idx += 1


         #print(f"Embedding generation complete. Successfully processed {processed_count}/{total_chunks} embeddings.")
         # Check if all vectors were populated
         if any(v.size == 0 for v in vdb_temp.vecs):
              #print("Warning: Some vectors could not be generated.")
              # Replace empty arrays with zero vectors
              zero_vec = np.zeros(1536, dtype=np.float32) # Adjust dim if needed
              vdb_temp.vecs = [v if v.size > 0 else zero_vec for v in vdb_temp.vecs]


         vdb_temp.timestamp = time.time() # Set timestamp after successful build
         return vdb_temp


    async def _load_or_build_vdb_async(self):
        """Loads VDB from cache if valid, otherwise builds it asynchronously."""
        if self._is_cache_valid():
            log_cache("Valid", f"Loading VDB from {self.cache_path}")
            try:
                with open(self.cache_path, "rb") as f:
                    loaded_vdb = pickle.load(f)
                    if isinstance(loaded_vdb, VDB):
                         self.vdb = loaded_vdb
                         log_cache("Load Success")
                         #print(f"[ingest] Loaded {len(self.vdb.ids)} chunks from cache.")
                         return # Successfully loaded from cache
                    else:
                         log_cache("Load Failed", "Invalid data format in cache file.")
            except Exception as e:
                log_cache("Load Failed", f"Error reading cache: {e}. Rebuilding.")
        else:
            log_cache("Stale or Missing", f"Cache file {self.cache_path} not found or outdated. Building VDB.")

        # --- Build VDB ---
        log_status("Ingesting XML Data")
        xml_files = list(self.data_dir.glob("*.xml"))
        if not xml_files:
             #print(f"Warning: No XML files found in {self.data_dir}. VDB will be empty.")
             self.vdb = VDB() # Ensure VDB is empty
             return

        all_chunks = []
        # Parsing and chunking remain synchronous as they are CPU-bound
        for f_path in xml_files:
            graph = build_graph(f_path)
            if graph:
                 all_chunks.extend(make_chunks(graph))

        if not all_chunks:
             #print(f"Warning: No text chunks could be extracted from XML files in {self.data_dir}. VDB will be empty.")
             self.vdb = VDB()
             return

        # Generate embeddings asynchronously
        log_status("Generating Embeddings (Async)")
        start_embed_time = time.time()
        new_vdb = await self._generate_embeddings_async(all_chunks)
        embed_duration = time.time() - start_embed_time
        log_status(f"Embedding Generation Finished ({embed_duration:.2f}s)")


        if new_vdb and not new_vdb.is_empty():
            self.vdb = new_vdb
            # --- Save to cache ---
            log_cache("Saving", f"Saving VDB to {self.cache_path}")
            try:
                # Create cache directory if it doesn't exist
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.vdb, f)
                log_cache("Save Success")
                #print(f"[ingest] Processed {len(xml_files)} XMLs -> {len(self.vdb.ids)} chunks.")
            except Exception as e:
                log_cache("Save Failed", f"Error writing cache: {e}")
        else:
             #print("Error: Failed to generate embeddings. VDB remains empty.")
             self.vdb = VDB() # Ensure VDB is empty on failure

    # --- Chat completion using synchronous client (can be switched to async if agent logic becomes async) ---
    def _chat(self, model: str, msgs: list, temp: float = 0.0) -> str:
        # Using the synchronous client for chat calls within the answer method for simplicity
        # If the whole answer method becomes async, switch to aclient here too.
        try:
            sync_client = openai.OpenAI(api_key=aclient.api_key) # Create sync client instance
            response = sync_client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temp
            )
            content = response.choices[0].message.content
            return content.strip() if content else "Error: LLM returned empty response."
        except openai.OpenAIError as api_err:
            #print(f"Error calling LLM ({model}): {api_err}")
            return f"Error: LLM call failed ({api_err})"
        except Exception as e:
             #print(f"Unexpected error during chat call: {e}")
             return f"Error: Unexpected issue generating response."


    # --- Answer method (now uses async specialist reviews) ---
    async def answer(self, q: str) -> str:
        """Generates answer, including async specialist review."""
        # --- 0) Ensure VDB is ready ---
        # Typically loaded during __init__, but add check if agent might be used before loading finishes
        if self.vdb.is_empty():
             # Option 1: Wait if loading is still happening (needs more complex state management)
             # Option 2: Try loading again (might be redundant if init failed)
             # Option 3: Return error
             #print("Warning: VDB is empty. Attempting to load/build again.")
             await self._load_or_build_vdb_async() # Ensure it's loaded
             if self.vdb.is_empty():
                  return "Error: Knowledge base (VDB) could not be loaded or built. Cannot answer question."


        # --- 1) Retrieval ---
        log_status("Knowledge Retrieval")
        ctx = self.vdb.search(q, 8)
        excerpts = "\n\n".join(f"[{cid}] {txt}" for cid, txt in ctx)
        #print(f"Retrieved {len(ctx)} excerpts.")

        # --- 2) Strict answer or fallback ---
        log_status("Reasoning (Initial Draft)")
        strict = self._chat(STRICT_MODEL, [
            {"role": "system", "content": "You are a Drug Label Expert. Use ONLY excerpts. Cite ids like [filename__id#chunk]. If fact missing reply exactly: I don't know based on the provided drug label excerpts."},
            {"role": "user", "content": f"Excerpts:\n{excerpts or 'None available.'}\n\nQuestion: {q}"},
        ])

        is_strict_fallback = strict.lower().startswith("i don't know")
        if is_strict_fallback:
            #print("Strict answer not found. Generating fallback.")
            ans = self._chat(
                FALLBACK_MODEL,
                [
                    {"role": "system", "content": "You are a clinical assistant. Use excerpts (cite ids like [filename__id#chunk]) + general knowledge. Provide a safe, accurate answer."},
                    {"role": "system", "content": f"Excerpts:\n{excerpts or 'None available.'}"},
                    {"role": "user", "content": q},
                ],
                temp=0.4,
            )
        else:
            #print("Generated strict answer from excerpts.")
            ans = strict

        # --- 3) Async Review cycles ---
        num_specialists_available = len(self.specialists)
        num_to_sample = min(NUM_SPECIALISTS_TO_USE, num_specialists_available)

        for i in range(MAX_REVISIONS + 1): # +1 to allow initial check
            log_status(f"Expertise (Review Cycle {i+1}/{MAX_REVISIONS+1})")
            if num_specialists_available == 0:
                 #print("Warning: No specialists configured. Skipping review.")
                 break
            if num_to_sample == 0:
                 #print("Warning: Configured to sample 0 specialists. Skipping review.")
                 break

            sampled_specialists = random.sample(self.specialists, num_to_sample)
            #print(f"Sampling {len(sampled_specialists)} specialists for review: {[s.name for s in sampled_specialists]}")

            # Run reviews concurrently
            review_tasks = [sp.review(q, excerpts, ans) for sp in sampled_specialists]
            feedback_results = await asyncio.gather(*review_tasks, return_exceptions=True)

            all_ok = True
            feedback_lines: List[str] = []
            for res in feedback_results:
                if isinstance(res, Exception):
                     #print(f"Error during specialist review task: {res}")
                     all_ok = False
                     feedback_lines.append(f"- Specialist Task Error: {res}") # Include error in feedback
                     continue # Treat task error as needing revision

                # We now expect res to be SpecialistFeedback TypedDict
                if isinstance(res, dict) and res.get("status") == "Needs Revision":
                    all_ok = False
                    if res.get("feedback"):
                        feedback_lines.append(f"- {res.get('specialist_name', 'Unknown Specialist')}: {res['feedback']}")
                    else:
                        # Handle case where feedback is None/empty despite Needs Revision status
                         feedback_lines.append(f"- {res.get('specialist_name', 'Unknown Specialist')}: Revision needed (no details provided).")


            if all_ok:
                log_status("Expertise (Satisfied)")
                # Add prefix only if we started with fallback
                final_answer = ans if not is_strict_fallback else f"I COULD NOT ANSWER FROM THE DRUG LABELS ALONE. BASED ON GENERAL KNOWLEDGE AND SPECIALIST REVIEW:\n\n{ans}"
                log_status("Done")
                return final_answer

            # Check if max revisions reached *before* attempting rewrite
            if i >= MAX_REVISIONS:
                 #print("Maximum revisions reached.")
                 break # Exit loop, will return last answer with warning


            # --- Needs revision: Roll back to reasoning ---
            log_status("Reasoning (Revision)")
            #print("Revising answer based on specialist feedback...")
            combined_feedback = "\n".join(feedback_lines) or "(No specific feedback details captured)"
            ans = self._chat(FALLBACK_MODEL, [
                {"role": "system", "content": "Revise your previous answer based on expert feedback below. Address the points raised. Ensure accuracy and safety. Cite excerpts like [filename__id#chunk] if used."},
                {"role": "user", "content": (
                    f"Question: {q}\n\n"
                    f"Excerpts:\n{excerpts or 'None available.'}\n\n"
                    f"Your Previous Answer:\n{ans}\n\n"
                    f"Combined Specialist Feedback:\n{combined_feedback}"
                )},
            ], temp=0.3)
            #print("Revision complete.")


        # --- Exceeded attempts or finished loop needing revision ---
        log_status("Done (Max Revisions Reached)")
        warning = "\n\n[Note: Specialists requested further revisions, but the maximum number of attempts was reached. Please review this answer carefully.]"
        # Add prefix only if we started with fallback
        final_answer = ans if not is_strict_fallback else f"I COULD NOT ANSWER FROM THE DRUG LABELS ALONE. BASED ON GENERAL KNOWLEDGE (REVISED {MAX_REVISIONS} TIMES):\n\n{ans}"
        return final_answer + warning


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
async def main():
     """Main async function to setup and run the agent."""
     global NUM_SPECIALISTS_TO_USE  # move this up immediately

     log_status("Started")
     p = argparse.ArgumentParser(
          description="Drug-info agent with async loading, caching & specialist review",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter
     )
     p.add_argument("query", nargs="+", help="User question")
     p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory with SPL XML files")
     p.add_argument("--cache-dir", default=".", help="Directory to store the cache file")
     p.add_argument("--num-specialists", type=int, default=NUM_SPECIALISTS_TO_USE, help="Max specialists sampled per round (randomly chosen)")
     p.add_argument("--force-rebuild", action="store_true", help="Force rebuild of cache, ignore existing cache file.")
     args = p.parse_args()

     # --- Validate Args ---
     data_dir = Path(args.data_dir)
     cache_dir = Path(args.cache_dir)
     if not data_dir.is_dir():
          #print(f"ERROR: Data directory '{data_dir}' not found or is not a directory.")
          sys.exit(1)

     NUM_SPECIALISTS_TO_USE = max(0, args.num_specialists) # Allow 0 specialists

     # Delete cache if forcing rebuild
     cache_path = cache_dir / CACHE_FILENAME
     if args.force_rebuild and cache_path.exists():
          log_cache("Forced Rebuild", f"Deleting existing cache file: {cache_path}")
          try:
               cache_path.unlink()
          except OSError as e:
              pass
               #print(f"Warning: Could not delete cache file {cache_path}: {e}")


     # --- Initialize Agent (which handles loading/building VDB) ---
     # Initialization now triggers the async loading/building process internally
     #print(f"Initializing agent with data from '{data_dir}'...")
     start_init_time = time.time()
     # Pass cache_dir to agent init
     bot = DrugInfoAgent(data_dir, cache_dir=cache_dir)
     init_duration = time.time() - start_init_time
     #print(f"Agent initialized. VDB Load/Build took: {init_duration:.2f}s")

     if bot.vdb.is_empty() and list(data_dir.glob("*.xml")):
         pass
          #print("ERROR: Agent initialized, but VDB is empty despite XML files being present. Check logs.")
          # Depending on strictness, you might exit here
          # sys.exit(1)


     # --- Run Query ---
     question = " ".join(args.query)
     #print(f"\n--- Answering Query: {question} ---")
     start_answer_time = time.time()
     final_answer = await bot.answer(question) # Now await the answer method
     answer_duration = time.time() - start_answer_time

     #print("\n" + "=" * 80)
     #print("Final Answer:")
     #print(final_answer)
     #print("=" * 80)
     #print(f"Answering took: {answer_duration:.2f}s")


# ------------------------------------------------------------------
# Public helper for external revision (used by main.py)
# ------------------------------------------------------------------
async def revise(self, q: str, prev_answer: str, external_feedback: str) -> str:
    """
    Create a revised answer using feedback coming from Leonardo
    rather than the built-in specialist loop.
    """
    log_status("Reasoning (External Revision)")
    revised = self._chat(
        FALLBACK_MODEL,
        [
            {"role": "system",
             "content": (
                 "You are revising your previous medical answer based on external "
                 "factual-accuracy feedback. Correct every statement the feedback "
                 "flags as wrong or dubious; keep citations; remove hallucinations; "
                 "be concise and clinically safe."
             )},
            {"role": "user",
             "content": (
                 f"Question: {q}\n\n"
                 f"Previous answer (to be fixed):\n{prev_answer}\n\n"
                 f"\"\"\"External feedback – ONE FACT PER LINE (incorrect ⇢ correct):"
                 f"{external_feedback}\"\"\"\n"
+                "Rewrite the whole answer.  Remove or correct every wrong fact, "
+                "keep correct facts, keep citations.  Respond **only** with the new answer."
             )},
        ],
        temp=0.3,  # deterministic but allows slight re-wording
    )
    return revised



if __name__ == "__main__":
    # Ensure the script runs within an asyncio event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
        #print("\nExecution interrupted by user.")
    except Exception as e:
         #print(f"\nFATAL ERROR in main execution: {e}")
         import traceback
         traceback.print_exc()
         sys.exit(1)
