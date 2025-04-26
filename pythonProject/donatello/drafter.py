from __future__ import annotations

"""
Drug‑information RAG agent with status logging & configurable specialist count
==============================================================================
Stages (printed to stdout):
  • [STATUS] Knowledge Retrieval – after pulling candidate chunks from VDB
  • [STATUS] Reasoning            – while the assistant drafts / revises an answer
  • [STATUS] Expertise            – when sampled specialists are satisfied
If *any* sampled specialist requests changes, status rolls back to **Reasoning**.

Run example (default data dir "structured_drug_data" must have *.xml files):
    python drug_info_agent.py --num-specialists 2 \
        "What are the interactions between prednisone and carvedilol?"
"""

import os, re, sys, json, random, xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TypedDict

import numpy as np
import openai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("ERROR: OPENAI_API_KEY not set (env or .env)")

STRICT_MODEL      = "gpt-4o-mini"
FALLBACK_MODEL    = "gpt-4o-mini"
SPECIALIST_MODEL  = "gpt-4o-mini"
EMBED_MODEL       = "text-embedding-3-small"
MAX_CHARS_PER_CHUNK = 2_000
MAX_REVISIONS       = 2
NUM_SPECIALISTS_TO_USE = 1  # default, can be overridden by CLI

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
    print(f"[STATUS] {stage}")

# ---- XML parsing → graph ---------------------------------------------------

def build_graph(xml_path: Path | str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"[build_graph] skip {xml_path}: {e}")
        return out

    ns = {"hl7": "urn:hl7-org:v3"}
    root = Path(xml_path).stem
    for n, sec in enumerate(tree.findall(".//hl7:section", ns)):
        label = ""; node_id = sec.attrib.get("ID", f"sec{n}")
        code = sec.find("hl7:code", ns)
        if code is not None and code.attrib.get("displayName"):
            label = code.attrib["displayName"].upper()
        else:
            ttl = sec.find("hl7:title", ns)
            if ttl is not None and ttl.text:
                label = ttl.text.upper()
            else:
                label = node_id.upper()
        norm = re.sub(r"\s*SECTION$", "", label).strip()
        if not any(norm == w or norm in w or w in norm for w in SECTION_WHITELIST):
            continue
        unique_id = f"{root}__{node_id}"
        parts = [t.strip() for elem in sec.iter() for t in (elem.text, elem.tail) if t and t.strip()]
        text = re.sub(r"\s+", " ", " ".join(parts))
        if text:
            out[unique_id] = text
    return out

# ---- chunking --------------------------------------------------------------

def make_chunks(graph: Dict[str, str]) -> List[Tuple[str, str]]:
    chunks: List[Tuple[str, str]] = []
    for sid, txt in graph.items():
        if len(txt) <= MAX_CHARS_PER_CHUNK:
            chunks.append((f"{sid}#0", txt))
        else:
            step = MAX_CHARS_PER_CHUNK - MAX_CHARS_PER_CHUNK // 5
            for i, st in enumerate(range(0, len(txt), step)):
                ck = txt[st:st+MAX_CHARS_PER_CHUNK].strip()
                if ck:
                    chunks.append((f"{sid}#chunk{i}", ck))
    return chunks

# ---- vector DB -------------------------------------------------------------
@dataclass
class VDB:
    ids: List[str] = field(default_factory=list)
    txts: List[str] = field(default_factory=list)
    vecs: List[np.ndarray] = field(default_factory=list)

    def add(self, pairs: List[Tuple[str, str]]):
        if not pairs:
            return
        pid, ptxt = zip(*pairs)
        self.ids.extend(pid); self.txts.extend(ptxt)
        embs = openai.embeddings.create(model=EMBED_MODEL, input=list(ptxt)).data
        self.vecs.extend([np.array(e.embedding, dtype=np.float32) for e in embs])

    def search(self, q: str, k: int = 8) -> List[Tuple[str, str]]:
        if not self.vecs:
            return []
        qv = np.array(openai.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding, dtype=np.float32)
        mat = np.stack(self.vecs)
        sims = mat @ qv / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv))
        idx = sims.argsort()[-k:][::-1]
        return [(self.ids[i], self.txts[i]) for i in idx]

# ---- specialists -----------------------------------------------------------
class SpecialistFeedback(TypedDict):
    status: str; feedback: str | None; specialist_name: str

class Specialist:
    def __init__(self, name: str, prompt: str):
        self.name = name; self.prompt = prompt

    def review(self, q: str, ex: str, ans: str) -> SpecialistFeedback:
        rsp = openai.chat.completions.create(
            model=SPECIALIST_MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Q:\n{q}\n\nExcerpts:\n{ex or 'None'}\n\nAnswer:\n{ans}"},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        ).choices[0].message.content.strip()
        try:
            data = json.loads(rsp)
            if data["status"] == "Satisfied":
                data["feedback"] = None
            return {"status": data["status"], "feedback": data.get("feedback"), "specialist_name": self.name}
        except Exception:
            return {"status": "Needs Revision", "feedback": f"{self.name}: bad JSON", "specialist_name": self.name}

JSON_HEADER = "Keep answers short. Output **ONLY** JSON with keys 'status' & 'feedback'."
INTERACTION_PROMPT = "You are a Drug Interaction Specialist. " + JSON_HEADER
PHARM_PROMPT       = "You are a Pharmacology Specialist. "     + JSON_HEADER
CLIN_PROMPT        = "You are a Clinical Diagnosis Specialist. " + JSON_HEADER

# ---------------------------------------------------------------------------
# 3. Main agent class
# ---------------------------------------------------------------------------
class DrugInfoAgent:
    def __init__(self, data_dir: Path):
        self.vdb = VDB(); self._load_dir(data_dir)
        self.specialists = [
            Specialist("Interaction", INTERACTION_PROMPT),
            Specialist("Pharmacology", PHARM_PROMPT),
            Specialist("Clinical", CLIN_PROMPT),
        ]

    def _load_dir(self, d: Path):
        xmls = list(d.glob("*.xml"))
        for f in xmls:
            self.vdb.add(make_chunks(build_graph(f)))
        print(f"[ingest] {len(xmls)} XMLs → {len(self.vdb.ids)} chunks")

    def _chat(self, model: str, msgs: list, temp: float = 0.0) -> str:
        return openai.chat.completions.create(model=model, messages=msgs, temperature=temp).choices[0].message.content.strip()

    def answer(self, q: str) -> str:
        # --- 1) Retrieval
        log_status("Knowledge Retrieval")
        ctx = self.vdb.search(q, 8)
        excerpts = "\n\n".join(f"[{cid}] {txt}" for cid, txt in ctx)

        # --- 2) Strict answer or fallback
        log_status("Reasoning")
        strict = self._chat(STRICT_MODEL, [
            {"role": "system", "content": "You are a Drug Label Expert. Use ONLY excerpts. Cite ids. If fact missing reply: I don't know based on the provided drug label excerpts."},
            {"role": "user", "content": f"Excerpts:\n{excerpts or 'None'}\n\nQuestion: {q}"},
        ])
        ans = strict if not strict.lower().startswith("i don't know") else self._chat(
            FALLBACK_MODEL,
            [
                {"role": "system", "content": "You are a clinical assistant. Use excerpts + general knowledge. Cite ids when using excerpts."},
                {"role": "system", "content": f"Excerpts:\n{excerpts or 'None'}"},
                {"role": "user", "content": q},
            ],
            temp=0.4,
        )

        # --- 3) Review cycles
        log_status("Expertise")
        for _ in range(MAX_REVISIONS):
            sampled = random.sample(self.specialists, min(NUM_SPECIALISTS_TO_USE, len(self.specialists)))
            all_ok = True; feedback_lines: List[str] = []
            for sp in sampled:
                fb = sp.review(q, excerpts, ans)
                if fb["status"] == "Needs Revision":
                    all_ok = False
                    if fb["feedback"]:
                        feedback_lines.append(f"- {sp.specialist_name}: {fb['feedback']}")
            if all_ok:
                log_status("Done")
                return ans
            # roll back to reasoning with feedback
            log_status("Reasoning")
            ans = self._chat(FALLBACK_MODEL, [
                {"role": "system", "content": "Revise answer based on expert feedback."},
                {"role": "user", "content": (
                    f"Question: {q}\n\nExcerpts:\n{excerpts or 'None'}\n\nPrevious Answer:\n{ans}\n\nFeedback:\n" + ("\n".join(feedback_lines) or "(no details)") )},
            ], temp=0.3)

        # exceeded attempts
        log_status("Done")
        return ans

# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    log_status("Started")
    p = argparse.ArgumentParser(description="Drug‑info agent with specialist review")
    p.add_argument("query", nargs="+", help="User question")
    p.add_argument("--data-dir", default="structured_drug_data", help="Directory with SPL XML files")
    p.add_argument("--num-specialists", type=int, default=NUM_SPECIALISTS_TO_USE, help="Reviewers sampled per round")
    args = p.parse_args()

    NUM_SPECIALISTS_TO_USE = max(1, args.num_specialists)
    bot = DrugInfoAgent(Path(args.data_dir))
    print(bot.answer(" ".join(args.query)))
