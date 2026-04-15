"""
fix_colorado_sections.py — Patch Colorado chunk metadata to use proper CRS section numbers.

Problem: chunks were stored with bill section labels ("SECTION 1", "SECTION 6") instead of
Colorado Revised Statutes section numbers ("§ 6-1-1701", "§ 6-1-1703", etc.).
This caused citation recall to be 0% for all Colorado questions because the validator
matched "§ 6" against expected "§ 6-1-1703" and found no match.

Fix:
  1. Determine the primary CRS section for each Colorado chunk from its text.
  2. Update the section field in data/chunks/colorado_ai_act_chunks.json.
  3. Push a metadata-only update to Pinecone (no re-embedding needed).

Usage:
    python src/fix_colorado_sections.py
    python src/fix_colorado_sections.py --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks" / "colorado_ai_act_chunks.json"

# Explicit overrides for chunks where automatic detection is ambiguous.
# Key = chunk_id, Value = CRS section label to assign.
EXPLICIT_MAP = {
    "colorado_ai_act_chunk_001":   None,          # Preamble — no CRS section
    "colorado_ai_act_chunk_002_a": "§ 6-1-1701",  # Definitions header
    "colorado_ai_act_chunk_002_b": "§ 6-1-1701",  # Definitions continuation
    "colorado_ai_act_chunk_002_c": "§ 6-1-1703",  # Deployer exceptions
    "colorado_ai_act_chunk_002_d": "§ 6-1-1702",  # Applicability/scope
    "colorado_ai_act_chunk_003_a": "§ 6-1-1703",  # Developer obligations (care standard)
    "colorado_ai_act_chunk_003_b": "§ 6-1-1703",  # Developer obligations (disclosures)
    "colorado_ai_act_chunk_003_c": "§ 6-1-1703",  # Developer public availability
    "colorado_ai_act_chunk_003_d": "§ 6-1-1703",  # Developer discrimination report
    "colorado_ai_act_chunk_003_e": "§ 6-1-1706",  # Consumer protections / rebuttable presumption
    "colorado_ai_act_chunk_003_f": "§ 6-1-1703",  # Developer risk mgmt framework
    "colorado_ai_act_chunk_003_g": "§ 6-1-1703",  # Deployer deployment assessment
    "colorado_ai_act_chunk_003_h": "§ 6-1-1703",  # Developer intended use consistency
    "colorado_ai_act_chunk_003_i": "§ 6-1-1706",  # Consumer disclosure statement
    "colorado_ai_act_chunk_003_j": "§ 6-1-1706",  # Consumer appeal rights
    "colorado_ai_act_chunk_003_k": "§ 6-1-1702",  # Impact assessment contents
    "colorado_ai_act_chunk_003_l": "§ 6-1-1704",  # Developer impact assessments
    "colorado_ai_act_chunk_003_m": "§ 6-1-1705",  # Developer/deployer annual assessment
    "colorado_ai_act_chunk_003_n": "§ 6-1-1702",  # Exemptions (research/statistical)
    "colorado_ai_act_chunk_003_o": "§ 6-1-1703",  # Developer substantial modification
    "colorado_ai_act_chunk_003_p": "§ 6-1-1702",  # Federal act exemptions (healthcare)
    "colorado_ai_act_chunk_004_a": "§ 6-1-1702",  # Insurance exemption (10-3-1104.9)
    "colorado_ai_act_chunk_005_a": "§ 6-1-1706",  # Banking / financial definitions
    "colorado_ai_act_chunk_005_b": "§ 6-1-1707",  # Attorney general enforcement
    "colorado_ai_act_chunk_006":   "§ 6-1-1703",  # Risk mgmt framework recognition
    "colorado_ai_act_chunk_007":   None,           # Safety clause — no CRS section
}


def _infer_crs_section(text: str) -> str | None:
    """
    Fall-back: find the first CRS section reference (6-1-170X) in the chunk text.
    Returns "§ 6-1-170X" or None.
    """
    m = re.search(r"6-1-170(\d)", text)
    if m:
        return f"§ 6-1-170{m.group(1)}"
    return None


def patch_chunks(dry_run: bool = False):
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    updated = []
    changes = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        old_section = chunk["section"]

        if chunk_id in EXPLICIT_MAP:
            new_section = EXPLICIT_MAP[chunk_id]
        else:
            new_section = _infer_crs_section(chunk["text"])

        if new_section is None:
            new_section = old_section  # keep original (Preamble, safety clause)

        if new_section != old_section:
            changes.append((chunk_id, old_section, new_section))

        updated_chunk = dict(chunk)
        updated_chunk["section"] = new_section
        updated.append(updated_chunk)

    print(f"Colorado chunks: {len(chunks)} total, {len(changes)} section updates")
    for chunk_id, old, new in changes:
        print(f"  {chunk_id}: '{old}' → '{new}'")

    if dry_run:
        print("\n[dry-run] No files or Pinecone records changed.")
        return

    # 1. Write updated chunk JSON
    with open(CHUNKS_PATH, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"\nUpdated {CHUNKS_PATH}")

    # 2. Push metadata-only updates to Pinecone
    _update_pinecone(updated, changes)


def _update_pinecone(chunks: list[dict], changes: list[tuple]):
    """Update only the changed chunks in Pinecone (metadata-only, no re-embedding)."""
    from pinecone import Pinecone

    api_key    = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "regugrounded").strip()

    if not api_key:
        print("ERROR: PINECONE_API_KEY not set — skipping Pinecone update.")
        return

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    changed_ids = {c[0] for c in changes}
    chunk_map   = {c["chunk_id"]: c for c in chunks}

    updated_count = 0
    for chunk_id in changed_ids:
        chunk = chunk_map[chunk_id]
        index.update(
            id=chunk_id,
            set_metadata={"section": chunk["section"]},
        )
        updated_count += 1

    print(f"Pinecone: updated metadata for {updated_count} vectors in '{index_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Colorado CRS section metadata")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    args = parser.parse_args()
    patch_chunks(dry_run=args.dry_run)
