"""Validate all 100 final candidates before promoting; preserve prior alignments."""

import argparse, json
from pathlib import Path
from datetime import datetime, timezone
from .alignment_store import (
    load_manifest,
    local_record,
    alignment_document,
    digest_bytes,
    json_bytes,
    atomic_bytes,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    a = p.parse_args()
    rows = load_manifest(a.manifest)
    assert len(rows) == 100
    summary = json.loads(a.summary.read_text(encoding="utf-8-sig"))
    assert (
        summary["samples"] == 100
        and summary["status"] == "candidates_complete_not_promoted"
    )
    records = {r["sample_key"]: r for r in summary["records"]}
    assert len(records) == len(summary["records"]) == 100 and set(records) == {
        r["sample_key"] for r in rows
    }
    assert summary["source_hashes"]["manifest_sha256"] == digest_bytes(
        a.manifest.read_bytes()
    ), "Summary belongs to a different manifest"
    plans = []
    for row in rows:
        folder = a.output / row["sample_key"]
        source = folder / "alignment.final.json"
        payload = source.read_bytes()
        doc = alignment_document(payload, row, local_record(a.manifest, a.output, row))
        assert (
            doc.get("finalization_fingerprint")
            and doc["finalization"]["automatic_rescued_words"] == 0
        )
        record = records[row["sample_key"]]
        assert (
            record["sample_id"] == doc["sample_id"]
            and record["status"] == doc["status"]
        )
        assert (
            record["official_words"] == doc["word_count"]
            and record["retained_word_times"] == doc["aligned_word_count"]
        )
        assert (
            record["segment_gate"] == doc["segment_gate"]["state"]
            and record["masked"] == doc["segment_gate"]["mask_word_alignment"]
        )
        assert all(
            (record[k] == v for k, v in doc["timing_agreement"].items())
        ), "Summary timing statistics differ from candidate"
        assert doc["finalization"]["policy"] == summary["policy"]
        assert all(
            (
                doc["finalization"]["source_hashes"][k] == v
                for k, v in summary["source_hashes"].items()
            )
        ), "Summary audit inputs differ from candidate"
        target = folder / "alignment.json"
        prior = target.read_bytes()
        backup = folder / ("alignment.before_final." + digest_bytes(prior) + ".json")
        if backup.exists():
            assert backup.read_bytes() == prior
        doc["finalization"]["formal_promotion_performed"] = True
        updated = json_bytes(doc)
        plans.append((target, backup, prior, updated, digest_bytes(payload)))
    for target, backup, prior, updated, candidate_hash in plans:
        if not backup.exists():
            atomic_bytes(backup, prior)
    receipt = []
    for target, backup, prior, updated, candidate_hash in plans:
        atomic_bytes(target, updated)
        receipt.append(
            {
                "sample_key": target.parent.name,
                "candidate_sha256": candidate_hash,
                "previous_alignment_sha256": digest_bytes(prior),
                "promoted_alignment_sha256": digest_bytes(updated),
                "backup_name": backup.name,
            }
        )
    summary.update(
        status="promoted_formal_all_100",
        formal_promotion_performed=True,
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        promotion_records=receipt,
    )
    atomic_bytes(a.output / "final_alignment_summary.json", json_bytes(summary))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "samples": len(plans),
                "totals": summary["totals"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
