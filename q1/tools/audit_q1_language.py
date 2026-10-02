"""Independent frozen Whisper diagnostic; never replace official text or labels."""

import argparse, json, hashlib, importlib.metadata, time
from pathlib import Path
import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs")
    p.add_argument("--model", default="models/whisper_tiny")
    p.add_argument("--ids", default="")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    from faster_whisper import WhisperModel

    model = WhisperModel(
        a.model,
        device=a.device,
        compute_type="float16" if a.device == "cuda" else "int8",
    )
    root = Path(a.output)
    ids = set(a.ids.split(",")) if a.ids else None
    sha = lambda x: hashlib.sha256(Path(x).read_bytes()).hexdigest()
    provenance = {
        "tool": "faster-whisper",
        "version": importlib.metadata.version("faster-whisper"),
        "model": "Systran/faster-whisper-tiny",
        "files": {x.name: sha(x) for x in Path(a.model).glob("*") if x.is_file()},
        "purpose": "independent_language_and_transcript_diagnostic_only",
        "beam_size": 5,
        "temperature": 0,
        "vad_filter": False,
        "official_text_replaced": False,
    }
    records = []
    for folder in sorted(root.glob("sample_*")):
        if ids and folder.name not in ids:
            continue
        x, sr = sf.read(folder / "audio16k.wav", dtype="float32")
        m = json.loads((folder / "media.json").read_text())
        n = int(np.floor(max((b for _, b in m["audio_observed_intervals"])) * sr))
        x = x[:n]
        if not np.any(x):
            result = {
                "sample_key": folder.name,
                "status": "silent_audio_no_language_inference",
            }
        else:
            start = time.monotonic()
            segments, info = model.transcribe(
                x,
                beam_size=5,
                temperature=0,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            parts = [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in segments
            ]
            result = {
                "sample_key": folder.name,
                "status": "diagnostic_only",
                "language": info.language,
                "model_language_score": info.language_probability,
                "segments": parts,
                "text": " ".join((s["text"].strip() for s in parts)),
                "elapsed_s": time.monotonic() - start,
                "warning": "ASR errors and language errors remain possible; this is not human ground truth or official replacement text",
            }
        result["audio_sha256"] = sha(folder / "audio16k.wav")
        records.append(result)
        (folder / "language_audit.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result), flush=True)
    (root / "language_audit_summary.json").write_text(
        json.dumps({"provenance": provenance, "records": records}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
