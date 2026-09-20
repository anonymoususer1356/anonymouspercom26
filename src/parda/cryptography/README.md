# Voice-track cryptography

This layer runs after source separation and SLM anonymisation. It takes one source-output directory,
uses its LS-EEND `diarization.json` to select active speaker regions from each
`tracks/*.mkv`, computes a separate CAM++ voice fingerprint, and seals each
track for a trusted third party (TTP).

Voice fingerprints are encrypted in transit. The TTP decrypts them only in its
trusted environment to compare against registered voice profiles. The audio
itself never leaves this layer in plaintext once the original source tracks are
removed.

## Security model

- A fresh random AES-256 key encrypts every track with AES-256-GCM.
- Track ID and source file name are authenticated as GCM associated data.
- Each AES key is wrapped under the TTP's RSA-4096 public key using
  RSA-OAEP/SHA-256.
- Each CAM++ fingerprint has a separate random AES-256-GCM key, which is also
  RSA-OAEP-wrapped for the TTP.
- The repository contains no TTP private key. Keep it on TTP-controlled storage.
- A TTP must explicitly name each approved track when issuing a consent release.
- The wearer restores only the tracks named in that release; restoration verifies
  both the GCM authentication tag and the original SHA-256.

This is a key-release implementation. A production deployment still needs an
authenticated consent service and a signed approval record before the TTP runs
`release_approved_keys.py`.

## Output layout

```text
Outputs/Output Bundle/<run>/
  server/
    sealed_track_keys.json              # AES track keys wrapped for the TTP
    encrypted_voice_fingerprints.json   # encrypted CAM++ fingerprints
  device/
    encrypted_tracks/SPEAKER_00.mkv.aesgcm
    anonymised_word_timestamps.jsonl   # anonymized words plus word timing
```

The device package never includes the raw ASR transcript, raw word timing log,
or endpoint-debug log. `turn_endpointing.jsonl` is only a local diagnostic.
The orchestrator aligns the SLM output against Nemotron's token timestamps and
sends only `anonymised_word_timestamps.jsonl`: unchanged words retain their
original timing, dropped words disappear, and replacement words inherit the
replaced span.

The CAM++ trace decodes each MKV to temporary 16 kHz mono audio, extracts only
the matching LS-EEND active intervals from `diarization.json`, and embeds the
joined speech. It does not independently estimate activity from RMS. The trace
and fingerprint are AES-GCM encrypted for the TTP; no plaintext fingerprint is
included in either transfer package.
Its voice fingerprint is a normalized 192-dimensional CAM++ embedding.

## Commands

Use the Python environment that contains the source-separation dependencies.

```bash
cd /path/to/anonymouspercom26
"Temp/UI/.venv/bin/python" -m pip install -r "Scripts/Cryptography/requirements.txt"

"Temp/UI/.venv/bin/python" "Scripts/Cryptography/generate_ttp_keypair.py" \
  --env-file .env

"Temp/UI/.venv/bin/python" "Scripts/Cryptography/protect_tracks.py" \
  --source-output "Outputs/Orchestrator/local_60c30959"
```

For normal runs, this is invoked automatically at the end of
`Scripts/orchestrator.py`. It writes `Outputs/Output Bundle/<run>/`; pass
`--skip-protection` only for debugging.

After verified consent, the TTP runs:

```bash
"Temp/UI/.venv/bin/python" "Scripts/Cryptography/release_approved_keys.py" \
  --sealed-manifest "Outputs/Output Bundle/local_60c30959/server/sealed_track_keys.json" \
  --track SPEAKER_00 \
  --consent-reference "consent-request-123" \
  --output "consent_release.json"
```

Before matching, the TTP decrypts its own fingerprint package locally:

```bash
"Temp/UI/.venv/bin/python" "Scripts/Cryptography/ttp_decrypt_fingerprints.py" \
  --package "Outputs/Output Bundle/local_60c30959/server/encrypted_voice_fingerprints.json" \
  --output "/secure/ttp/decrypted_fingerprints.json"
```

The wearer restores only the released track(s):

```bash
"Temp/UI/.venv/bin/python" "Scripts/Cryptography/restore_tracks.py" \
  --sealed-manifest "Outputs/Output Bundle/local_60c30959/server/sealed_track_keys.json" \
  --consent-release "consent_release.json" \
  --output "Outputs/Cryptography/restored"
```

## Validation

This uses the first real separated track, generates an ephemeral RSA-4096 key,
runs protection, releases one key, restores the track, and checks byte-exact
SHA-256 equality. Its private key and all artifacts stay in the chosen temporary
validation directory.

```bash
"Temp/UI/.venv/bin/python" "Scripts/Cryptography/validate.py" \
  --source-output "Outputs/Orchestrator/local_60c30959" \
  --campplus-model "Models/Deployment Optimised/campplus/campplus_cn_en_common_200k_ort_optimized.onnx" \
  --output "Temp/cryptography-validation"
```

The source output must contain `anonymised_timestamped_lines.jsonl`, which the
current orchestrator creates. Older source-pipeline-only runs do not have a
privacy-safe timing artifact and should not be packaged for a device.
