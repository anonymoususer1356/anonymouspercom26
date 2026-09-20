# Contributing

This is an anonymous research artifact. Open issues should include a minimal reproducer, environment details, and the exact configuration used. Do not upload CANDOR media or transcripts, model weights, credentials, private keys, or raw requests containing conversation text.

Before submitting changes, run:

```bash
python scripts/validate_dataset.py
python -m compileall -q src scripts
pytest -q
```
