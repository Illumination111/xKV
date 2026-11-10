# xKV BioCache proxy adapter

The package implements `reset`, `prefill_archive`, `query`,
`persistent_state`, `static_extra_state`, and `temporary_state`.

Persistent bytes enumerate every cross-layer `U`/`SV` factor plus serialized
layer-group metadata. Dense K/V reconstruction is query-local temporary state.
The adapter core imports neither NeMo nor CUTLASS; official backends are
provided through callbacks.

```bash
python -m biocache_adapt --dry-run --budget 25% \
  --output /tmp/xkv.jsonl --manifest /tmp/xkv-manifest.json
```

Budgets are actual-byte 12.5%, 25%, and 50% tiers. Needle and length-scan runs
share the `biocache-proxy-v1` JSONL schema and phase-separated timings.
