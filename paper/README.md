# ScholAR paper source

`eacl_industry/` is the only manuscript source retained in this repository. It is an
anonymous EACL Industry Track draft with fail-closed evidence gates: empirical tables
remain unavailable until a validated release exists.

```bash
make -C paper verify
make -C paper draft
make -C paper submission
```

`verify` checks source/provenance without requiring a complete empirical release.
`submission` intentionally fails while required data, human, ethics, model, hardware,
or venue gates remain open. See `paper/eacl_industry/README.md` and `claim_map.json`.

Do not add generated LaTeX intermediates or another historical manuscript tree.
