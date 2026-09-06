# ScholAR EACL 2027 Industry Track paper

This is the new anonymous paper source. It does not copy result prose or tables from the venue-neutral or quarantined manuscripts. System descriptions map to current code through `claim_map.json`; empirical claims remain `PENDING` until a validated `eacl_industry_v1` release exists.

The [official EACL 2027 Industry Track call](https://2027.eacl.org/calls/industry/) requires the official unmodified template, at most six review content pages, double-blind materials, a dedicated Limitations section before references, and appendices after the bibliography in the same PDF. General details such as A4 paper, embedded fonts, two columns, review rulers, and a 200-word abstract come from [official ACL formatting guidance](https://acl-org.github.io/ACLPUB/formatting.html). The Industry call is authoritative if requirements differ.

Commands:

```bash
make verify       # safe draft/source/provenance checks
make draft        # requires official style files; may show pending-gate box
make submission   # fails until all real evidence and venue gates clear
```

Install the official style exactly as documented in `style/README.md`; do not modify it. `make submission` intentionally fails today because the human, ethics, held-out evaluation, model-run, hardware, and exact-template review gates are pending.

Content was rephrased for compliance with licensing restrictions.
