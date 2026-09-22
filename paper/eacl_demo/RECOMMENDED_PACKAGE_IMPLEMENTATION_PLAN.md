# EACL 2027 Demo Installable Package Implementation Plan

## 1. Decision and release target

Use the **downloadable installable-package** route for the EACL 2027 System
Demonstrations submission. Do not make a hosted LLM service a submission
dependency. ScholAR's local model, document handling, and hardware requirements
make a versioned source package more faithful to the system being evaluated.

The three submission artifacts must point to the same frozen system:

1. the EACL Demo paper;
2. a screencast no longer than 2.5 minutes; and
3. a publicly downloadable, installable ScholAR package.

Planned release identity:

| Field | Value |
|---|---|
| Release name | `ScholAR EACL 2027 Demo v1.0.0` |
| Git tag | `eacl-demo-2027-v1.0.0` |
| Archive | `ScholAR_EACL2027_Demo_v1.0.0.zip` |
| Archive root | `ScholAR-EACL2027-Demo-v1.0.0/` |
| Planned release page | `https://github.com/prithvi-kaizen/ScholAR/releases/tag/eacl-demo-2027-v1.0.0` |
| License | MIT |
| Supported reference environment | macOS on Apple Silicon, Python 3.12, Node.js 20/npm 10, local Ollama |

The planned URL must remain labelled as planned until the release exists and
has been tested while signed out of GitHub.

Official requirement source:
`https://2027.eacl.org/calls/demos/`.

## 2. Current state and blockers

The repository already contains the application, exact Python and npm locks,
the quickstart script, an MIT license, tests, a deterministic anonymous
artifact packager, and a separate six-page Demo Track manuscript. These are a
strong base, but the current public-delivery path is not yet submission-ready.

| Area | Current state | Required correction |
|---|---|---|
| Public repository | Public, but behind the current working tree | Publish one reviewed, frozen release commit and tag |
| Downloadable package | Anonymous Industry artifact exists | Build a named, installable Demo package with a separate fail-closed policy |
| Installation | `scripts/quickstart.sh` and `docs/SETUP.md` exist | Add a reviewer-focused demo quickstart and verify it from a clean extraction |
| Demo input | Local/private papers are excluded | Add a self-authored redistributable sample PDF and source |
| Video | Script exists; URL is a placeholder | Record the frozen release, upload it, and replace every placeholder URL |
| Demo paper | Structurally valid | Correct unsupported latency, named-judge, visual-judge, human-study, and availability claims |
| Screenshot | Authentic workspace capture exists | Capture a final frame that visibly shows the citation highlight claimed by its caption |

## 3. Package scope

### 3.1 Include

The package should contain only what a reviewer needs to install, run, inspect,
and test the demonstration:

- `backend/` runtime source, schemas, and services;
- `frontend/` source plus `package.json` and `package-lock.json`;
- `requirements/locks/base-py312.txt` and the direct requirement files needed
  by the runtime;
- `scripts/quickstart.sh`, the environment doctor, explicit model setup, and
  any migration needed by the demo;
- the top-level `Makefile` targets required by the demo;
- `README.md`, `DEMO_INSTALL.md`, `docs/CODEBASE.md`, `docs/PIPELINE.md`, and
  the relevant portion of `docs/SETUP.md`;
- the MIT `LICENSE` and `THIRD_PARTY_NOTICES.md`;
- the self-authored demo PDF, its editable source, and a short list of verified
  questions;
- a minimal model-free smoke fixture for checking the UI/API boundary without
  downloading a model;
- tests required by the clean-install smoke path;
- `DEMO_PACKAGE_MANIFEST.json` containing the release version, build date,
  file hashes, required tools, and package limitations.

### 3.2 Exclude

The packager must reject, rather than silently rewrite, any of the following:

- `.git`, local branches, commit credentials, or repository tokens;
- `.env`, `.env.local`, API keys, cookies, private keys, or user-specific paths;
- `backend/data/`, uploaded PDFs, parsed text, page images, crops, SQLite files,
  embeddings, and indexes;
- model weights and model caches;
- private human-evaluation databases, returned ratings, consent records, or
  evaluator identifiers;
- private/raw evaluation results, prompts, completions, and copyrighted paper
  excerpts;
- build caches, `.venv`, `node_modules`, `.next`, logs, reports, and OS files;
- unsupported manuscripts or internal planning material not needed to operate
  the demo.

### 3.3 Licensing rule

The package may distribute only project-owned files, MIT-compatible project
source, and specifically reviewed example material. Dependency lockfiles may
name third-party packages, but installed dependencies and model weights should
not be bundled. Their acquisition remains an explicit user action under their
own licenses.

## 4. File-level implementation

### 4.1 Add a dedicated Demo release packager

Create `scripts/package_demo_release.py` rather than overloading
`scripts/package_supplementary.py`. The anonymous artifact and public demo
package have different identities, contents, and acceptance criteria.

The new packager must:

1. use an explicit allowlist;
2. reject symlinks and path traversal;
3. reject private directories, blocked suffixes, oversized files, secrets, and
   absolute home-directory paths;
4. copy files byte for byte without content rewriting;
5. assign deterministic archive ordering and timestamps;
6. generate a SHA-256 entry for every included file;
7. generate an archive SHA-256 sidecar;
8. embed the release version and source commit in the manifest;
9. support `--validate-only` and `--verify-archive` modes; and
10. fail when a required runtime, documentation, license, or sample file is
    missing.

Expected commands:

```bash
.venv/bin/python scripts/package_demo_release.py --validate-only
.venv/bin/python scripts/package_demo_release.py
.venv/bin/python scripts/package_demo_release.py \
  --verify-archive release/ScholAR_EACL2027_Demo_v1.0.0.zip
```

### 4.2 Add Makefile entry points

Add these targets:

```text
demo-setup            Install exact backend/frontend dependencies
demo-model            Explicitly acquire the configured local Ollama model
demo-run              Start backend and frontend with documented ports
demo-doctor           Check versions, ports, model availability, and disk/RAM
demo-smoke            Run model-free API/UI/package checks
demo-package          Build the deterministic public ZIP
demo-package-verify   Re-open and validate the generated ZIP
demo-release-check    Run every gate required before publishing
```

`demo-setup` must not silently download model weights, papers, datasets, or
Docling assets. `demo-model` is the explicit acquisition step.

### 4.3 Add reviewer-facing installation documentation

Create `DEMO_INSTALL.md` with a short path first and detailed troubleshooting
after it. The first screen should contain:

```bash
unzip ScholAR_EACL2027_Demo_v1.0.0.zip
cd ScholAR-EACL2027-Demo-v1.0.0
bash scripts/quickstart.sh
make demo-model
make demo-run
```

It must also state:

- exact tested operating system and tool versions;
- approximate model download and disk size after these are measured;
- why model acquisition is separate;
- local URLs for the application and API;
- how to upload the included sample PDF;
- the exact demo questions;
- expected behavior after clicking a citation;
- how to open the evidence graph and export a trace;
- how to shut the services down;
- common failures and recovery steps;
- which Linux/Windows configurations are tested versus best-effort.

Avoid claiming Docker or container support unless a container definition is
actually added and smoke-tested.

### 4.4 Add a redistributable example

Create `examples/eacl_demo/` containing:

- a short, project-authored scientific-style PDF;
- its editable source;
- one text section, one numbered quantitative table, one simple figure, and
  unambiguous page-local facts;
- `QUESTIONS.md` with one textual, one quantitative, and one visual question;
- `LICENSE.md` explicitly permitting redistribution with the demo package; and
- expected evidence page/region descriptions for smoke testing.

Do not copy a third-party research PDF merely because it is publicly
downloadable. The example should be authored for this release so its
redistribution status is unambiguous.

### 4.5 Add package tests

Create `tests/test_demo_release_package.py` covering:

- required runtime and documentation files are included;
- the regular MIT license, not `LICENSE-ANONYMOUS`, is included;
- private state and blocked binary/data formats are excluded;
- the demo manuscript's internal planning files are excluded;
- the self-authored example and its license are included;
- two builds from identical inputs have identical SHA-256 hashes;
- archive paths remain under the single expected root;
- the embedded manifest exactly matches archive contents;
- tampering with an archived file is detected;
- secrets and absolute user paths stop the build;
- a clean extracted package passes the model-free smoke test.

Extend the existing submission-integrity test so it rejects placeholder URLs
such as `#demo-video`, `TODO`, `TBD`, or `example.com` in the final paper and
submission metadata.

## 5. Clean-install demonstration path

The release is not ready merely because the ZIP builds. Test the reviewer path
from a new temporary directory or clean machine:

1. download the release while signed out;
2. verify the published SHA-256;
3. extract the ZIP;
4. run `demo-doctor` before installation;
5. run `demo-setup` using only the documented prerequisites;
6. acquire the named model through the explicit step;
7. run `demo-smoke`;
8. start the application;
9. upload the included example PDF;
10. ask each question from `QUESTIONS.md`;
11. click a citation and confirm the correct page and visible bounding box;
12. inspect a visual crop;
13. open the evidence graph;
14. export a trace and confirm that it contains no external secrets or local
    absolute paths; and
15. stop and restart the application to confirm persisted local state behaves
    as documented.

Record the actual commands, environment, duration, failures, and final result
in `paper/eacl_demo/ARTIFACT_SMOKE_TEST.md`. Do not retain old pass counts after
the test suite or package changes.

## 6. Freeze and publish procedure

### Phase A — Curate the release tree

- Inventory the current dirty worktree and preserve unrelated user changes.
- Select only the code and documentation used by the demo.
- Run formatting, type checks, targeted backend tests, and frontend production
  build.
- Remove stale statements from the public README, especially obsolete metrics,
  prior-paper status, and unsupported feature comparisons.
- Commit the reviewed state on a dedicated release branch.

Gate: `git diff --check` passes and the release commit contains no private or
generated runtime state.

### Phase B — Build and verify the ZIP

- Build the package twice and confirm identical hashes.
- Verify the embedded manifest.
- Scan for secrets, private paths, databases, PDFs other than the authorized
  self-authored sample, model files, and evaluator data.
- Extract into a new temporary directory and run `demo-smoke` there.

Gate: the deterministic archive and clean-extraction smoke test both pass.

### Phase C — Perform the full local-model smoke test

- Use the documented reference workstation.
- Run the exact reviewer installation path.
- Execute the three example questions.
- Capture the final UI screenshot used in the paper.
- Record only measured timing observations; do not generalize from a single
  interactive run.

Gate: the citation click, bounding-box highlight, visual crop, evidence graph,
and trace export shown in the paper/video work in the packaged release.

### Phase D — Publish the GitHub release

- Push the reviewed release commit.
- Create annotated tag `eacl-demo-2027-v1.0.0`.
- Create a GitHub Release with the ZIP, checksum file, concise requirements,
  MIT license statement, and known limitations.
- Test the release page and asset download in a signed-out browser.
- Record the final immutable release URL and archive SHA-256.

Gate: an unauthenticated reviewer can download the exact package named in the
paper.

## 7. Paper and video synchronization

After the release URL is live, update the Demo paper—not the Industry paper.

### Paper changes

- Put a visible resource box on page 1 containing the package URL, video URL,
  MIT license, and minimum tested environment.
- Replace the repository-only link with the immutable release page.
- Describe the explicit model-acquisition step and distinguish installation
  from strict-local analysis.
- Replace unsupported latency claims with the audited timing distribution.
- Report the 150-case review only as an LLM-based evaluation; do not name a
  judge model that retained metadata does not verify.
- Exclude the out-of-schema visual-judgment distribution.
- State the human-study freeze status exactly: 0 of 250 ratings submitted.
- Replace the UI screenshot with the frame captured from the released package.
- Ensure the screenshot caption describes only visible UI elements.

### Video changes

- Record the tagged release using the included example and questions.
- Keep the video at or below 150 seconds.
- Do not say “under seven seconds” or show unsupported metrics.
- Show the release URL and version in the final frame.
- Upload the video as unlisted YouTube or a publicly accessible release asset.
- Test the video link while signed out.
- Put the identical URL in the PDF and OpenReview form.

Gate: paper, video, and package show the same interface, release version,
features, and limitations.

## 8. Release validation matrix

| Gate | Command or check | Pass condition |
|---|---|---|
| Python syntax/lint | existing repository checks | No errors |
| Backend tests | targeted runtime and API test set | All pass in strict-local mode |
| Frontend types/build | `make frontend-build` | Production build succeeds |
| Package policy | `make demo-package-verify` | No blocked files or manifest errors |
| Determinism | build twice and hash | SHA-256 values match |
| Clean extraction | `make demo-smoke` inside extracted ZIP | All model-free checks pass |
| Full workflow | manual checklist with local model | Ingest, answer, citation, crop, graph, export work |
| Licensing | file inventory and notices | Every distributed file has a valid basis |
| Availability | signed-out browser test | Release ZIP downloads without authentication |
| Video | duration and signed-out playback | Plays and is `<= 2:30` |
| PDF | compile plus Demo validator | Six content pages or fewer; no missing links |
| Link consistency | PDF, metadata, OpenReview checklist | Exact package and video URLs match |
| Originality policy | author confirmation | No conflicting concurrent archival submission |
| Reviewer duty | OpenReview metadata | Reciprocal reviewer nominated |

## 9. Stop-ship conditions

Do not publish or cite the package if any of these remains true:

- the package URL is a repository branch rather than a frozen release;
- the ZIP cannot be installed from a clean extraction;
- any PDF, database, model cache, raw completion, evaluator record, secret, or
  personal path is present unintentionally;
- the paper or video names a feature that does not work in the released build;
- either mandatory link is a placeholder, private, or inaccessible while
  signed out;
- paper, video, package, and OpenReview metadata use different versions;
- the screenshot is not from the released application;
- the software license or example-document redistribution basis is unclear;
- the work conflicts with the venue's multiple-submission policy.

## 10. Definition of done

The recommended package is complete only when all of the following exist:

- `release/ScholAR_EACL2027_Demo_v1.0.0.zip`;
- its published SHA-256 checksum;
- an exact embedded file manifest;
- a clean-install smoke-test record;
- an accessible GitHub Release page and downloadable asset;
- an MIT license and third-party notices;
- a self-authored example PDF and questions;
- a final paper containing both working links;
- a public or unlisted video no longer than 2.5 minutes containing the same
  release workflow; and
- matching OpenReview metadata with the reciprocal reviewer identified.

