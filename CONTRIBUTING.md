# Contributing to Aevom

Contributions are accepted under [Apache-2.0](LICENSE). Preserve the original
licenses and attribution of third-party code.

## Submitting changes

1. Create a branch for your change.
2. Make the change and run the relevant checks below.
3. Sign off every commit as described in [Signing Your Work](#signing-your-work).
4. Submit a pull request or merge request against `main`, describing the change
   and how you verified it. Address feedback from the maintainers before merge.

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies
that the contribution is your original work, or you have rights to submit it
under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be
accepted. We use the Developer Certificate of Origin (DCO); a Contributor
License Agreement (CLA) is not required.

To sign off on a commit, use the `--signoff` (or `-s`) option:

```bash
git commit -s -m "Describe your change"
```

This appends a line to the commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

Use your own name and email address. DCO sign-off is a commit-message statement
and does not require a cryptographic commit signature.

## Full text of the DCO

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Full text: https://developercertificate.org/

## Development workflow

NVIDIA contributors use merge requests against `main` in the internal GitLab project.
Every push runs the CI pipeline; a push to `main` also builds images and deploys
to the internal cluster.

Before opening a merge request:

```bash
make check       # prerequisites: docker, keys, disk
make test-fast   # the backend suite, ~30s, no containers
make test        # everything, including the containerized suites
```

`make test-fast` covers roughly 780 of the ~800 tests and is the loop to use
while developing. Four suites need Docker because they depend on binaries the
devcontainer deliberately lacks: `make test-rdkit`, `make test-vina`,
`make test-pyexec` and `make test-training`. Run the full `make test` before
pushing.

If you touched the deployment, `helm lint` and `helm template` the chart, or just
let the `chart-lint` CI job do it. If you touched agent behaviour, prompts or
tools, run `make eval-smoke` — the unit suite cannot see a regression in what the
model is able to chain together. See [evals/README.md](evals/README.md).

## Things worth knowing

- Never `pip install` into the devcontainer. A missing dependency means the image
  is stale; rebuild it. New dependencies go in `backend/requirements.txt` or
  `requirements-dev.txt`.
- The eval gate thresholds are calibrated against a specific model. Changing
  `AEVOM_MODEL` for production or the eval stack means re-baselining them.
- Network failures against NVIDIA-internal hosts usually mean the VPN dropped.
