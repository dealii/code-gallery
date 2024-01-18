---
title: "Scheduled GitHub Actions worker is failing"
---

The following projects failed to either configure, build or test: {{ env.error_compile }} {{ env.error_test }}

See [the action log](https://github.com/{{ env.GITHUB_REPOSITORY }}/actions/runs/{{ env.GITHUB_RUN_ID }}) for more details.
