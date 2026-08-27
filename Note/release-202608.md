# DeepWiki August Release Notes

**Date:** 2026-08-27

**Branch:** `update/xixia/llmmodels`

**Period covered:** 2026-06-01 → 2026-08-27 (since [release-202605.md](./release-202605.md))

---

## Highlights

- **GPT-5.6 workload routing** — interactive chat uses `gpt-5.6-luna`, CodeTrace and wiki page generation use `gpt-5.6-terra`, and Deep Research plus wiki structure generation use `gpt-5.6-sol`.
- **Azure OpenAI v1 integration** — GPT-5.6 calls use the `/openai/v1/` data-plane endpoint with explicit reasoning and verbosity controls.
- **Organization package proxies** — web and processor container builds route npm, pip, and Poetry downloads through `packagefeedproxy.microsoft.io`.
- **Deterministic AML execution** — new schedules submit one immediate job, while `--run-now` provides explicit submission for existing schedules.
- **Existing VNet reuse** — private-network deployment can reuse an existing VNet without rewriting its managed subnet configuration.

---

## GPT-5.6 Model Routing

Azure OpenAI configuration now separates three text-generation tiers:

| Workload | Deployment | Request settings |
|----------|------------|------------------|
| Chat Q&A | `gpt-5.6-luna` | `reasoning_effort=low`, `verbosity=medium` |
| CodeTrace | `gpt-5.6-terra` | `reasoning_effort=medium`, `verbosity=medium` |
| Wiki pages | `gpt-5.6-terra` | `reasoning_effort=medium`, verbosity follows page importance |
| Wiki review | `gpt-5.6-terra` | `reasoning_effort=low`, verbosity follows page importance |
| Deep Research | `gpt-5.6-sol` | `reasoning_effort=xhigh`, `verbosity=medium` |
| Wiki structure | `gpt-5.6-sol` | `reasoning_effort=high`, `verbosity=medium` |

### Configuration and client changes

- Added `azure_openai.premium_reasoning` alongside the existing `chat`, `reasoning`, and `embedding` sections.
- Added `model_name` so capability detection does not depend on the Azure deployment alias.
- GPT-5.6 text deployments use `api_version: "v1"` and the `/openai/v1/` endpoint.
- Added per-deployment `reasoning_effort`, `verbosity`, and `max_completion_tokens` configuration.
- Removed `temperature` from reasoning-family requests while retaining it for compatible non-reasoning chat models.
- Removed the Qwen-specific `/no_think` prefix from chat and wiki prompts.
- Azure OpenAI clients are cached by endpoint, API version, and managed identity rather than sharing one client across potentially different endpoints.
- Updated the OpenAI Python SDK requirement and lock entry to `2.54.0`.

### Azure deployment templates

- Added separate ARM parameters and deployment resources for:
  - `gpt-5.6-luna`
  - `gpt-5.6-terra`
  - `gpt-5.6-sol`
- All three deployments use model version `2026-07-09`.
- Updated the frontend and backend fallback model to `gpt-5.6-luna`.

---

## Organization Package Registry Support

Container dependency installation now uses the configured package proxies:

| Package manager | Registry |
|-----------------|----------|
| npm | `https://packagefeedproxy.microsoft.io/npm/` |
| pip and Poetry | `https://packagefeedproxy.microsoft.io/pypi/simple/` |

- `Dockerfile` configures npm to replace lockfile registry hosts and installs packages through the npm proxy.
- `Dockerfile` and `Dockerfile.processor` install Poetry through the PyPI proxy.
- `poetry-plugin-pypi-mirror==0.5.0` routes Poetry dependency resolution through the same proxy without changing project package sources.
- `publish-web.ps1` passes both registry URLs as Docker build arguments.
- `DEEPWIKI_NPM_REGISTRY` and `DEEPWIKI_PYPI_REGISTRY` can override the defaults.
- A full no-cache web container build completed through the configured proxies.

---

## Azure ML Pipeline Scheduling

The AML dispatcher now separates immediate job submission from recurring schedule execution:

- A new schedule submits exactly one immediate pipeline job.
- An existing schedule submits immediately only when `--run-now` is specified.
- The recurring trigger starts at UTC now plus `azure_ml.schedule_interval_hours`.
- Schedule lookup treats only `ResourceNotFoundError` as a missing schedule; authentication and service errors propagate.
- Existing schedules are updated in place.
- A disabled schedule is re-enabled only after the replacement schedule is accepted.
- Immediate submission failures preserve the configured schedule and report that the operator can retry with `--run-now`.
- The cloud resource summary includes `aml_job` when an immediate job is submitted.

### Run an existing schedule now

```bash
python -m backend.processor.aml_dispatcher \
    --config=backend/run.json --run-now
```

The flag controls AML job submission only. It is separate from `code_processor --full-reprocess`, which controls incremental content processing inside a running job.

---

## Network Deployment Safety

- Added `manage_virtual_network` to control whether `NETWORK.Template.json` creates or updates the parent VNet and subnet definitions.
- `private_network` continues to control private DNS, private endpoints, App Service VNet integration, and related role assignments.
- The current environment sets `manage_virtual_network = False` so deployment reuses the existing VNet and subnets.
- Added `NETWORK.SubnetCompliance.Template.json` and its parameter file for explicit updates to existing private-endpoint and App Service subnets.
- Subnet templates set `defaultOutboundAccess` to `false`.
- The subnet compliance template sets the configured NSG associations and the `Microsoft.Web/serverFarms` delegation.

---

## Tests and Documentation

- Added model-routing tests covering workload selection, GPT-5.6 parameter compatibility, the three configured tiers, and ARM deployment defaults.
- Added package-registry tests covering the Dockerfile defaults and `publish-web.ps1` build arguments.
- Updated the root README model-routing, Azure OpenAI configuration, deployment, and package-proxy guidance.
- Updated chat module documentation for reasoning-model keepalive behavior.

---

## Files Changed (Highlights)

| File | Change |
|------|--------|
| `backend/model_routing.py` | Added workload selection, reasoning-model detection, and compatible request-parameter construction |
| `backend/config/infra.json` | Added Luna, Terra, and Sol configuration with Azure OpenAI v1 settings |
| `backend/config.py` | Added three-tier accessors, model-name lookup, and per-endpoint client caching |
| `backend/clients/azureai_client.py` | Added Azure OpenAI v1 sync and async client initialization |
| `backend/modules/chat/http_handler.py` | Routed Chat and Deep Research to their configured tiers |
| `backend/modules/chat/ws_handler.py` | Routed chat, wiki page, wiki structure, and Deep Research requests |
| `backend/modules/codetrace/service.py` | Routed CodeTrace to `gpt-5.6-terra` |
| `backend/processor/wiki_generator.py` | Split wiki structure and page generation between Sol and Terra |
| `Deployments/templates/AOAI.Template.json` | Added three GPT-5.6 model deployment resources |
| `Deployments/templates/NETWORK.Template.json` | Added optional parent-VNet management |
| `Deployments/templates/NETWORK.SubnetCompliance.Template.json` | Added explicit existing-subnet compliance deployment |
| `backend/processor/aml_dispatcher.py` | Added `--run-now` |
| `backend/processor/cloud_setup.py` | Added deterministic schedule timing and immediate submission |
| `Dockerfile` | Routed npm, pip, and Poetry through organization proxies |
| `Dockerfile.processor` | Routed processor Python dependencies through the organization proxy |
| `publish-web.ps1` | Passed configurable package registry build arguments |
| `pyproject.toml` / `poetry.lock` | Updated the OpenAI SDK requirement and lock entry |
| `tests/test_model_routing.py` | Added GPT-5.6 routing regression coverage |
| `tests/test_package_registry_config.py` | Added package-proxy regression coverage |
| `Note/release-202608.md` | Expanded August release notes |
