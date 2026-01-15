# Orcas CodeWiki (For MS Internal use)

> **Inspired by [DeepWiki-Open](https://github.com/AsyncFuncAI/deepwiki-open)** - This is a fork optimized exclusively for **Azure OpenAI** with Managed Identity authentication. 

**Orcas CodeWiki** automatically creates  interactive wikis for Azure DevOps repository! Just enter a repo url, and CodeWiki will:

1. Analyze the code structure
2. Generate comprehensive documentation
3. Create visual diagrams to explain how everything works
4. Organize it all into an easy-to-navigate wiki

## 🚀 Quick Start (if you want to deloy your own service)

### Prerequisites

#### 1. Development Env

- **Python 3.10+**
- **Node.js 18+**
- **Docker**

#### 2. Azure Resources Deployment

##### Option A: Automatic deployment leverage ARM (Recommended)

1. locate folder "Deployments"
1. Modify "config.py" to fill in your preferred resource name
1. Run "deploy_required.ipynb" for each required resource


##### Option B Manual deployment (Depercated)
<details>

Below are required resources you will need to manually created in Azure.
- **Azure OpenAI Service** with deployed models:
  - Text generation model (e.g., `o4-mini`, `gpt-4o`)
  - Embedding model (e.g., `text-embedding-3-large`)
- **Azure Storage Blob container**
- **Managed Identity (MSI)** 
  - configured with access **Cognitive Services OpenAI User** to Azure OpenAI
  - configured with access **Storage Blob Data Contributor** to Azure blob
  - configured with access **Monitoring Metrics Publisher** to Azure Application Insight
- **Application Insight** if you would like to emit logs to Azure 

</details>

#### 3. Security Requirements [**IMPORTANT**]

1. Public access should be disabled for below resources:
  - blob storage
  - keyvault, if any

  Instead, create a NSP (Network Security Perimeter) assosciate to above resources those who disable public access. And add below rules in NSP:
  - inbound
    - allow your subscriptions
    - allow service tag "MicrosoftPublicIPSpace"
  - outbound
    - allow * FQDNs

2. For Web App, 
- create identity provider following: [Quickstart: Add app authentication to your web app running on Azure App Service](https://learn.microsoft.com/en-us/azure/app-service/scenario-secure-app-authentication-app-service?tabs=workforce-configuration)
- Create Network Access rule to allow CorpNet using service tag "CorpNetPublic"

>Note: <br>
> Some of above security values are ONLY available in production tenant if deploying in Portal.
> However, you may want to try using ARM to pass in those values using API, which should be accepted.

### Configure infra.json


Once you have Azure resources ready, edit `backend/config/infra.json` with your Azure details:

```json
{
  "managed_identity": {
    "name": "your-msi-name",
    "client_id": "your-msi-client-id"
  },
  "azure_openai": {
    "endpoint": "https://your-resource.openai.azure.com",
    "api_version": "2024-12-01-preview",
    "deployment": "o4-mini"
  },
  "azure_openai_embedding": {
    "endpoint": "https://your-resource.openai.azure.com",
    "api_version": "2024-12-01-preview",
    "deployment": "text-embedding-3-large"
  },
  "azure_blob_storage": {
    "enabled": true,
    "account_name": "your-storage-account",
    "container_name": "deepwiki-data"
  },
  "azure_application_insights": {
    "enabled": true,
    "name": "your-app-insights",
    "connection_string": ""
  }
}
```

> Note <br>
> Keep `azure_application_insights` and `azure_blob_storage` disabled if testing local.
> These would be auto enabled when deployed to Azure cloud.

Then you have two options to continue setup in LOCAL.

### 🐳 Option A: Docker test in local environment

Test the containerized application locally before deploying to Azure:

#### 1. Setup local env
Locate the sample.env file in root, copy and create a new file named .env. Safely pasting your Azure OpenAI API key in this newly created .env file. 

> This .env will be ignored in online so it will be ONLY available in your local environment

#### 2. run below

The test-local.ps1 file automatically load API key from .env and use it for local docker env variables. 

> Both blob data path and Application Insight logger will be disabled automatically when testing in local docker.

```powershell
# Run the local testing script
.\test-local.ps1
```

This will:
1. Build the Docker image locally
2. Run the container with ports 3000 (frontend) + 8001 (backend)
3. Mount your config files and `.adalflow` cache
4. Open browser to http://localhost:3000

**Useful commands:**
```powershell
# View container logs
docker logs -f codewiki-local

# Stop the container
docker stop codewiki-local

# Access container shell
docker exec -it codewiki-local bash
```

### Option B: manual setup in local

#### Step 1: Install Dependencies

```bash
# Clone the repository

# Optional: create virutal environment and activate it
python -m venv .venv
.venv\scripts\activate   

# Install Poetry (if not already installed)
pip install poetry

# Install Python dependencies
poetry install

# Install JavaScript dependencies
npm install
```

#### Step 2: Start the Application

```bash
# Terminal 1: Start the API server
python -m backend.main

# Terminal 2: Start the frontend
npm run dev
```

#### Step 3: Use CodeWiki!

1. Open [http://localhost:3000](http://localhost:3000) in your browser
2. Enter a repository URL (e.g., `https://github.com/microsoft/autogen`)
3. Enter your personal access token
4. Click "Generate Wiki", it would take some time to generate wiki for large code base. You can check back on the homepage.


## How to deploy to Azure cloud

### Using Web App (Recommended)
- locate publish-web.ps1 under root
- Ensure have the detailed Azure Resources names ready and filled in `config.py`
- Run the script, which will build docker image and pull to ACR used for the web app service created during deployment steps

### Using Container App (Depercated due to security)

<details>

- locate publish-container.ps1 under root
- fill the `Configuration` section
- Run the script, which will create an Azure Container Registry, pull images, setup container env, and upload code.

</details>

### Docker Architecture

This project uses a single `Dockerfile` with nginx for both local testing and Azure deployment:

| File | Purpose | nginx | Use Case |
|------|---------|-------|----------|
| `Dockerfile` | All environments | ✅ Yes | Local testing (`test-local.ps1`) and Azure deployment |

**Why nginx?**

Azure Container Apps has a **240-second hard limit** on HTTP request timeouts. Large repository embedding can take 30-60+ minutes, which would fail with 504 Gateway Timeout errors.

The nginx reverse proxy enables **WebSocket connections** which bypass HTTP timeout limits:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Container                                │
│                                                                  │
│   External :3000                                                 │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────┐     /ws/*      ┌──────────────────────────┐       │
│   │  nginx  │ ────────────▶  │   FastAPI Backend :8001  │       │
│   │         │  (WebSocket)   │   - Wiki generation      │       │
│   │         │                │   - Chat API             │       │
│   │         │     /*         │   - Embeddings           │       │
│   │         │ ────────────▶  └──────────────────────────┘       │
│   │         │  (HTTP)                                            │
│   │         │                ┌──────────────────────────┐       │
│   │         │ ────────────▶  │   Next.js Frontend :3001 │       │
│   └─────────┘                │   - UI/React app         │       │
│                              └──────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

**nginx configuration highlights** (`nginx.conf`):
- WebSocket timeout: 7 days (for long-running embedding operations)
- HTTP timeout: 1 hour
- Proper WebSocket upgrade headers for `/ws/` routes

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Azure Container Apps                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              codewiki (Container App)                    │   │
│  │  ┌─────────────────┐    ┌──────────────────────────┐   │   │
│  │  │   Next.js       │    │       FastAPI            │   │   │
│  │  │   Frontend      │───▶│       Backend            │   │   │
│  │  │   :3001         │    │       :8001              │   │   │
│  │  └─────────────────┘    └──────────────────────────┘   │   │
│  │           ▲                        ▲                    │   │
│  │           └────────┬───────────────┘                    │   │
│  │                    │                                    │   │
│  │              ┌─────┴─────┐                              │   │
│  │              │   nginx   │ ◀── External :3000           │   │
│  │              └───────────┘                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                    Managed Identity                              │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Azure OpenAI   │  │  Azure Blob     │  │  Application    │
│  - GPT Models   │  │  Storage        │  │  Insights       │
│  - Embeddings   │  │  - Wiki Data    │  │  - Logs         │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Resource Requirements

For large repositories (5000+ files), the container needs sufficient memory for embedding:

| Profile | CPU | Memory | Suitable For |
|---------|-----|--------|-------------|
| Consumption | 2.0 | 4 GB | Small repos (<1000 files) |
| Consumption | 4.0 | 8 GB | Medium repos (1000-3000 files) |
| **D4 (Dedicated)** | 4.0 | **16 GB** | Large repos (5000+ files) - **Default** |

The `deploy-azure.ps1` script creates a **D4 dedicated workload profile** with 4 CPU / 16 GB by default. This is required because:
- Consumption tier max is 8GB, which causes OOM (exit code 137) for large repos
- D4 dedicated profile provides up to 16GB RAM per container
- Costs more than Consumption, but necessary for large repository embedding

### Useful Azure CLI Commands

```bash
# View container app logs
az containerapp logs show -n codewiki -g RG-ORCAS-DEEPWIKI --follow

# Restart the app
az containerapp revision restart -n codewiki -g RG-ORCAS-DEEPWIKI

# Scale the app
az containerapp update -n codewiki -g RG-ORCAS-DEEPWIKI --min-replicas 2 --max-replicas 10

# Switch to Consumption tier (after embedding, for cost savings)
az containerapp update -n codewiki -g RG-ORCAS-DEEPWIKI --workload-profile-name Consumption --cpu 4 --memory 8Gi

# Switch to D4 dedicated (for large repo embedding)
az containerapp update -n codewiki -g RG-ORCAS-DEEPWIKI --workload-profile-name D4 --cpu 4 --memory 16Gi

# Get app URL
az containerapp show -n codewiki -g RG-ORCAS-DEEPWIKI --query properties.configuration.ingress.fqdn -o tsv


##  How It Works

```mermaid
graph TD
    A[User inputs repo URL] --> AA{Private repo?}
    AA -->|Yes| AB[Add access token]
    AA -->|No| B[Clone Repository]
    AB --> B
    B --> C[Analyze Code Structure]
    C --> D[Create Embeddings with Azure OpenAI]
    D --> E[Generate Documentation with Azure OpenAI]
    D --> F[Create Visual Diagrams]
    E --> G[Organize as Wiki]
    F --> G
    G --> H[Interactive DeepWiki]
```

DeepWiki uses Azure OpenAI to:

1. Clone and analyze the repository (including private repos with token authentication)
2. Create embeddings of the code using Azure OpenAI's `text-embedding-3-large`
3. Generate documentation using Azure OpenAI's GPT models
4. Create visual diagrams to explain code relationships
5. Organize everything into a structured wiki
6. Enable intelligent Q&A with the repository through the Ask feature

## 🛠️ Project Structure

```
deepwiki/
├── pyproject.toml        # Python dependencies (Poetry)
├── poetry.lock           # Poetry lock file
├── package.json          # Node.js dependencies
│
├── backend/              # Backend API server
│   ├── main.py           # API entry point
│   ├── api.py            # FastAPI implementation
│   ├── rag.py            # Retrieval Augmented Generation
│   ├── data_pipeline.py  # Data processing utilities
│   ├── azureai_client.py # Azure OpenAI client
│   └── config/           # Configuration files
│       ├── generator.json    # Model configuration
│       ├── embedder.json     # Embedding configuration
│       └── infra.json        # Infrastructure & MSI configuration
│
├── src/                  # Frontend Next.js app
│   ├── app/              # Next.js app directory
│   └── components/       # React components
│
├── img/                  # Images and screenshots
│   ├── public/           # Next.js public assets
│   └── screenshots/      # Documentation screenshots
│
└── logs/                 # Application logs
    ├── backend-*.log     # Backend logs (daily rotation)
    └── frontend-*.log    # Frontend logs (daily rotation)
```

## ⚙️ Configuration

All configuration is centralized in `backend/config/infra.json`. 

### infra.json Structure

| Field | Description |
|-------|-------------|
| `managed_identity.name` | Name of the User-Assigned Managed Identity |
| `managed_identity.client_id` | Client ID of the Managed Identity |
| `azure_openai.endpoint` | Azure OpenAI endpoint URL |
| `azure_openai.api_version` | API version (e.g., `2024-12-01-preview`) |
| `azure_openai.deployment` | Deployment name for text generation |
| `azure_openai_embedding.endpoint` | Azure OpenAI endpoint for embeddings |
| `azure_openai_embedding.api_version` | API version for embeddings |
| `azure_openai_embedding.deployment` | Deployment name for embeddings |
| `azure_blob_storage.enabled` | Enable Azure Blob Storage for persistence (`true`/`false`) |
| `azure_blob_storage.account_name` | Storage account name |
| `azure_blob_storage.container_name` | Blob container name (e.g., `deepwiki-data`) |
| `azure_application_insights.enabled` | Enable Application Insights for centralized logging (`true`/`false`) |
| `azure_application_insights.name` | Application Insights resource name |
| `azure_application_insights.connection_string` | Application Insights connection string |

### Other Configuration Files

- **`backend/config/generator.json`**: Text generation model parameters (temperature)
- **`backend/config/embedder.json`**: Embedding model and text processing settings

## 💾 Storage Architecture

DeepWiki supports two **mutually exclusive** storage modes:

### Storage Modes

| Mode | When | Use Case |
|------|------|----------|
| **Blob Mode** | `azure_blob_storage.enabled: true` | Production deployments (Azure Container Apps, etc.) |
| **Local Mode** | `azure_blob_storage.enabled: false` | Local development and testing |

> **Important**: These modes are mutually exclusive. There is NO syncing between blob and local storage.

### What Gets Stored

| Data | Description | Blob Path | Local Path |
|------|-------------|-----------|------------|
| **Repositories** | Cloned repository files | `repos/{repo_name}/` | `~/.adalflow/repos/{repo_name}/` |
| **Databases** | Embedded document databases (pkl) | `databases/{repo_name}.pkl` | `~/.adalflow/databases/{repo_name}.pkl` |
| **Wiki Cache** | Generated wiki content (JSON) | `wikicache/*.json` | `~/.adalflow/wikicache/*.json` |

### Local Working Directory

Even in **Blob Mode**, you may see files in `~/.adalflow/`. This is the **temporary working directory**:

```
~/.adalflow/
├── repos/                           # ← Temporary: downloaded from blob for processing
├── databases/                       # ← Not used in blob mode (blob is source of truth)
├── wikicache/                       # ← Not used in blob mode (blob is source of truth)
└── cache_AzureAIClient_*.db/        # ← Local-only LLM response cache (intentional)
```

**Why local copies exist in blob mode:**
- File parsing and embedding requires local file access
- FAISS index building reads files from disk
- The local copy is a **working cache**, not persistent storage

**Source of truth:**
- **Blob Mode**: Azure Blob Storage is the source of truth. Local is temporary.
- **Local Mode**: Local filesystem is the source of truth.

### LLM Response Cache

The `cache_AzureAIClient_*.db/` folder is **intentionally local-only**:
- Created by adalflow's DiskCache for caching LLM API responses
- Avoids redundant API calls for identical queries
- Machine-specific, ephemeral performance optimization
- Not persistent data - safe to delete anytime

## 📊 Logging

DeepWiki uses daily rotating log files with optional Azure Application Insights integration:

### Local Logs

- **Backend logs**: `logs/backend-YYMMDD.log`
- **Frontend logs**: `logs/frontend-YYMMDD.log`

Set logging level in your environment:

```bash
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

### Application Insights (Optional)

For centralized cloud logging, configure Azure Application Insights:

1. **Create Application Insights** in Azure Portal
2. **Get the connection string** from the Application Insights overview page
3. **Configure in `infra.json`**:
   ```json
   "azure_application_insights": {
     "enabled": true,
     "name": "your-app-insights-name",
     "connection_string": "InstrumentationKey=...;IngestionEndpoint=..."
   }
   ```
4. **Assign role** - Your identity needs **"Monitoring Metrics Publisher"** role on the Application Insights resource:
   ```bash
   az role assignment create \
     --assignee <your-user-or-msi-object-id> \
     --role "Monitoring Metrics Publisher" \
     --scope <application-insights-resource-id>
   ```

**Authentication:**
- In Azure (VMs, Container Apps): Uses Managed Identity from `infra.json`
- Locally: Uses Azure CLI credentials (`az login`)

**View logs** in Azure Portal → Application Insights → Logs → Query the `traces` table:
```kusto
traces
| where cloud_RoleName == "backend"
| order by timestamp desc
| take 100
```

## 🤖 Ask & DeepResearch Features

### Ask Feature

Chat with your repository using RAG (Retrieval Augmented Generation):

- **Context-Aware**: Get accurate answers based on the actual code
- **Real-Time Streaming**: See responses as they're generated
- **Conversation History**: Maintains context between questions

### DeepResearch Feature

Multi-turn research for complex topics:

- **In-Depth Investigation**: Multiple research iterations
- **Structured Process**: Clear research plan with updates
- **Comprehensive Conclusion**: Final answer based on all iterations

Toggle "Deep Research" in the Ask interface for thorough analysis.

## 📱 Screenshots

![DeepWiki Main Interface](img/screenshots/Interface.png)
*The main interface of DeepWiki*

![Snippet of Wiki Generated](img/screenshots/wiki.png)
*Sample Wiki generated*

![DeepResearch Feature](img/screenshots/DeepResearch.png)
*DeepResearch conducts multi-turn investigations*

## Limitation (Working in progress)
- Multi-threaded: Currently, if there is already an ongoing code embedding process running, other newly coming request will be queued. In future, CodeWiki will allow multi-threaded chunking process.
- Timely scheduled pipeline: with development of any code projects, code repo will change time to time. To ensure accuracy with Wiki generated, need to have a scheduled pipeline to analyze the code changes time to time.
