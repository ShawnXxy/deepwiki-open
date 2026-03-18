# General
# ---------------------------------------------------------------------------

# Chose the location of your copilot.
# If you plan to deploy a new Open AI workspace, this needs to be in a region with GPT4 enabled.
# Typically, East us is a good choice.
location = "eastasia"

# This is the id of YOUR identity.
# You can find it by looking at yourself on Microsoft Intra ID on the Azure portal.
# IMPORTANT: this is the object id of your identity.
# EXAMPLE: e0976b19-655c-421d-8043-f77989a894db
deployment_identity_principal_id = "3cd75edc-71a6-4ee7-a769-6998262553a8"

# This is the name of the managed identity that will be given to your copilot.
# EXAMPLE: my-dricopilot-identity
dri_copilot_identity_name = "mid-orcas-deepwiki"

# This is the tenant id in which you are deploying your resource.
# The only one we have tested to far is the Microsoft tenant, even if the process should work in others.
# Unless you want to deploy in a different tenant, you can leave this as is.
tenant_id = "72f988bf-86f1-41af-91ab-2d7cd011db47"

# This will be the subscription id in which you are going to deploy the resources.
subscription_id = "4f3f8f41-5643-4664-8c12-ce6b78ceb81f"

# This will be the resource group in which you are going to deploy your resources (within your subscription).
resource_group = "RG-ORCAS-DEEPWIKI"

# Storage Account
# ---------------------------------------------------------------------------

# This is the name of the storage account that will be created for your copilot.
# It must be globally unique, lowercase, 3-24 characters, no dashes or spaces.
# EXAMPLE: deepwikiorcasstorage
storage_account_name = "bloborcasdeepwiki"

# Application Insights & Monitoring
# ---------------------------------------------------------------------------

# This is the name of the Application Insights resource for monitoring.
# EXAMPLE: deepwiki-appinsights
application_insights_name = "appinsite-orcas-deepwiki-ea"

# This is the name of the Log Analytics workspace for Application Insights.
# EXAMPLE: deepwiki-loganalytics  
log_analytics_workspace_name = "log-orcas-deepwiki-ea"

# App Service (Container Mode)
# ---------------------------------------------------------------------------

# This is the name of the App Service Plan (Linux).
# EXAMPLE: deepwiki-appservice-plan
app_service_plan_name = "ASP-codewiki-orcas"

# This is the name of the App Service (Web App).
# It must be globally unique.
# EXAMPLE: deepwiki-webapp
app_service_name = "orcascodewiki"

# App Service Plan SKU. Recommended: P1v3 for production, B1 for dev/test.
# Options: F1 (Free), B1, B2, B3, S1, S2, S3, P1v2, P2v2, P3v2, P1v3, P2v3, P3v3
app_service_plan_sku = "P2mv4"

# Enable network restrictions with service tags (AzureTrafficManager, CorpNetPublic, CorpNetSAW).
# Set to False only for development/testing.
enable_app_service_network_restrictions = True

# Azure Container Registry Configuration
# ---------------------------------------------------------------------------

# This is the name of the Azure Container Registry for storing container images.
# Must be globally unique, lowercase, alphanumeric only (no dashes/spaces), 5-50 characters.
# EXAMPLE: acrdeepwikiorcas
container_registry_name = "acrorcascodewiki"

# Container image name (without registry prefix or tag).
# EXAMPLE: deepwiki
container_image_name = "deepwiki"

# Open AI
# ---------------------------------------------------------------------------

# Set this to true if you want to create a new Open AI workspace.
# Since there is a fixed quota of deployment numbers for some models, you might want to re-use existing deployments.
# In this case, set this to false.
#
# IMPORTANT: if you do not let us create and manage your Open AI endpoint, you need to make sure that accesses are set properly.
# Both you and the DRI Copilot identity need to have the `Cognitive Services OpenAI User` Role.
is_creating_open_ai_endpoint = False

# This will be the name of your Azure Open AI resource.
# It has to be unique within Azure.
# Note that if you don't want us to create a new workspace for you, you will need to set this name to the name of an existing Open AI resource.
# EXAMPLE: dricopilot-aoai
open_ai_resource_name = "aoai-orcas-deepwiki-kc"

# If using an existing Azure OpenAI resource in a DIFFERENT resource group, specify it here.
# Leave empty "" to use the same resource group as other resources (default).
# EXAMPLE: "RG-SHARED-OPENAI"
open_ai_resource_group = "RG-ORCAS-DEEPWIKI"

# Reasoning Model Configuration
# Purpose: A specialized model for complex reasoning and decision-making tasks (e.g., o4-mini).
# Use Cases: Code generation, debugging, and complex data interpretation.
open_ai_reasoning_model_deployment_name = "o4-mini"
open_ai_reasoning_model_model_name = "o4-mini"
open_ai_reasoning_model_tokens_per_minute = 200
open_ai_reasoning_model_version = "2025-04-16"
open_ai_reasoning_model_sku_name = "GlobalStandard"

# Embedding Model Configuration
# Purpose: Text embedding model for vector search and RAG.
open_ai_embedding_deployment_name = "text-embedding-3-large"
open_ai_embedding_model_name = "text-embedding-3-large"
open_ai_embedding_tokens_per_minute = 50
open_ai_embedding_version = "1"
open_ai_embedding_sku_name = "Standard"


# NSP (network service perimeter) Configuration
# ---------------------------------------------------------------------------

# Only if you want to enable NSP, please provide the following information.
# Currently, we added NSP support for Storage and Key Vault. We will be adding more services over time.

# NSP name - leave empty to skip NSP deployment
# EXAMPLE: DeepWikiNSP
nsp_name = "nsp-codewiki-orcas"

# NSP profile name
# EXAMPLE: DeepWikiNSPProfile
nsp_profile_name = "nspfile-codewiki-orcas"

# NSP access mode for each resource type.
# Options: "Enforced" (blocks violations) or "Learning" (allows traffic, logs violations)
# Enforced mode tested successfully in "corp" with Storage and Key Vault.
# Manual diagnostic setup required for NSP logs. See: https://eng.ms/docs/cloud-ai-platform/azure-core/azure-networking/sdn-dbansal/azure-virtual-network-manager/nsp-dataplane-library/articles/ns22tsg/nsp/nsplogging
# EXAMPLE: Learning/Enforced
nsp_access_mode_for_storage = "Enforced"
nsp_access_mode_for_key_vault = "Enforced"

# For the deployment of some features, you may want to create tags that follow them throughout the resource. Please provide any such tags below
# EXAMPLE: {"Owner" : "DRICopilot"}, {}
resource_tags = {"Owner" : "DaP CN Orcas"}


########################################################################################
## SECTION Reserved for future use - currently not implemented in deployment scripts ###
########################################################################################

# Azure Machine Learning
# ---------------------------------------------------------------------------

# This will be the name of your machine learning workspace.
# It has to be unique within Azure.
# EXAMPLE: dricopilot-aml
machine_learning_workspace_name = "aml-orcas-codewiki"

# As part of deploying a Azure Machine Learning workspace, we will also create a set of required resources.
# Those are: a storage account, a container registry, a key vault, and an application insights.
# This name should be without spaces, dashes, and numbers. It should all be lower case and not more than 16 characters.
# EXAMPLE: dricoaml
machine_learning_workspace_sub_components_name_prefix = "amlorcascw"

# When set to true, this will create an Azure Virtual Network for the Azure Machine Learning workspace.
# The blob storage also will be added to a VNet.
# "Azure AI Enterprise Network Connection Approver" will be required to be assigned to your managed identity for below resources:
#  - keyvault
#  - storage
#  - container register
is_creating_vnet_for_azure_ml = True

# When set to true, this will enable disk encryption for azure ml compute clusters,
# which is a best practice as per the Software Development Lifecycle (SDL) guidelines to keep this enabled.
# For more info, please refer: https://learn.microsoft.com/en-us/azure/machine-learning/concept-data-encryption?view=azureml-api-2#compute-cluster
# NOTE: This setting cannot be changed after the workspace is created. 
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?view=azureml-api-2&tabs=python
is_enabling_disk_encryption_for_azure_ml = False

# Azure Cognitive Search
# ---------------------------------------------------------------------------

# This is the name of the Azure Cognitive Search Service that will host your indexes.
# It needs to be all lower case, no spaces and dashes, and no numbers, and unique within Azure.
# EXAMPLE: dricopilotsearch
search_service_name = "acsorcascodewiki"

# Azure Key Vault (Optional - for deploy_planning)
# ---------------------------------------------------------------------------

# This is the name of the Azure Key Vault for storing secrets.
# Must be globally unique, 3-24 characters, alphanumeric and hyphens only.
# EXAMPLE: deepwiki-keyvault
key_vault_name = "kvorcascodewiki"

# Enable RBAC authorization for Key Vault (recommended).
# When true, uses Azure RBAC instead of access policies.
key_vault_enable_rbac = True


# And you are done with the basics!