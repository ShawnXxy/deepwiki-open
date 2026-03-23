- Key vault access: this managed identity has below roles assigned and myself "Key Vault Administrator" role.
    - Key Vault Crypto User
    - Key Vault Secrets Officer
    - Key Vault Certificate User
    - Azure AI Administrator

- Azure Cognitive Search (AI Search) access: the identity has granted below roles :
    - Search Index Data Reader
    - Search Service Contributor

- Azure OpenAI access: the identity was granted "Cognitive Services User" and "Cognitive Services OpenAI User " role 

- Access to storage account: this managed identity has below roles assigned:
    - Storage File Data Privileged Contributor
    - Storage Blob Data Contributor
    - Storage Blob Data Reader
    - Azure AI Administrator
    - Azure AI Enterprise Network Connection Approver (used to setup managed VNet for AML)

- Access to the AML workspace:  identity has assigned "AzureML Data Scientist" 

- Application Insight: this managed identity has below roles assigned:
    - Azure AI Administrator

- Container Registry for AML: this managed identity has below roles assigned:
    - Azure AI Administrator