- Key vault access: this managed identity has below roles assigned and myself "Key Vault Administrator" role.
    - Key Vault Crypto User
    - Key Vault Secrets Officer
    - Key Vault Certificate User
    - Azure AI Administrator
    - Key Vault Administrator

- Azure Cognitive Search (AI Search) access: the identity has granted below roles :
    - Search Index Data Reader
    - Search Service Contributor

- Azure OpenAI access: the identity was granted "Cognitive Services User" and "Cognitive Services OpenAI User " role 

- Access to storage account: this managed identity has below roles assigned:
    - Storage File Data Privileged Contributor
    - Storage Blob Data Contributor
    - Azure AI Administrator
    - Azure AI Enterprise Network Connection Approver (used to setup managed VNet for AML)

- Access to the AML workspace:  identity has assigned "AzureML Data Scientist" 

- Application Insight: this managed identity has below roles assigned:
    - Azure AI Administrator

- Container Registry for AML: this managed identity has below roles assigned:
    - Azure AI Administrator

## DeepWiki RAI Policy Manager (custom role)

The processor pipeline's `GuardSession`
([backend/utils/guard_session.py](../backend/utils/guard_session.py))
snapshots the Azure OpenAI content-filter (RAI) policy on entry,
temporarily relaxes a single safe-to-toggle filter row when a
`content_filter` `BadRequestError` fires (e.g. Profanity), retries the
LLM call once, and restores the original policy on exit. This needs
`raiPolicies/read` + `raiPolicies/write` on the AOAI account, which
the built-in `Cognitive Services User` does not grant.

The plan ([backend/processor/content_filter_autorelax_plan.md](../backend/processor/content_filter_autorelax_plan.md)
§7) defines a custom role narrower than the built-in
`Cognitive Services Contributor` (no data-plane, no resource
lifecycle). The role definition body is checked in at
[parameters/DeepWikiRAIPolicyManager.RoleDefinition.json](parameters/DeepWikiRAIPolicyManager.RoleDefinition.json).

**Create the role (once per subscription):**

```powershell
az role definition create `
  --role-definition "@Deployments/parameters/DeepWikiRAIPolicyManager.RoleDefinition.json"
```

**Assign to the user-assigned managed identity (cloud mode):**

```powershell
$mi = az identity show `
  --name mid-orcas-deepwiki `
  --resource-group RG-ORCAS-DEEPWIKI `
  --query principalId -o tsv

az role assignment create `
  --assignee-object-id $mi `
  --assignee-principal-type ServicePrincipal `
  --role "DeepWiki RAI Policy Manager" `
  --scope "/subscriptions/4f3f8f41-5643-4664-8c12-ce6b78ceb81f/resourceGroups/RG-ORCAS-DEEPWIKI/providers/Microsoft.CognitiveServices/accounts/aoai-orcas-deepwiki-kc"
```

**Assign to a developer (local mode):**

```powershell
az role assignment create `
  --assignee <user@microsoft.com> `
  --role "DeepWiki RAI Policy Manager" `
  --scope "/subscriptions/4f3f8f41-5643-4664-8c12-ce6b78ceb81f/resourceGroups/RG-ORCAS-DEEPWIKI/providers/Microsoft.CognitiveServices/accounts/aoai-orcas-deepwiki-kc"
```

Without this role, `GuardSession` enters degraded mode at startup
(one WARNING line) and the pipeline runs unchanged with no
auto-relax. Docker mode auto-skips the feature regardless of role.
