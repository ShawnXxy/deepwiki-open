"""
Utility functions for DeepWiki infrastructure deployment.

Provides helper functions for ARM template deployment and Azure ML datastore configuration.
"""

import datetime
import json
import re
import importlib.util
from pathlib import Path
from azure.identity import AzureCliCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import DeploymentMode

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AzureBlobDatastore, AzureFileDatastore
from azure.ai.ml.entities._credentials import NoneCredentialConfiguration
from azureml.core import Workspace, Datastore


# Resource type to Azure provider mapping
RESOURCE_TYPE_MAP = {
    "managed_identity": "Microsoft.ManagedIdentity/userAssignedIdentities",
    "storage": "Microsoft.Storage/storageAccounts",
    "openai": "Microsoft.CognitiveServices/accounts",
    "appinsights": "Microsoft.Insights/components",
    "log_analytics": "Microsoft.OperationalInsights/workspaces",
    "app_service_plan": "Microsoft.Web/serverfarms",
    "app_service": "Microsoft.Web/sites",
    "nsp": "Microsoft.Network/networkSecurityPerimeters",
    "acr": "Microsoft.ContainerRegistry/registries",
}


def check_resource_exists(resource_name, resource_type, subscription_id, resource_group, credential=None):
    """
    Check if an Azure resource exists and return its details.
    
    Args:
        resource_name: Name of the resource
        resource_type: Type key (e.g., "storage", "managed_identity") or full Azure type
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        credential: Azure credential (optional, will create if not provided)
        
    Returns:
        Tuple of (exists: bool, resource_info: dict or None)
        resource_info contains: {'location': str, 'id': str, 'provisioning_state': str}
    """
    if credential is None:
        credential = AzureCliCredential()
    
    # Map short type to full Azure resource type
    azure_type = RESOURCE_TYPE_MAP.get(resource_type, resource_type)
    
    client = ResourceManagementClient(credential, subscription_id)
    
    try:
        resource = client.resources.get(
            resource_group,
            azure_type.split('/')[0],  # Provider namespace
            "",  # Parent resource path
            azure_type.split('/')[1],  # Resource type
            resource_name,
            api_version=_get_api_version(azure_type)
        )
        return True, {
            'location': resource.location,
            'id': resource.id,
            'provisioning_state': resource.properties.get('provisioningState', 'Unknown') if resource.properties else 'Unknown'
        }
    except Exception:
        return False, None


def _get_api_version(resource_type):
    """Get appropriate API version for resource type."""
    api_versions = {
        "Microsoft.ManagedIdentity/userAssignedIdentities": "2023-01-31",
        "Microsoft.Storage/storageAccounts": "2023-01-01",
        "Microsoft.CognitiveServices/accounts": "2023-05-01",
        "Microsoft.Insights/components": "2020-02-02",
        "Microsoft.OperationalInsights/workspaces": "2022-10-01",
        "Microsoft.Web/serverfarms": "2023-01-01",
        "Microsoft.Web/sites": "2023-01-01",
        "Microsoft.Network/networkSecurityPerimeters": "2023-08-01-preview",
        "Microsoft.ContainerRegistry/registries": "2023-07-01",
    }
    return api_versions.get(resource_type, "2023-01-01")


def check_and_prepare_deployment(resource_name, resource_type, config_location, subscription_id, resource_group, credential=None):
    """
    Check if resource exists and prepare for deployment.
    
    Handles three scenarios:
    1. Resource doesn't exist -> Proceed with deployment
    2. Resource exists in same location -> Proceed with update
    3. Resource exists in different location -> Warning, ask user to update config
    
    Args:
        resource_name: Name of the resource
        resource_type: Type key (e.g., "storage", "managed_identity")
        config_location: Location from config.py
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        credential: Azure credential (optional)
        
    Returns:
        Tuple of (should_deploy: bool, effective_location: str, message: str)
    """
    exists, info = check_resource_exists(
        resource_name, resource_type, subscription_id, resource_group, credential
    )
    
    if not exists:
        return True, config_location, f"Resource does not exist. Will create in '{config_location}'."
    
    existing_location = info['location']
    
    if existing_location.lower().replace(' ', '') == config_location.lower().replace(' ', ''):
        return True, config_location, f"⚠ Resource already exists in '{existing_location}'. Will UPDATE existing resource."
    else:
        return False, existing_location, (
            f"⚠ LOCATION MISMATCH: Resource exists in '{existing_location}' but config specifies '{config_location}'.\n"
            f"  ARM cannot move resources between regions.\n"
            f"  Options:\n"
            f"    1. Update config.py to use location = '{existing_location}'\n"
            f"    2. Delete the existing resource and redeploy\n"
            f"    3. Use a different resource name"
        )


def get_formatted_datetime():
    """Returns the current datetime in a formatted string."""
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y%m%d%H%M")
    return formatted_datetime


def load_config(config_path="config.py"):
    """
    Load configuration from config.py and return as a dictionary.
    
    Args:
        config_path: Path to config.py file (default: "config.py")
        
    Returns:
        Dictionary containing all configuration variables
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract all non-private variables
    config = {}
    for attr in dir(config_module):
        if not attr.startswith('_'):
            config[attr] = getattr(config_module, attr)
    
    return config


def resolve_parameters(template_path, config=None, config_path="config.py"):
    """
    Resolve parameter template file by replacing {{config_key}} placeholders with values from config.py.
    
    Placeholder format: {{config_key}}
    - config_key should match the variable name in config.py (case-sensitive)
    - Supports nested values for objects like resource_tags
    
    Args:
        template_path: Path to the parameter template file (.json)
        config: Pre-loaded config dictionary (optional, will load from config_path if not provided)
        config_path: Path to config.py file (default: "config.py")
        
    Returns:
        Dictionary containing resolved parameters ready for ARM deployment
        
    Example:
        Template: {"value": "{{storage_account_name}}"}
        Config: storage_account_name = "mystorageaccount"
        Result: {"value": "mystorageaccount"}
    """
    if config is None:
        config = load_config(config_path)
    
    # Read template file
    template_file = Path(template_path)
    if not template_file.exists():
        raise FileNotFoundError(f"Parameter template not found: {template_path}")
    
    with open(template_file, "r", encoding="utf-8") as f:
        template_content = f.read()
    
    # Find all placeholders: {{key}}
    placeholder_pattern = r'\{\{(\w+)\}\}'
    
    def replace_placeholder(match):
        key = match.group(1)
        if key in config:
            value = config[key]
            # For JSON, we need to handle different types
            if isinstance(value, str):
                return value
            elif isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, (dict, list)):
                # For complex types, return JSON representation without outer quotes
                return json.dumps(value)
            else:
                return str(value)
        else:
            print(f"  ⚠ Warning: Config key '{key}' not found, keeping placeholder")
            return match.group(0)
    
    # Replace placeholders
    resolved_content = re.sub(placeholder_pattern, replace_placeholder, template_content)
    
    # Parse as JSON
    try:
        resolved_params = json.loads(resolved_content)
        return resolved_params
    except json.JSONDecodeError as e:
        print(f"  ✗ Error parsing resolved parameters: {e}")
        print(f"  Content preview: {resolved_content[:500]}...")
        raise


def generate_all_parameters(config=None, config_path="config.py", output_dir="parameters/resolved"):
    """
    Generate all resolved parameter files from templates.
    
    This reads parameter templates from parameters/ and generates resolved versions
    with all {{config_key}} placeholders replaced with actual values from config.py.
    
    Args:
        config: Pre-loaded config dictionary (optional)
        config_path: Path to config.py file
        output_dir: Directory to write resolved parameter files (default: "parameters/resolved")
        
    Returns:
        Dictionary mapping template name to resolved parameters
    """
    if config is None:
        config = load_config(config_path)
    
    templates_dir = Path("parameters")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find all parameter template files
    for template_file in templates_dir.glob("*.Parameters.json"):
        print(f"Resolving: {template_file.name}")
        try:
            resolved = resolve_parameters(template_file, config)
            results[template_file.stem] = resolved
            
            # Write resolved file
            output_file = output_path / template_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(resolved, f, indent=4)
            print(f"  ✓ Written to {output_file}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    return results


def get_parameters_for_deployment(template_name, config=None, config_path="config.py"):
    """
    Get resolved parameters for a specific deployment template.
    
    This is a convenience function that resolves parameters and extracts
    just the parameter values (without the ARM template wrapper).
    
    Args:
        template_name: Name of the template (e.g., "UMI", "STORAGE", "AOAI")
        config: Pre-loaded config dictionary (optional)
        config_path: Path to config.py file
        
    Returns:
        Dictionary of parameter name -> value, ready for deploy_arm()
    """
    if config is None:
        config = load_config(config_path)
    
    template_path = Path("parameters") / f"{template_name}.Parameters.json"
    resolved = resolve_parameters(template_path, config)
    
    # Extract just the values from the ARM parameter format
    params = {}
    if "parameters" in resolved:
        for key, value_obj in resolved["parameters"].items():
            if isinstance(value_obj, dict) and "value" in value_obj:
                params[key] = value_obj["value"]
            else:
                params[key] = value_obj
    
    return params


def deploy_arm(template_file_path, deployment_name, parameters, subscription_id, resource_group, skip_role_assignment=False):
    """
    Deploys an ARM template to Azure.
    
    Args:
        template_file_path: Path to the ARM template JSON file
        deployment_name: Base name for the deployment
        parameters: Dictionary of parameters for the template
        subscription_id: Azure subscription ID
        resource_group: Name of the resource group
        skip_role_assignment: If True, removes role assignment resources from template
        
    Returns:
        Dictionary of deployment outputs if successful, empty dict otherwise
    """
    
    # Load ARM template
    with open(template_file_path, "r") as template_file:
        template = json.load(template_file)
    
    if skip_role_assignment:
        # Remove the role assignment section from the template
        template["resources"] = [
            resource for resource in template["resources"] 
            if resource["type"] != "Microsoft.Authorization/roleAssignments"
        ]

    # Authenticate with Azure
    credential = AzureCliCredential()
    client = ResourceManagementClient(credential, subscription_id)

    # Wrap parameters in {"value": ...} format if not already wrapped
    formatted_parameters = {}
    for key, value in parameters.items():
        if isinstance(value, dict) and "value" in value:
            formatted_parameters[key] = value
        else:
            formatted_parameters[key] = {"value": value}

    # Deployment properties
    deployment_properties = {
        'properties': {
            'mode': DeploymentMode.incremental,
            'template': template,
            'parameters': formatted_parameters
        }
    }

    try:
        # Create unique deployment name with timestamp
        unique_deployment_name = f"{deployment_name}-{get_formatted_datetime()}"
        
        print(f"Starting deployment: {unique_deployment_name}")
        print("This may take several minutes...")
        
        deployment_async_operation = client.deployments.begin_create_or_update(
            resource_group,
            unique_deployment_name,
            deployment_properties
        )

        deployment_result = deployment_async_operation.result()

        # Check the provisioning state
        if deployment_result.properties.provisioning_state == 'Succeeded':
            print("✓ Deployment successful!")
            if deployment_result.properties.outputs:
                return deployment_result.properties.outputs
            else:
                return {}
        else:
            print("✗ Deployment did not succeed.")
            print(f"Deployment finished with state: {deployment_result.properties.provisioning_state}")
            return {}

    except Exception as e:
        error_message = str(e)
        # Check if the error is only about role assignments that already exist
        if "RoleAssignmentExists" in error_message:
            print("⚠ Note: Some role assignments already exist (this is normal for redeployments)")
            print("✓ Deployment completed successfully (ignoring pre-existing role assignments)")
            # Try to get the deployment result even with this error
            try:
                deployment_result = client.deployments.get(resource_group, unique_deployment_name)
                if deployment_result.properties.outputs:
                    return deployment_result.properties.outputs
            except:
                pass
            return {}
        else:
            print(f"✗ Deployment failed: {e}")
            raise


def set_datastore_credential_to_managed_identity(subscription_id, resource_group, workspace_name):
    """
    Update Azure ML workspace datastores to use Managed Identity authentication.
    
    This function updates the default datastores (blob, file, etc.) to use
    credential-less authentication via managed identity instead of account keys.
    
    Args:
        subscription_id: Azure subscription ID
        resource_group: Name of the resource group
        workspace_name: Name of the Azure ML workspace
    """
    
    print("=" * 60)
    print("UPDATING: AML Datastore Authentication")
    print("=" * 60)
    print(f"Workspace: {workspace_name}")
    print(f"Resource Group: {resource_group}")
    
    try:
        # Authenticate with Azure using both APIs:
        # 1. azure.ai.ml (MLClient) - for updating credentials
        # 2. azureml.core - for setting default datastore
        credential = AzureCliCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        workspace = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )

        # List of datastores to update
        # Order matters: update non-default datastores first
        datastore_names = [
            "workspacefilestore", 
            "workspaceworkingdirectory",
            "workspaceartifactstore",
            "workspaceblobstore"  # Default datastore - update last
        ]

        # Update each datastore to use Managed Identity
        updated_count = 0
        for datastore_name in datastore_names:
            try:
                print(f"\nProcessing datastore: {datastore_name}")
                datastore = ml_client.datastores.get(datastore_name)
                
                if datastore and isinstance(datastore, (AzureBlobDatastore, AzureFileDatastore)):
                    print(f"  - Updating to use managed identity...")
                    
                    # Check if this is a default datastore
                    is_default = getattr(datastore, 'is_default', False)
                    
                    # Update credentials
                    datastore.credentials = NoneCredentialConfiguration()
                    
                    # Preserve the is_default property if it exists
                    if hasattr(datastore, 'is_default'):
                        datastore.is_default = is_default
                    
                    ml_client.datastores.create_or_update(datastore)
                    
                    # For default blob datastores, re-set as default after update
                    if datastore_name in ["workspaceblobstore", "workspaceartifactstore"]:
                        print(f"  - Setting as default datastore...")
                        blob_datastore = Datastore.get(workspace, datastore_name)
                        blob_datastore.set_as_default()
                    
                    print(f"  ✓ Updated successfully")
                    updated_count += 1
                else:
                    print(f"  - Skipping (not a supported datastore type)")
                    
            except Exception as e:
                error_msg = str(e)
                if "Could not find datastore" in error_msg or "does not exist" in error_msg:
                    print(f"  - Skipping (datastore does not exist)")
                else:
                    print(f"  ⚠ Warning: Failed to update - {e}")

        print(f"\n✓ Datastore update completed")
        print(f"  - Updated {updated_count} datastore(s) to use managed identity")
        
    except Exception as e:
        print(f"\n✗ Datastore update failed: {e}")
        print("\nYou can manually update datastores from Azure Portal:")
        print(f"  1. Navigate to workspace: {workspace_name}")
        print(f"  2. Go to 'Data' -> 'Datastores'")
        print(f"  3. Select each datastore and edit authentication to use 'Identity-based access'")
        raise


def provision_managed_network(subscription_id, resource_group, workspace_name, include_spark=False):
    """
    Provision the managed virtual network for an Azure ML workspace.
    
    This activates the managed network and outbound rules for the workspace.
    The process can take 5-10 minutes to complete.
    
    Args:
        subscription_id: Azure subscription ID
        resource_group: Name of the resource group
        workspace_name: Name of the Azure ML workspace
        include_spark: Whether to include Spark in the managed network
        
    Returns:
        True if provisioning was initiated successfully, False otherwise
    """
    
    print("=" * 60)
    print("PROVISIONING: Azure ML Managed Network")
    print("=" * 60)
    print(f"Workspace: {workspace_name}")
    print(f"Resource Group: {resource_group}")
    
    try:
        # Authenticate with Azure
        credential = AzureCliCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )

        print("\nProvisioning managed network...")
        print("This will activate the outbound rules and may take 5-10 minutes...")
        
        # Start provisioning
        provision_result = ml_client.workspaces.begin_provision_network(
            workspace_name=workspace_name,
            include_spark=include_spark
        ).result()
        
        print(f"\n✓ Managed network provisioned successfully")
        print(f"  - Status: Active")
        print(f"  - Required outbound rules are now active")
        
        # Get and display network profile
        workspace = ml_client.workspaces.get(workspace_name)
        network_profile = workspace.managed_network
        
        if network_profile and hasattr(network_profile, 'outbound_rules'):
            print(f"\nOutbound rules:")
            for rule in network_profile.outbound_rules:
                print(f"  - {rule.name}: {rule.type} (Status: {rule.status})")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Managed network provisioning failed: {e}")
        print("\nYou can manually provision the managed network from Azure Portal:")
        print(f"  1. Navigate to workspace: {workspace_name}")
        print(f"  2. Go to 'Networking' -> 'Workspace managed outbound access'")
        print(f"  3. Click 'Provision' to activate the managed network")
        return False


def _arm_request(method, url, body=None):
    """Make an authenticated request to the Azure Resource Manager REST API.

    Uses the Azure CLI credential (same auth as the rest of this module).
    Returns the parsed JSON response (or {} for empty bodies).
    Raises on HTTP errors, surfacing the ARM error body for diagnostics.
    """
    import json as _json
    import urllib.request
    import urllib.error

    token = AzureCliCredential().get_token("https://management.azure.com/.default").token
    data = _json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return _json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ARM {method} failed ({e.code}): {detail}") from None


def add_aml_managed_private_endpoint_rule(
    subscription_id, resource_group, workspace_name,
    rule_name, target_resource_id, subresource_target,
):
    """Add a managed-network outbound PRIVATE ENDPOINT rule to an AML workspace.

    Required so the AML processor can reach a dependency (e.g. Azure OpenAI or
    Azure AI Search) after that dependency's public network access is disabled.

    The workspace's managed identity must hold a role that can approve private
    endpoint connections on the target (e.g. "Azure AI Enterprise Network
    Connection Approver"). That role is assigned by NETWORK.Template.json; if it
    was just granted, allow a few minutes for RBAC propagation before this
    succeeds (retry on a 400 "does not have required permissions" error).

    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group of the workspace
        workspace_name: AML workspace name
        rule_name: Name for the outbound rule (e.g. "pe-aoai-deepwiki")
        target_resource_id: Full ARM resource ID of the target service
        subresource_target: Private-link sub-resource (e.g. "account" for
            OpenAI/Cognitive Services, "searchService" for AI Search)

    Returns:
        The created/updated outbound rule object.
    """
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices"
        f"/workspaces/{workspace_name}/outboundRules/{rule_name}?api-version=2024-10-01"
    )
    body = {
        "properties": {
            "type": "PrivateEndpoint",
            "category": "UserDefined",
            "destination": {
                "serviceResourceId": target_resource_id,
                "subresourceTarget": subresource_target,
                "sparkEnabled": False,
            },
        }
    }
    return _arm_request("PUT", url, body)


def set_openai_private_network(subscription_id, resource_group, openai_resource_name, enabled):
    """Toggle private networking on an existing Azure OpenAI account.

    When ``enabled`` is True, sets ``publicNetworkAccess=Disabled`` AND
    ``restrictOutboundNetworkAccess=true`` (outbound DLP) in a single PATCH.
    Both are required together: the CloudGov_DLP_AzOpenAI policy rejects
    disabling public access unless outbound DLP is also enabled.

    Use this for the case where the OpenAI account already exists
    (``is_creating_open_ai_endpoint = False``), so the AOAI ARM template that
    would otherwise apply these settings is skipped.

    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group of the OpenAI account
        openai_resource_name: Name of the Cognitive Services / OpenAI account
        enabled: True to lock down to private; False to re-open public access
    """
    url = (
        f"https://management.azure.com/subscriptions/{subscription_id}"
        f"/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices"
        f"/accounts/{openai_resource_name}?api-version=2024-10-01"
    )
    if enabled:
        properties = {
            "publicNetworkAccess": "Disabled",
            "restrictOutboundNetworkAccess": True,
            "allowedFqdnList": [],
            "networkAcls": {"defaultAction": "Deny"},
        }
    else:
        properties = {
            "publicNetworkAccess": "Enabled",
            "restrictOutboundNetworkAccess": False,
            "networkAcls": {"defaultAction": "Allow"},
        }
    return _arm_request("PATCH", url, {"properties": properties})
