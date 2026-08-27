from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NPM_REGISTRY = "https://packagefeedproxy.microsoft.io/npm/"
PYPI_REGISTRY = "https://packagefeedproxy.microsoft.io/pypi/simple/"


def test_web_dockerfile_uses_organization_package_proxies():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert f'ARG NPM_REGISTRY="{NPM_REGISTRY}"' in dockerfile
    assert f'ARG PYPI_REGISTRY="{PYPI_REGISTRY}"' in dockerfile
    assert "npm config set replace-registry-host always" in dockerfile
    assert "poetry-plugin-pypi-mirror==${POETRY_MIRROR_PLUGIN_VERSION}" in dockerfile
    assert 'POETRY_PYPI_MIRROR_URL="${PYPI_REGISTRY}"' in dockerfile


def test_processor_dockerfile_uses_python_package_proxy():
    dockerfile = (
        ROOT / "Dockerfile.processor"
    ).read_text(encoding="utf-8")

    assert f'ARG PYPI_REGISTRY="{PYPI_REGISTRY}"' in dockerfile
    assert "poetry-plugin-pypi-mirror==${POETRY_MIRROR_PLUGIN_VERSION}" in dockerfile
    assert 'POETRY_PYPI_MIRROR_URL="${PYPI_REGISTRY}"' in dockerfile


def test_publish_script_passes_registry_build_arguments():
    script = (ROOT / "publish-web.ps1").read_text(encoding="utf-8")

    assert NPM_REGISTRY in script
    assert PYPI_REGISTRY in script
    assert '--build-arg "NPM_REGISTRY=$NPM_REGISTRY"' in script
    assert '--build-arg "PYPI_REGISTRY=$PYPI_REGISTRY"' in script
