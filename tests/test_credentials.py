"""Firestore credentials: JSON key (secret or file) and Workload Identity Federation (external_account)."""

import json
import urllib.parse

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.auth import identity_pool
from google.oauth2 import service_account

from analysis.firestore import SCOPES, load_credentials

SA = "analysis@meinungspuls-hu.iam.gserviceaccount.com"
PROVIDER = "projects/123456/locations/global/workloadIdentityPools/github/providers/meinungspuls"
OIDC_URL = "https://pipelines.actions.githubusercontent.example/token?api-version=2.0"


@pytest.fixture
def key_info():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return {
        "type": "service_account",
        "project_id": "meinungspuls-hu",
        "private_key_id": "abc",
        "private_key": pem.decode(),
        "client_email": SA,
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    return monkeypatch


def _wif_file(tmp_path):
    """The file google-github-actions/auth writes (credential_source = the runner's OIDC token endpoint)."""
    info = {
        "type": "external_account",
        "audience": f"//iam.googleapis.com/{PROVIDER}",
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {
            "url": OIDC_URL + "&audience=" + urllib.parse.quote(f"https://iam.googleapis.com/{PROVIDER}"),
            "headers": {"Authorization": "Bearer request-token"},
            "format": {"type": "json", "subject_token_field_name": "value"},
        },
        "service_account_impersonation_url": (
            f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{SA}:generateAccessToken"
        ),
    }
    path = tmp_path / "gha-creds-0123456789abcdef.json"
    path.write_text(json.dumps(info))
    return path


def test_secret_json_key_wins(clean_env, key_info, tmp_path):
    clean_env.setenv("FIREBASE_SERVICE_ACCOUNT", json.dumps(key_info))
    clean_env.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(_wif_file(tmp_path)))
    creds = load_credentials()
    assert isinstance(creds, service_account.Credentials)
    assert creds.service_account_email == SA and list(creds.scopes) == SCOPES


def test_key_file(clean_env, key_info, tmp_path):
    path = tmp_path / "sa.json"
    path.write_text(json.dumps(key_info))
    clean_env.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(path))
    creds = load_credentials()
    assert isinstance(creds, service_account.Credentials) and list(creds.scopes) == SCOPES


def test_key_with_escaped_newlines(clean_env, key_info):
    key_info["private_key"] = key_info["private_key"].replace("\n", "\\n")
    clean_env.setenv("FIREBASE_SERVICE_ACCOUNT", json.dumps(key_info))
    assert isinstance(load_credentials(), service_account.Credentials)


def test_no_credentials(clean_env):
    with pytest.raises(RuntimeError, match="Set FIREBASE_SERVICE_ACCOUNT"):
        load_credentials()


class _Resp:
    def __init__(self, status, payload):
        self.status = status
        self.data = json.dumps(payload).encode()
        self.headers = {"content-type": "application/json"}


def test_workload_identity_federation(clean_env, tmp_path):
    clean_env.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(_wif_file(tmp_path)))
    creds = load_credentials()
    assert isinstance(creds, identity_pool.Credentials)
    assert creds.service_account_email == SA

    calls = []

    def request(url, method="GET", body=None, headers=None, **kwargs):
        calls.append((method, url, body, headers or {}))
        if url.startswith(OIDC_URL):
            return _Resp(200, {"value": "github-oidc-jwt"})
        if url == "https://sts.googleapis.com/v1/token":
            return _Resp(
                200,
                {
                    "access_token": "federated",
                    "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        if url.startswith("https://iamcredentials.googleapis.com/"):
            return _Resp(200, {"accessToken": "sa-token", "expireTime": "2099-01-01T00:00:00Z"})
        raise AssertionError(url)

    creds.refresh(request)
    assert creds.token == "sa-token"
    (_, _, _, oidc_headers), (_, _, sts_body, _), (_, _, iam_body, iam_headers) = calls
    assert oidc_headers["Authorization"] == "Bearer request-token"
    sts = urllib.parse.parse_qs(sts_body.decode() if isinstance(sts_body, bytes) else sts_body)
    assert sts["subject_token"] == ["github-oidc-jwt"] and sts["audience"] == [f"//iam.googleapis.com/{PROVIDER}"]
    assert json.loads(iam_body)["scope"] == SCOPES
    assert iam_headers.get("Authorization", iam_headers.get("authorization")) == "Bearer federated"
