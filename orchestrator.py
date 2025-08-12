#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Literal, Optional, Tuple, NoReturn
from urllib.parse import urlparse

import typer
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# ----------------------------
# CLI
# ----------------------------

app = typer.Typer(no_args_is_help=True, add_completion=False)


# ----------------------------
# Models
# ----------------------------

class AgentProvider(StrEnum):
    pydanticai = "pydanticai"
    crew_ai = "crew_ai"
    langchain = "langchain"


class ModelProvider(StrEnum):
    openai = "openai"
    anthropic = "anthropic"
    azure_openai = "azure_openai"
    gemini = "gemini"


class ProxyKind(StrEnum):
    mitmproxy = "mitmproxy"
    none = "none"


class ProxyCfg(BaseModel):
    enabled: bool = True
    kind: ProxyKind = ProxyKind.mitmproxy
    listen_host: str = "127.0.0.1"
    listen_port: int = 8002


class TracingCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7000


class ClientCfg(BaseModel):
    cmd: list[str] = Field(...)
    args: list[str] = Field(default_factory=list)


class Config(BaseModel):
    agent_provider: AgentProvider = AgentProvider.pydanticai
    model_provider: ModelProvider = ModelProvider.openai
    proxy: ProxyCfg = ProxyCfg()
    tracing_api: TracingCfg = TracingCfg()
    client: ClientCfg

    # Flattened MCP variant choices (selected values)
    mcp_server_variant: str = "stdio"
    mcp_client_variant: str = "stdio"


class Lookups(BaseModel):
    # e.g., {"openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com"}
    model_reverse_target: dict[str, str]
    # Per agent, variants for servers and clients
    # {
    #   "pydanticai": {
    #       "servers": {"stdio": {"type":"stdio","cmd":[...]}, "http": {"type":"http","url":"...","cmd":[...] }},
    #       "clients": {"stdio": {"type":"stdio","client_path":"..."}, "http": {"type":"http","client_path":"..."}}
    #   },
    #   "crewai": {...}
    # }
    mcp: dict[str, dict[str, dict[str, dict[str, object]]]]


# ----------------------------
# Utilities
# ----------------------------

def die(msg: str) -> NoReturn:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise SystemExit(1)


def _deep_merge(a: dict, b: dict) -> None:
    """Deep-merge b into a."""
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v


def wait_port(host: str, port: int, timeout: float = 20.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.settimeout(0.5)
            try:
                if s.connect_ex((host, port)) == 0:
                    return True
            except OSError:
                pass
        time.sleep(0.2)
    return False


def is_host_only_url(url: str) -> bool:
    """Return True if URL is scheme+host (no path/query/fragment)."""
    u = urlparse(url)
    if not u.scheme or not u.netloc:
        return False
    return (u.path in ("", "/")) and (not u.params) and (not u.query) and (not u.fragment)


# ----------------------------
# Loading config
# ----------------------------

def load_all(config_path: str, profile: Optional[str]) -> Tuple[Config, Lookups, dict]:
    raw = yaml.safe_load(Path(config_path).read_text())

    # Start with defaults
    cfg_dict = dict(raw.get("defaults", {}))

    # Apply profile overrides (if any)
    if profile:
        profiles = raw.get("profiles", {})
        prof = profiles.get(profile)
        if not prof:
            die(f"Profile '{profile}' not found in {config_path}")
        _deep_merge(cfg_dict, prof)

    # If nested mcp keys exist (e.g., {"mcp":{"server_variant":"http","client_variant":"http"}}), flatten
    mcp_nested = cfg_dict.pop("mcp", None)
    if isinstance(mcp_nested, dict):
        if "server_variant" in mcp_nested:
            cfg_dict["mcp_server_variant"] = mcp_nested["server_variant"]
        if "client_variant" in mcp_nested:
            cfg_dict["mcp_client_variant"] = mcp_nested["client_variant"]

    lookups_raw = raw.get("lookups", {})

    try:
        lookups = Lookups.model_validate(lookups_raw)
    except ValidationError as e:
        die(f"Invalid 'lookups' config: {e}")


    cfg_dict["client"] = lookups.mcp[cfg_dict["agent_provider"]]["clients"][cfg_dict["mcp_client_variant"]]

    try:
        cfg = Config.model_validate(cfg_dict)
    except ValidationError as e:
        die(f"Invalid 'defaults/profiles' config: {e}")

    return cfg, lookups, raw


# ----------------------------
# Process starters
# ----------------------------

def start_tracing_api(cfg: TracingCfg, env: dict) -> subprocess.Popen:
    # Example assumes uvicorn module path trace_api.app:app exists in your repo
    cmd = [
        "uv",
        "run",
        "uvicorn",
        "src.api.api:app",
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
    ]
    return subprocess.Popen(cmd, env=env)


def start_proxy(kind: ProxyKind, listen_host: str, listen_port: int, reverse_target: str, env: dict) -> Optional[
    subprocess.Popen]:
    if kind == "none":
        return None

    if kind == "mitmproxy":
        # Reverse mode: the target MUST be scheme+host only (no path)
        if not is_host_only_url(reverse_target):
            die("Proxy reverse_target must be base host (e.g., https://api.openai.com), not a path.")
        cmd = [
            "mitmdump",
            "-s",
            "src/proxy/reverse_logger.py",
            "--mode",
            f"reverse:{reverse_target}",
            "--listen-host",
            listen_host,
            "--listen-port",
            str(listen_port),
        ]
        return subprocess.Popen(cmd, env=env)

    die(f"Unsupported proxy kind: {kind}")
    return None


def run_command(cmd: list[str], env: dict) -> subprocess.Popen:
    """Run a command and return the process handle."""
    return subprocess.Popen(cmd, env=env)


def run_client(client: ClientCfg, env: dict) -> int:
    proc = subprocess.Popen(client.cmd + client.args, env=env)
    return proc.wait()


def shutdown_all(procs: list[subprocess.Popen]) -> None:
    # Terminate in reverse order
    for p in procs[::-1]:
        if p and p.poll() is None:
            p.terminate()
    # Wait then kill if needed
    for p in procs[::-1]:
        if p and p.poll() is None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def build_env(base_env: dict, cfg: Config) -> dict:
    env = base_env.copy()
    # Knobs your client/agent can read
    env["AGENT_PROVIDER"] = cfg.agent_provider
    env["MODEL_PROVIDER"] = cfg.model_provider
    env["TRACING_API"] = f"http://{cfg.tracing_api.host}:{cfg.tracing_api.port}"
    # Proxy for HTTP(S) if enabled; do not override if already set outside
    if cfg.proxy.enabled and cfg.proxy.kind != "none":
        proxy_url = f"http://{cfg.proxy.listen_host}:{cfg.proxy.listen_port}"
        env.setdefault("BASE_URL", proxy_url)
    return env


# ----------------------------
# Command
# ----------------------------

@app.command()
def up(
        profile: Optional[str] = typer.Option(None, help="Profile name from orchestrator.yaml"),
        config_path: str = typer.Option("orchestrator.yaml", help="Path to config file"),
        env_file: str = typer.Option(".env", help="Path to .env file with secrets"),
        # Optional overrides:
        agent: Optional[AgentProvider] = typer.Option(None, help="Override agent provider"),
        model: Optional[ModelProvider] = typer.Option(None, help="Override model provider"),
        mcp_server_variant: Optional[str] = typer.Option(None, help="Override MCP server variant (e.g., stdio|http)"),
        mcp_client_variant: Optional[str] = typer.Option(None, help="Override MCP client variant (e.g., stdio|http)"),
):
    # Load secrets/config from .env into process environment
    load_dotenv(env_file, override=False)

    cfg, lookups, _raw = load_all(config_path, profile)

    # CLI overrides
    if agent:
        cfg.agent_provider = agent
    if model:
        cfg.model_provider = model
    if mcp_server_variant:
        cfg.mcp_server_variant = mcp_server_variant
    if mcp_client_variant:
        cfg.mcp_client_variant = mcp_client_variant

    # Derive reverse proxy base from selected model provider
    reverse_target = lookups.model_reverse_target.get(cfg.model_provider)
    if cfg.proxy.enabled:
        if not reverse_target:
            die(f"No reverse target defined for model_provider '{cfg.model_provider}' in lookups.model_reverse_target")

    # Resolve MCP variants for the selected agent provider
    agent_key = cfg.agent_provider
    agent_mcp = lookups.mcp.get(agent_key)
    if not agent_mcp:
        die(f"No MCP lookups for agent '{agent_key}' in lookups.mcp")

    server_def = (agent_mcp.get("servers") or {}).get(cfg.mcp_server_variant)
    client_def = (agent_mcp.get("clients") or {}).get(cfg.mcp_client_variant)

    if not server_def:
        die(f"No MCP server variant '{cfg.mcp_server_variant}' for agent '{agent_key}'")
    if not client_def:
        die(f"No MCP client variant '{cfg.mcp_client_variant}' for agent '{agent_key}'")

    # Validate server/client transports align
    server_type = str(server_def.get("type"))
    client_type = str(client_def.get("type"))
    if server_type != client_type:
        die(
            f"MCP server/client transport mismatch: server='{server_type}' vs client='{client_type}'. "
            f"Pick matching variants."
        )

    # Prepare environment for downstream processes
    env = build_env(os.environ, cfg)
    env["MCP_AGENT_PROVIDER"] = cfg.agent_provider
    env["MCP_SERVER_VARIANT"] = cfg.mcp_server_variant
    env["MCP_CLIENT_VARIANT"] = cfg.mcp_client_variant
    env["MCP_TRANSPORT"] = server_type  # stdio | http

    # Client needs to know how to connect
    client_cfg_path = client_def.get("client_path")
    if client_cfg_path:
        env["MCP_CLIENT_CONFIG"] = str(client_cfg_path)

    http_url = None
    if server_type == "http":
        http_url = server_def.get("url")
        if not http_url:
            die("HTTP MCP server requires a 'url' field in lookups.mcp.<agent>.servers.<variant>")
        env["MCP_SERVER_URL"] = str(http_url)

    # Lifecycle management
    procs: list[subprocess.Popen] = []

    def _shutdown_and_exit(code: int) -> None:
        shutdown_all(procs)
        raise SystemExit(code)

    def _handle_sig(*_):
        _shutdown_and_exit(130)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # 1) tracing api
    tproc = start_tracing_api(cfg.tracing_api, env)
    procs.append(tproc)
    if not wait_port(cfg.tracing_api.host, cfg.tracing_api.port, timeout=20):
        die("Tracing API failed to start or open its port")

    # 2) proxy (optional)
    if cfg.proxy.enabled and cfg.proxy.kind != "none":
        pproc = start_proxy(cfg.proxy.kind, cfg.proxy.listen_host, cfg.proxy.listen_port, reverse_target, env)
        if pproc:
            procs.append(pproc)
            if not wait_port(cfg.proxy.listen_host, cfg.proxy.listen_port, timeout=20):
                die("Proxy failed to start or open its port")

    # 3) MCP server (variant dependent)
    if server_type == "stdio":
        if server_def.get("cmd"):
            mproc = run_command(list(server_def.get("cmd", [])), env)
            procs.append(mproc)
            # light grace period; for robust readiness, implement a ping if your server supports it
            time.sleep(0.5)
    elif server_type == "http":
        if not http_url:
            die("Internal: missing http_url after server_type=='http'")
        mproc = run_command(list(server_def.get("cmd", [])), env)
        procs.append(mproc)
        u = urlparse(http_url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            die(f"Invalid MCP HTTP URL: {http_url}")
        if not wait_port(host, port, timeout=25):
            die(f"MCP HTTP server failed to open {host}:{port}")
    else:
        die(f"Unsupported MCP server type: {server_type}")

    # 4) client (blocks until completion)
    exit_code = run_client(cfg.client, env)
    _shutdown_and_exit(exit_code)


# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":
    app()
