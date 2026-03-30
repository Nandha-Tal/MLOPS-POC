"""
MCP Server — K8s Monitoring Tools for Claude Desktop
=====================================================
Exposes K8s monitoring tools, prompts, and resources via the Model Context Protocol.

Claude Desktop config (~/.claude/claude_desktop_config.json):
{
  "mcpServers": {
    "k8s-monitoring": {
      "command": "python3",
      "args": ["/Users/nandak/Downloads/MLOPS-poc/mcp_server/server.py"],
      "env": { "API_BASE_URL": "http://localhost:8080" }
    }
  }
}
"""
import asyncio
import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult, ListPromptsResult, ListResourcesResult,
    ListToolsResult, Prompt, PromptArgument, PromptMessage,
    ReadResourceResult, Resource, TextContent, Tool,
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8080")
server = Server("k8s-monitoring")


def api_get(path: str) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=10) as client:
        return client.get(path).json()


def api_post(path: str, data: dict = None) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=10) as client:
        return client.post(path, json=data or {}).json()


@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[
        Tool(name="get_cluster_health",   description="Get K8s cluster health and info",         inputSchema={"type": "object", "properties": {}}),
        Tool(name="detect_anomaly",       description="Run anomaly detection on current metrics", inputSchema={"type": "object", "properties": {"force_anomaly": {"type": "boolean"}}}),
        Tool(name="get_alert_history",    description="Get recent alert history",                  inputSchema={"type": "object", "properties": {"limit": {"type": "integer"}}}),
        Tool(name="get_pipeline_status",  description="Get ML pipeline status",                   inputSchema={"type": "object", "properties": {}}),
        Tool(name="run_pipeline_stage",   description="Trigger pipeline stage 1-4",               inputSchema={"type": "object", "properties": {"stage": {"type": "integer"}}, "required": ["stage"]}),
        Tool(name="get_model_metrics",    description="Get model evaluation metrics",             inputSchema={"type": "object", "properties": {}}),
        Tool(name="get_current_metrics",  description="Get current cluster metrics snapshot",    inputSchema={"type": "object", "properties": {}}),
        Tool(name="check_service_health", description="Check API liveness and readiness",        inputSchema={"type": "object", "properties": {}}),
    ])


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "get_cluster_health":
            result = api_get("/health/info")
        elif name == "detect_anomaly":
            result = api_post("/alerts/simulate", {"force_anomaly": arguments.get("force_anomaly", False)})
        elif name == "get_alert_history":
            result = api_get(f"/alerts/history?limit={arguments.get('limit', 20)}")
        elif name == "get_pipeline_status":
            result = api_get("/pipeline/status")
        elif name == "run_pipeline_stage":
            result = api_post(f"/pipeline/run/{arguments['stage']}")
        elif name == "get_model_metrics":
            result = api_get("/pipeline/metrics")
        elif name == "get_current_metrics":
            result = api_get("/metrics/current")
        elif name == "check_service_health":
            result = {"live": api_get("/health/live"), "ready": api_get("/health/ready")}
        else:
            result = {"error": f"Unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


@server.list_prompts()
async def list_prompts() -> ListPromptsResult:
    return ListPromptsResult(prompts=[
        Prompt(name="k8s-incident-response", description="Structured K8s incident investigation",
               arguments=[PromptArgument(name="namespace", description="K8s namespace", required=False)]),
        Prompt(name="k8s-daily-standup",     description="Generate cluster health standup report", arguments=[]),
        Prompt(name="k8s-retrain-model",     description="Guided anomaly detector retraining",     arguments=[]),
        Prompt(name="k8s-explain-anomaly",   description="Explain anomaly score in plain language",
               arguments=[PromptArgument(name="anomaly_score", description="Anomaly score value", required=True)]),
    ])


@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    if name == "k8s-incident-response":
        ns = arguments.get("namespace", "default")
        content = f"""You are a K8s SRE. Investigate the '{ns}' namespace:
1. Check cluster health with get_cluster_health
2. Run anomaly detection with detect_anomaly
3. Review alert history with get_alert_history
4. Identify root cause and propose remediation
5. Write a structured incident report"""
    elif name == "k8s-daily-standup":
        content = """Generate a daily K8s cluster standup report:
1. Check cluster health summary
2. Review any anomalies in the last 24h (get_alert_history)
3. Check model performance metrics (get_model_metrics)
4. Provide recommendations for the day"""
    elif name == "k8s-retrain-model":
        content = """Retrain the anomaly detector with fresh data:
1. Check current model metrics (get_model_metrics)
2. Run stage 1 — data ingestion (run_pipeline_stage stage=1)
3. Run stage 2 — prepare model (run_pipeline_stage stage=2)
4. Run stage 3 — train (run_pipeline_stage stage=3)
5. Run stage 4 — evaluate (run_pipeline_stage stage=4)
6. Compare metrics before and after retraining"""
    elif name == "k8s-explain-anomaly":
        score = arguments.get("anomaly_score", "unknown")
        content = f"""Explain what anomaly score {score} means for cluster health:
- What does this IsolationForest score indicate?
- Which cluster components are likely affected?
- What severity level does this represent (OK/WARNING/CRITICAL)?
- What should the on-call engineer check first?
- What remediation steps are recommended?"""
    else:
        content = f"Unknown prompt: {name}"

    return GetPromptResult(messages=[
        PromptMessage(role="user", content=TextContent(type="text", text=content))
    ])


@server.list_resources()
async def list_resources() -> ListResourcesResult:
    return ListResourcesResult(resources=[
        Resource(uri="k8s://cluster/health", name="Cluster Health",  mimeType="application/json"),
        Resource(uri="k8s://model/metrics",  name="Model Metrics",   mimeType="application/json"),
        Resource(uri="k8s://alerts/recent",  name="Recent Alerts",   mimeType="application/json"),
    ])


@server.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    if uri == "k8s://cluster/health":
        data = api_get("/health/info")
    elif uri == "k8s://model/metrics":
        data = api_get("/pipeline/metrics")
    elif uri == "k8s://alerts/recent":
        data = api_get("/alerts/history?limit=10")
    else:
        data = {"error": f"Unknown resource: {uri}"}
    return ReadResourceResult(contents=[TextContent(type="text", text=json.dumps(data, indent=2))])


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
