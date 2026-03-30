"""
K8s SRE Agent — Claude-powered autonomous Kubernetes operations agent.
"""
import json
import os

import anthropic

from agentic_ai.tools.k8s_tools import TOOL_DEFINITIONS, execute_tool
from mlops_poc.logging import logger

SYSTEM_PROMPT = """You are an expert Kubernetes SRE and MLOps specialist with deep knowledge of:
- Kubernetes cluster operations, troubleshooting, and optimization
- ML model monitoring, retraining, and deployment
- Anomaly detection and incident response

Your responsibilities:
- MONITOR: Check cluster health and detect anomalies
- DIAGNOSE: Identify root causes from metrics and alerts
- REMEDIATE: Take corrective actions (restart, scale, retrain)
- REPORT: Provide clear, actionable incident summaries

Always:
1. Start by gathering current cluster state
2. Check recent alerts for patterns
3. Use ML anomaly detection to validate concerns
4. Propose or take remediation steps
5. Verify recovery after actions
6. Summarize findings clearly

Be decisive but explain your reasoning. Escalate to humans when unsure."""


class K8sAgent:
    def __init__(self, api_key: str | None = None, max_iterations: int = 10):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.max_iterations = max_iterations
        self.model = "claude-opus-4-5"

    def run(self, user_message: str) -> str:
        logger.info("Agent task: %s", user_message)
        messages = [{"role": "user", "content": user_message}]
        iterations = 0

        while iterations < self.max_iterations:
            iterations += 1
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            logger.info("Agent iteration %d: stop_reason=%s", iterations, response.stop_reason)

            if response.stop_reason == "end_turn":
                text = next((b.text for b in response.content if hasattr(b, "text")), "")
                return text

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info("Tool call: %s(%s)", block.name, json.dumps(block.input)[:100])
                        try:
                            result = execute_tool(block.name, block.input)
                        except Exception as e:
                            result = {"error": str(e)}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached. Please review the cluster manually."

    def run_incident_response(self, namespace: str = "default") -> str:
        return self.run(
            f"Run a complete incident response for the '{namespace}' namespace. "
            "Check cluster health, detect anomalies, identify any issues, "
            "propose remediations, and provide a full incident report."
        )


def interactive_cli():
    print("\n" + "=" * 60)
    print("  K8s SRE Agent — Powered by Claude")
    print("=" * 60)
    print("Describe your K8s issue or ask anything about the cluster.")
    print("Commands: 'incident' = full incident response, 'quit' = exit\n")

    agent = K8sAgent()
    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "incident":
                user_input = "Run a full incident response for the default namespace"
            print("\nAgent > Thinking...\n")
            response = agent.run(user_input)
            print(f"Agent > {response}\n")
        except KeyboardInterrupt:
            break

    print("\nAgent session ended.")


if __name__ == "__main__":
    interactive_cli()
