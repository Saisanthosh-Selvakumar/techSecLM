from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(system_prompt, user_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

def safe_parse(text):
    try:
        return json.loads(text)
    except:
        return None


def domain_agent(text):
    prompt = """
Return JSON in this format:
{
  "domains": ["..."]
}

Choose from:
Healthcare, Biotechnology, AI, Cybersecurity, IoT, Genetics,
Medical Devices, Data Privacy, Software Engineering, General

Rules:
- Always return at least 1 domain
"""
    result = call_llm(prompt, text)
    data = safe_parse(result)

    if data and "domains" in data:
        return data["domains"]

    return ["General"]


def security_agent(text):
    prompt = """
Return JSON in this format:
{
  "security_findings": ["..."]
}

Rules:
- Always return at least 1 item
- Focus on data leaks, wireless risks, device vulnerabilities, access risks
"""
    result = call_llm(prompt, text)
    data = safe_parse(result)

    if data and "security_findings" in data:
        return data["security_findings"]

    return ["No major security risks detected"]


def compliance_agent(text):
    prompt = """
Return JSON in this format:
{
  "compliance_issues": ["..."]
}

Rules:
- Always return at least 1 item
- Focus on HIPAA, patient data, privacy, regulation
"""
    result = call_llm(prompt, text)
    data = safe_parse(result)

    if data and "compliance_issues" in data:
        return data["compliance_issues"]

    return ["No compliance issues detected"]


def risk_agent(security, compliance):

    prompt = f"""
Return JSON in this format:
{
  "score": number (0-10),
  "severity": "LOW" | "MEDIUM" | "HIGH",
  "reason": "short explanation"
}

Evaluate realistically based on:

Security:
{security}

Compliance:
{compliance}

Rules:
- HIGH if medical data + system exposure exists
- MEDIUM if moderate issues
- LOW if minimal issues
"""

    result = call_llm(prompt, "")
    data = safe_parse(result)

    if data:
        return data

    return {
        "score": 5,
        "severity": "MEDIUM",
        "reason": "Fallback due to parsing issue"
    }


def run_analysis(text):

    domains = domain_agent(text)
    security = security_agent(text)
    compliance = compliance_agent(text)
    risk = risk_agent(security, compliance)

    return {
        "analysis": {
            "domains": domains,
            "security_findings": security,
            "compliance_issues": compliance
        },
        "risk_assessment": {
            "score": risk.get("score", 5),
            "severity": risk.get("severity", "MEDIUM"),
            "reason": risk.get("reason", "")
        },
        "meta": {
            "agent_version": "v2-multi-agent-llm"
        }
    }