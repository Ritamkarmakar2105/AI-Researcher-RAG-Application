import gradio as gr
from groq import Groq
from duckduckgo_search import DDGS

import json
import os
import time

from typing import Optional, List, Any, Dict
 
GROQ_API_KEY = "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"   # <-- CHANGE THIS

MODEL_NAME = "openai/gpt-oss-120b"

# ==============================
# 2. Groq + LLM helper
# ==============================

client = Groq(api_key=GROQ_API_KEY)


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
    """
    Call Groq, ask for JSON, but parse it ourselves.
    If parsing fails, return {'raw_output': content} instead of crashing.
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content

    # Try direct JSON parse
    try:
        return json.loads(content)
    except Exception:
        pass

    # Try to grab first {...} block
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start: end + 1]
            return json.loads(json_str)
    except Exception:
        pass

    # Fallback: just give raw output
    print("\n[WARN] JSON parse failed, raw output:\n", content[:400], "...\n")
    return {"raw_output": content}


def safe_get(d: dict, key: str, default):
    try:
        return d.get(key, default)
    except Exception:
        return default


# ==============================
# 3. Simple "RAG": combine notes + file text
# ==============================

def read_file_text(f) -> str:
    """
    Gradio can pass a file path (str) OR a file-like object.
    This handles both safely.
    """
    try:
        if isinstance(f, str):
            with open(f, "rb") as fh:
                return fh.read().decode("utf-8", errors="ignore")
        else:
            data = f.read()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="ignore")
            return str(data)
    except Exception as e:
        print(f"[WARN] Failed to read uploaded file: {e}")
        return ""


def build_notes(extra_notes: str, uploaded_files: Optional[List[Any]]) -> str:
    """
    Combine pasted notes + any uploaded .txt files into one context string.
    (Simple version instead of full vector DB RAG – more robust.)
    """
    parts: List[str] = []

    if extra_notes and extra_notes.strip():
        parts.append("### User notes ###\n" + extra_notes.strip())

    if uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        for f in uploaded_files:
            txt = read_file_text(f)
            if not txt.strip():
                continue
            name = getattr(f, "name", None)
            if isinstance(name, str):
                base = os.path.basename(name)
            else:
                base = "uploaded_note.txt"
            parts.append(f"\n### File: {base} ###\n{txt.strip()}")

    return "\n\n".join(parts).strip()


# ==============================
# 4. Agents
# ==============================

def researcher_agent(topic: str, notes: str) -> dict:
    system = """
You are ResearcherAgent.
You read rough notes and external info about a ML topic and extract only
the important, relevant information.

Output JSON ONLY (no extra text):
{
  "topic": string,
  "main_points": [string],
  "important_concepts": [string],
  "notable_sources": [string]
}
"""
    user = f"""
Topic: {topic}

Here are my notes and pasted content (may be empty):

{notes or "(no extra notes provided)"}

Summarize only what's relevant to the topic. Return ONLY JSON.
"""
    return call_llm_json(system, user)


def analyst_agent(topic: str, research: dict) -> dict:
    system = """
You are AnalystAgent.
You analyze the research summary and propose possible technical approaches
with pros/cons, then pick a recommended direction.

Output JSON ONLY:
{
  "key_ideas": [string],
  "approaches": [
    {
      "name": string,
      "description": string,
      "pros": [string],
      "cons": [string]
    }
  ],
  "recommended_direction": string
}
"""
    user = f"""
Topic: {topic}

Research summary:
{json.dumps(research, indent=2, ensure_ascii=False)}

Return ONLY JSON.
"""
    return call_llm_json(system, user)


def planner_agent(topic: str, analysis: dict) -> dict:
    system = """
You are PlannerAgent.
You design a CONCRETE ML experiment plan for this topic.

Output JSON ONLY:
{
  "objective": string,
  "assumptions": [string],
  "datasets": [string],
  "baselines": [string],
  "metrics": [string],
  "phases": [
    {
      "name": string,
      "steps": [
        {"id": int, "description": string, "depends_on": [int]}
      ]
    }
  ],
  "risks": [
    {"description": string, "mitigation": string}
  ]
}
"""
    user = f"""
Topic: {topic}

Analysis:
{json.dumps(analysis, indent=2, ensure_ascii=False)}

Use realistic datasets, baselines, and metrics for this topic.
Return ONLY JSON.
"""
    return call_llm_json(system, user)


def reviewer_agent(topic: str, research: dict, analysis: dict, plan: dict) -> dict:
    system = """
You are ReviewerAgent.
You critique the ML experiment plan.

Output JSON ONLY:
{
  "scores": {
    "coverage": float,
    "relevance": float,
    "coherence": float,
    "actionability": float
  },
  "overall_score": float,
  "issues": [string],
  "suggested_improvements": [string]
}
"""
    user = f"""
Topic: {topic}

Research:
{json.dumps(research, indent=2, ensure_ascii=False)}

Analysis:
{json.dumps(analysis, indent=2, ensure_ascii=False)}

Plan:
{json.dumps(plan, indent=2, ensure_ascii=False)}

Evaluate honestly, then return ONLY JSON.
"""
    return call_llm_json(system, user)


# ==============================
# 5. DuckDuckGo validation
# ==============================

def ddg_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title"),
                    "snippet": r.get("body"),
                    "link": r.get("href"),
                }
            )
    return results


def validate_plan(plan: dict, max_results: int = 3) -> Dict[str, Any]:
    datasets = safe_get(plan, "datasets", [])
    baselines = safe_get(plan, "baselines", [])
    metrics = safe_get(plan, "metrics", [])

    report: Dict[str, Any] = {
        "objective": plan.get("objective"),
        "datasets": {},
        "baselines": {},
        "metrics": {},
    }

    for ds in datasets:
        q = f"{str(ds).split('(')[0].strip()} dataset"
        try:
            res = ddg_search(q, max_results=max_results)
            report["datasets"][ds] = {
                "found": len(res) > 0,
                "query": q,
                "top_result": res[0] if res else None,
            }
        except Exception as e:
            report["datasets"][ds] = {"found": False, "query": q, "error": str(e)}

    for bl in baselines:
        q = f"{str(bl).split('(')[0].strip()} model"
        try:
            res = ddg_search(q, max_results=max_results)
            report["baselines"][bl] = {
                "found": len(res) > 0,
                "query": q,
                "top_result": res[0] if res else None,
            }
        except Exception as e:
            report["baselines"][bl] = {"found": False, "query": q, "error": str(e)}

    for mt in metrics:
        q = f'"{str(mt).split("(")[0].strip()}" metric'
        try:
            res = ddg_search(q, max_results=max_results)
            report["metrics"][mt] = {
                "found": len(res) > 0,
                "query": q,
                "top_result": res[0] if res else None,
            }
        except Exception as e:
            report["metrics"][mt] = {"found": False, "query": q, "error": str(e)}

    return report


# ==============================
# 6. Summaries (dicts -> markdown)
# ==============================

def summarize_research(r: dict) -> str:
    if "raw_output" in r:
        return "Raw output (JSON failed):\n\n" + r["raw_output"]

    topic = r.get("topic", "")
    main_points = safe_get(r, "main_points", [])
    concepts = safe_get(r, "important_concepts", [])
    sources = safe_get(r, "notable_sources", [])

    lines = []
    if topic:
        lines.append(f"### Topic\n{topic}\n")

    if main_points:
        lines.append("### Key Takeaways")
        for p in main_points:
            lines.append(f"- {p}")

    if concepts:
        lines.append("\n### Important Concepts")
        for c in concepts:
            lines.append(f"- {c}")

    if sources:
        lines.append("\n### Notable Sources")
        for s in sources:
            lines.append(f"- {s}")

    return "\n".join(lines) if lines else "No specific research points extracted."


def summarize_analysis(a: dict) -> str:
    if "raw_output" in a:
        return "Raw output (JSON failed):\n\n" + a["raw_output"]

    key_ideas = safe_get(a, "key_ideas", [])
    approaches = safe_get(a, "approaches", [])
    rec = a.get("recommended_direction", "")

    lines = []
    if key_ideas:
        lines.append("### Key Ideas")
        for k in key_ideas:
            lines.append(f"- {k}")

    if approaches:
        lines.append("\n### Approaches")
        for i, ap in enumerate(approaches, start=1):
            name = ap.get("name", f"Approach {i}")
            desc = ap.get("description", "")
            pros = safe_get(ap, "pros", [])
            cons = safe_get(ap, "cons", [])
            lines.append(f"\n**{name}**")
            if desc:
                lines.append(f"- Description: {desc}")
            if pros:
                lines.append("- Pros:")
                for p in pros:
                    lines.append(f"  - ✅ {p}")
            if cons:
                lines.append("- Cons:")
                for c in cons:
                    lines.append(f"  - ⚠️ {c}")

    if rec:
        lines.append("\n### Recommended Direction\n" + rec)

    return "\n".join(lines) if lines else "No analysis produced."


def summarize_plan(p: dict) -> str:
    if "raw_output" in p:
        return "Raw output (JSON failed):\n\n" + p["raw_output"]

    obj = p.get("objective", "")
    assumptions = safe_get(p, "assumptions", [])
    datasets = safe_get(p, "datasets", [])
    baselines = safe_get(p, "baselines", [])
    metrics = safe_get(p, "metrics", [])
    phases = safe_get(p, "phases", [])
    risks = safe_get(p, "risks", [])

    lines = []
    if obj:
        lines.append(f"### Objective\n{obj}\n")

    if assumptions:
        lines.append("### Assumptions")
        for a in assumptions:
            lines.append(f"- {a}")

    if datasets:
        lines.append("\n### Datasets")
        for d in datasets:
            lines.append(f"- {d}")

    if baselines:
        lines.append("\n### Baseline Models")
        for b in baselines:
            lines.append(f"- {b}")

    if metrics:
        lines.append("\n### Metrics")
        for m in metrics:
            lines.append(f"- {m}")

    if phases:
        lines.append("\n### Experiment Phases & Steps")
        for ph in phases:
            pname = ph.get("name", "Phase")
            lines.append(f"\n**{pname}**")
            for st in safe_get(ph, "steps", []):
                sid = st.get("id", "?")
                desc = st.get("description", "")
                deps = safe_get(st, "depends_on", [])
                dep_str = ", ".join(str(d) for d in deps) if deps else "None"
                lines.append(f"- Step {sid}: {desc} (depends on: {dep_str})")

    if risks:
        lines.append("\n### Risks & Mitigations")
        for r in risks:
            desc = r.get("description", "")
            mit = r.get("mitigation", "")
            lines.append(f"- ⚠️ {desc}")
            if mit:
                lines.append(f"  → Mitigation: {mit}")

    return "\n".join(lines) if lines else "No plan generated."


def summarize_review(r: dict) -> str:
    if "raw_output" in r:
        return "Raw output (JSON failed):\n\n" + r["raw_output"]

    scores = safe_get(r, "scores", {})
    overall = r.get("overall_score", None)
    issues = safe_get(r, "issues", [])
    improvements = safe_get(r, "suggested_improvements", [])

    lines = []
    if scores or overall is not None:
        lines.append("### Scores (0–10)")
        if overall is not None:
            try:
                lines.append(f"- Overall: **{float(overall):.2f}**")
            except Exception:
                lines.append(f"- Overall: {overall}")
        for k, v in scores.items():
            try:
                lines.append(f"- {k.capitalize()}: {float(v):.2f}")
            except Exception:
                lines.append(f"- {k.capitalize()}: {v}")

    if issues:
        lines.append("\n### Issues")
        for i in issues:
            lines.append(f"- ⚠️ {i}")

    if improvements:
        lines.append("\n### Suggested Improvements")
        for s in improvements:
            lines.append(f"- 💡 {s}")

    return "\n".join(lines) if lines else "No review information."


def summarize_validation(v: dict) -> str:
    if "raw_output" in v:
        return "Raw output (JSON failed):\n\n" + v["raw_output"]

    objective = v.get("objective", "")
    datasets = safe_get(v, "datasets", {})
    baselines = safe_get(v, "baselines", {})
    metrics = safe_get(v, "metrics", {})

    lines = []
    if objective:
        lines.append(f"### Objective Checked\n{objective}\n")

    def section(name: str, items: dict):
        if not items:
            return
        lines.append(f"\n### {name}")
        for key, info in items.items():
            found = info.get("found", False)
            query = info.get("query", "")
            top = info.get("top_result")
            status = "✅ Found" if found else "❓ Not clearly found"
            lines.append(f"- **{key}** → {status}")
            if query:
                lines.append(f"  - Query used: `{query}`")
            if top:
                title = top.get("title") or "(no title)"
                link = top.get("link") or ""
                snippet = top.get("snippet") or ""
                lines.append(f"  - Top result: {title}")
                if snippet:
                    lines.append(f"    - Snippet: {snippet}")
                if link:
                    lines.append(f"    - Link: {link}")

    section("Datasets", datasets)
    section("Baselines", baselines)
    section("Metrics", metrics)

    return "\n".join(lines) if lines else "No validation data."


# ==============================
# 7. Orchestrator used by the button (with timer)
# ==============================

def run_pipeline(topic: str, extra_notes: str, files: Optional[List[Any]]):
    """
    This is what the orange button calls.
    It NEVER throws (errors go into the first tab).
    Also returns a timer string as 6th output.
    """
    start = time.time()
    try:
        topic = (topic or "").strip()
        if not topic:
            elapsed = time.time() - start
            timer_text = f"⏱️ Run failed (no topic) in {elapsed:.1f} s"
            return ("❌ Please enter a topic.", "", "", "", "", timer_text)

        notes = build_notes(extra_notes or "", files)

        research = researcher_agent(topic, notes)
        analysis = analyst_agent(topic, research)
        plan = planner_agent(topic, analysis)
        review = reviewer_agent(topic, research, analysis, plan)

        # Optional refinement if score low
        try:
            score = float(review.get("overall_score", 0))
        except Exception:
            score = 0.0

        if score < 7.5:
            improved_analysis = {
                "original_analysis": analysis,
                "review_issues": safe_get(review, "issues", []),
                "suggested_improvements": safe_get(review, "suggested_improvements", []),
            }
            plan = planner_agent(topic, improved_analysis)
            review = reviewer_agent(topic, research, improved_analysis, plan)

        validation = validate_plan(plan, max_results=3)

        elapsed = time.time() - start
        timer_text = f"⏱️ Last run: {elapsed:.1f} seconds"

        return (
            summarize_research(research),
            summarize_analysis(analysis),
            summarize_plan(plan),
            summarize_review(review),
            summarize_validation(validation),
            timer_text,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        elapsed = time.time() - start
        timer_text = f"⏱️ Run errored after {elapsed:.1f} seconds"
        msg = f"🔥 Internal error: {e}\n\nCheck your terminal for the full traceback."
        return (msg, "", "", "", "", timer_text)


# ==============================
# 8. Gradio UI – blue & white theme + timer
# ==============================

blue_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="gray",
).set(
    body_background_fill="#ffffff",
)

with gr.Blocks(title="AI Research Assistant") as demo:
    gr.Markdown(
        """
# 🧠 AI Research Assistant – Multi-Agent Experiment Planner

- Enter a **topic** (required)  
- Optionally paste some notes and/or upload `.txt` files  
- Click the button to run: **Research → Analysis → Plan → Review → Validation**
"""
    )

    with gr.Row():
        topic_in = gr.Textbox(
            label="Research topic / problem statement",
            placeholder="e.g. Efficient image-to-video diffusion model for low compute",
        )

    with gr.Row():
        notes_in = gr.Textbox(
            label="Optional extra notes (plain text)",
            placeholder="Any extra context, constraints, or ideas...",
            lines=4,
        )

    with gr.Row():
        files_in = gr.File(
            label="Drag & drop .txt notes (optional)",
            file_count="multiple",
            file_types=["text"],
        )

    # Button row + timer (no scale args)
    with gr.Row():
        run_btn = gr.Button(
            "🚀 Run Research → Analysis → Plan → Validation",
            variant="primary",
        )
        timer_out = gr.Markdown("⏱️ Not run yet", elem_id="timer-box")

    with gr.Tab("Research"):
        research_out = gr.Markdown()

    with gr.Tab("Analysis"):
        analysis_out = gr.Markdown()

    with gr.Tab("Plan"):
        plan_out = gr.Markdown()

    with gr.Tab("Review"):
        review_out = gr.Markdown()

    with gr.Tab("Validation"):
        validation_out = gr.Markdown()

    run_btn.click(
        fn=run_pipeline,
        inputs=[topic_in, notes_in, files_in],
        outputs=[research_out, analysis_out, plan_out, review_out, validation_out, timer_out],
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=blue_theme,
        css="#timer-box {text-align:right; font-weight:600; color:#1d4ed8;}",
    )
