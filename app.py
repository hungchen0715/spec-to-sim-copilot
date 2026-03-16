"""
AI-to-USD Battery Factory Pipeline — Gradio UI

Main entry point. Interactive interface for:
1. Describe a battery module assembly task in natural language
2. Generate structured ModuleTask spec via LLM
3. Validate with industrial safety rules
4. Auto-repair if validation fails
5. Visualize the assembly layout with top-down preview
"""
import gradio as gr

from schema import ModuleTask
from llm import generate_task_spec, repair_task_spec
from validator import validate, format_report, format_issues_for_llm
from preview import render_preview
from config import MAX_REPAIR_ATTEMPTS


def process(prompt: str) -> tuple[str, str, str | None, str]:
    """
    Full pipeline: generate → validate → repair (if needed) → preview.
    """
    if not prompt.strip():
        return "", "⚠️ Please enter an assembly task description.", None, ""

    log_lines = []

    # ── Step 1: Generate ──
    log_lines.append("=" * 55)
    log_lines.append("STEP 1: Generating ModuleTask from prompt...")
    log_lines.append("=" * 55)

    try:
        spec, provider = generate_task_spec(prompt)
        log_lines.append(f"✅ Generated via: {provider}")
        log_lines.append(f"   Cells: {len(spec.cells)}")
        log_lines.append(f"   Robot: {spec.robot.model.value}")
        log_lines.append(f"   Gripper: {spec.robot.gripper.value}")
    except Exception as e:
        error_msg = f"❌ Generation failed: {e}"
        log_lines.append(error_msg)
        return "", error_msg, None, "\n".join(log_lines)

    # ── Step 2: Industrial Validation ──
    log_lines.append("")
    log_lines.append("=" * 55)
    log_lines.append("STEP 2: Running industrial safety validation...")
    log_lines.append("=" * 55)

    report = validate(spec)
    log_lines.append(format_report(report))

    # ── Step 3: Repair loop ──
    attempt = 0
    while not report.passed and attempt < MAX_REPAIR_ATTEMPTS:
        attempt += 1
        log_lines.append("")
        log_lines.append("=" * 55)
        log_lines.append(f"STEP 3: LLM Self-Correction (attempt {attempt}/{MAX_REPAIR_ATTEMPTS})...")
        log_lines.append("=" * 55)

        errors_text = format_issues_for_llm(report)
        try:
            spec, repair_provider = repair_task_spec(prompt, spec, errors_text)
            log_lines.append(f"✅ Repaired via: {repair_provider}")
        except Exception as e:
            log_lines.append(f"❌ Repair failed: {e}")
            break

        report = validate(spec)
        log_lines.append(format_report(report))

    # ── Step 4: Preview ──
    log_lines.append("")
    log_lines.append("=" * 55)
    log_lines.append("STEP 4: Rendering assembly layout preview...")
    log_lines.append("=" * 55)

    try:
        preview_path = render_preview(spec)
        log_lines.append(f"✅ Preview saved to: {preview_path}")
    except Exception as e:
        preview_path = None
        log_lines.append(f"⚠️ Preview failed: {e}")

    # ── Final output ──
    spec_json = spec.model_dump_json(indent=2)
    report_text = format_report(report)

    if attempt > 0:
        if report.passed:
            report_text = (
                f"🔧 Self-corrected after {attempt} attempt(s)\n\n{report_text}"
            )
        else:
            report_text = (
                f"⚠️ Could not fully repair after {attempt} attempt(s)\n\n"
                f"{report_text}"
            )

    return spec_json, report_text, preview_path, "\n".join(log_lines)


# ── Example prompts ──
EXAMPLES = [
    [
        "Assemble a 2x3 grid of LG E63 battery cells into a module. "
        "Use a UR10e robot arm with vacuum gripper. Place an overhead "
        "inspection camera."
    ],
    [
        "Create a large battery module with 20 HY 50Ah prismatic cells "
        "packed tightly in a single row. Use a KUKA KR6 robot."
    ],
    [
        "Assemble 6 CATL LFP cells but place Cell_05 at coordinates "
        "[2.5, 1.0, 0.0] — far from the robot base. Also rotate "
        "Cell_03 by 45 degrees."
    ],
]

# ── Build Gradio UI ──
with gr.Blocks(
    title="🔋 AI-to-USD Battery Factory Pipeline",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="slate",
    ),
) as demo:
    gr.Markdown(
        """
        # 🔋 AI-to-USD Battery Factory Pipeline
        **Generate validated battery module assembly specs for NVIDIA Isaac Sim.**
        
        Describe an assembly task → get structured JSON + industrial safety validation + layout preview.
        
        *LLM structured output → Pydantic schema → Industrial validator → Self-correction loop*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="📝 Assembly Task Description",
                placeholder=(
                    "e.g. Assemble a 2x3 grid of LG E63 battery cells "
                    "with a UR10e robot arm and vacuum gripper..."
                ),
                lines=4,
            )
            submit_btn = gr.Button(
                "🚀 Generate & Validate", variant="primary"
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=prompt_input,
                label="💡 Try these examples",
            )

        with gr.Column(scale=1):
            preview_output = gr.Image(
                label="🖼️ Assembly Layout (Top-Down)",
                type="filepath",
            )

    with gr.Row():
        with gr.Column(scale=1):
            report_output = gr.Textbox(
                label="✅ Industrial Safety Report",
                lines=14,
                interactive=False,
            )
        with gr.Column(scale=1):
            spec_output = gr.Code(
                label="📄 Generated ModuleTask (JSON)",
                language="json",
                lines=22,
            )

    with gr.Accordion("🔍 Processing Log (Validator → Repair Loop)", open=False):
        log_output = gr.Textbox(
            label="Full Pipeline Log",
            lines=20,
            interactive=False,
        )

    submit_btn.click(
        fn=process,
        inputs=[prompt_input],
        outputs=[spec_output, report_output, preview_output, log_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
