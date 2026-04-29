import os
import sys
import argparse
import logging
from llm import generate_task_spec
from validator import validate, format_report
from usd_export import export_usda

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SpecCLI")

def main():
    parser = argparse.ArgumentParser(description="Generate and validate battery module spec from prompt")
    parser.add_argument("--prompt", type=str, required=True, help="LLM Prompt for the battery layout")
    parser.add_argument("--out-json", type=str, required=True, help="Path to save the generated JSON spec")
    parser.add_argument("--out-usda", type=str, required=True, help="Path to save the generated USDA scene")
    args = parser.parse_args()
    
    logger.info(f"Processing prompt: {args.prompt}")
    
    # 1. Generate Spec
    try:
        spec, provider = generate_task_spec(args.prompt)
        logger.info(f"Spec generated successfully via {provider}.")
    except Exception as e:
        logger.error(f"Failed to generate spec: {e}")
        sys.exit(1)
        
    # 2. Validate Spec
    report = validate(spec)
    if not report.passed:
        logger.warning("Spec validation failed with the following issues:")
        logger.warning("\n" + format_report(report))
        # Note: In a real pipeline we might try to repair here, 
        # but for CLI we'll just log and proceed.
    else:
        logger.info("Spec validated successfully against industrial rules.")
    
    # 3. Export JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        f.write(spec.model_dump_json(indent=2))
    logger.info(f"Spec JSON saved to {args.out_json}")

    # 4. Export USDA
    os.makedirs(os.path.dirname(args.out_usda), exist_ok=True)
    usd_path = export_usda(spec, output_path=args.out_usda)
    logger.info(f"USDA scene saved to {usd_path}")

if __name__ == "__main__":
    main()
