import argparse
import logging
import os

from openai import OpenAI

from openai_cts.test_compatibility import test_chat_completion_parameters

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Compatibility Test")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="Base URL of the OpenAI compatible API",
    )
    parser.add_argument("--model", default="gpt-3.5-turbo", help="Model name to test")
    args = parser.parse_args()

    client = OpenAI(api_key="mock-key", base_url=args.base_url)
    model = args.model
    results = []

    logger.info(f"Starting OpenAI Compatibility Test against {client.base_url} using model {model}")
    test_chat_completion_parameters(client, model, results)

    os.makedirs("output", exist_ok=True)
    report_file = "output/compatibility_report.md"
    with open(report_file, mode="w") as f:
        f.write("# OpenAI Compatibility Report\n\n")
        f.write(f"- **Base URL**: {client.base_url}\n")
        f.write(f"- **Model**: {model}\n\n")
        f.write("| Parameter | Status | Note | Value |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in results:
            f.write(
                f"| {row['parameter']} | {row['status']} | {row['note']} | `{row['value']}` |\n"
            )

    logger.info(f"\nReport generated: {report_file}")


if __name__ == "__main__":
    main()
