import subprocess
from pathlib import Path

TEMPLATE_PATH = Path("README.md.in")
OUTPUT_PATH = Path("README.md")


def main():
    result = subprocess.run(["pypatree"], capture_output=True, text=True, check=True)
    readme = TEMPLATE_PATH.read_text().replace("{{MODULE_TREE}}", result.stdout.strip())
    OUTPUT_PATH.write_text(readme)


if __name__ == "__main__":
    main()
