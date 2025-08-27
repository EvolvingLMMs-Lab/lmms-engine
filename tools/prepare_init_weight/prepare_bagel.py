import os
import yaml
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table

# --- 1. Download Model ---
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
console = Console()
console.print(f"Downloading model from [bold cyan]{repo_id}[/]...")

path = snapshot_download(
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
console.print(f"Model downloaded to: [green]{path}[/]")

# --- 2. Define Paths ---
# These paths are now defined upfront
vae_path = os.path.join(path, "ae.safetensors")
vit_path = "hf/siglip-so400m-14-980-flash-attn2-navit"
llm_path = "hf/Qwen2.5-0.5B-Instruct" # This remains illustrative

# --- 3. Update YAML Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, "..", "..", "examples/bagel/bagel_example.yaml")
yaml_path = os.path.normpath(yaml_path)

console.print(f"Updating YAML file: [yellow]{yaml_path}[/]")

try:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if (
        isinstance(data, list)
        and len(data) > 0
        and "config" in data[0]
        and "dataset_config" in data[0]["config"]
        and "model_config" in data[0]["config"]
    ):
        config = data[0]["config"]
        # Update processor and main model path
        config["dataset_config"]["processor_config"]["processor_name"] = path
        config["model_config"]["load_from_pretrained_path"] = path
        # Add/Update specific component paths
        config["model_config"]["vae_path"] = vae_path
        config["model_config"]["vit_path"] = vit_path
    else:
        console.print("[bold red]Error: Unexpected YAML structure.[/]")
        exit(1)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

    console.print("YAML file updated successfully!")

except FileNotFoundError:
    console.print(f"[bold red]Error: YAML file not found at {yaml_path}[/]")
    exit(1)
except KeyError as e:
    console.print(f"[bold red]Error: Could not find key {e} in the YAML file. Structure might be wrong.[/]")
    exit(1)
except Exception as e:
    console.print(f"[bold red]An error occurred: {e}[/]")
    exit(1)


# --- 4. Formatted Output ---
table = Table(title="BAGEL Model Paths", show_header=True, header_style="bold magenta")
table.add_column("Component", style="dim", width=25)
table.add_column("Path", style="green")

table.add_row("Local Model Path", path)
table.add_row("VAE Path (in YAML)", vae_path)
table.add_row("VIT Path (in YAML)", vit_path)
table.add_row("LLM (illustrative)", llm_path)

console.print(table)

console.print("--- For easy copying ---")
console.print(f"MODEL_PATH='{path}'")
console.print(f"VAE_PATH='{vae_path}'")
console.print(f"VIT_PATH='{vit_path}'")
console.print(f"LLM_PATH='{llm_path}'")