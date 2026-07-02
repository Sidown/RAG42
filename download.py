from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Qwen/Qwen3-0.6B-GGUF",
    filename="Qwen3-0.6B-Q8_0.gguf",
    local_dir="./models"
)
print(path)