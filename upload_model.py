from huggingface_hub import HfApi

HF_TOKEN = "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"

LOCAL_FOLDER = "blip_finetuned"

REPO_ID = "fatma29/blip-finetuned"

api = HfApi()

api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN
)

print("Upload terminé avec succès ")