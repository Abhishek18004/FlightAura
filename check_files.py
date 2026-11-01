from pathlib import Path
import json

# Run this script from the FlightAura project root.
PROJECT_ROOT = Path.cwd()

IMPORTANT_FILES = [
    "flightaura_tokenizer.json",
    "sequence_config.json",
    "final_sentiment_config.json",
    "recommendation_config.json",
    "sentiment_ensemble_config.json",
    "sentiment_model_comparison.csv",
    "recommendation_model_comparison.csv",
    "sentiment_model_probabilities.npz",
    "recommendation_probabilities.npz",
    "flightaura_word2vec_skipgram.model",
    "flightaura_word2vec_cbow.model",
    "flightaura_embedding_matrix.npy",
]

print("=" * 70)
print("FLIGHTAURA PROJECT ROOT")
print("=" * 70)
print(PROJECT_ROOT.resolve())

print("\n" + "=" * 70)
print("1. MODEL AND ARTIFACT FILES")
print("=" * 70)

for folder_name in ["models", "dataset"]:
    folder = PROJECT_ROOT / folder_name
    print(f"\n[{folder_name}/]")

    if not folder.exists():
        print("Folder not found.")
        continue

    for path in sorted(folder.rglob("*")):
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{path.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")

print("\n" + "=" * 70)
print("2. SEARCH FOR NOTEBOOK ARTIFACTS")
print("=" * 70)

for filename in IMPORTANT_FILES:
    matches = list(PROJECT_ROOT.rglob(filename))

    if matches:
        for match in matches:
            print(f"[FOUND] {match.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[MISSING] {filename}")

print("\n" + "=" * 70)
print("3. CONFIGURATION FILE CONTENTS")
print("=" * 70)

CONFIG_FILES = [
    "sequence_config.json",
    "final_sentiment_config.json",
    "recommendation_config.json",
    "sentiment_ensemble_config.json",
]

for filename in CONFIG_FILES:
    matches = list(PROJECT_ROOT.rglob(filename))

    for path in matches:
        print(f"\n--- {path.relative_to(PROJECT_ROOT)} ---")

        try:
            with open(path, "r", encoding="utf-8") as file:
                config = json.load(file)

            print(json.dumps(config, indent=2))

        except Exception as error:
            print(f"Could not read configuration: {error}")

print("\n" + "=" * 70)
print("4. PROCESSED DATASET FILES")
print("=" * 70)

for filename in [
    "airline_reviews_cleaned.csv",
    "airline_reviews_unique.csv",
]:
    matches = list(PROJECT_ROOT.rglob(filename))

    if matches:
        for match in matches:
            print(f"[FOUND] {match.relative_to(PROJECT_ROOT)}")
    else:
        print(f"[MISSING] {filename}")

print("\nVerification finished. No files were modified.")