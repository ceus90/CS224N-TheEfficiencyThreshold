from datasets import load_dataset

def main():
    ds = load_dataset("ought/raft", "ade_corpus_v2", split="train", trust_remote_code=True)
    print("First 10 labels:")
    print(ds[:10]["Label"])

    unique_labels = sorted(set(ds["Label"]))
    print("\nUnique labels:")
    print(unique_labels)

    counts = {}
    for label in ds["Label"]:
        counts[label] = counts.get(label, 0) + 1

    print("\nLabel counts:")
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")

if __name__ == "__main__":
    main()