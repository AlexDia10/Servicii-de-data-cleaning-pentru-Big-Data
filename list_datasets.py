"""List all available NAB datasets."""

from data_loader import DataLoader

loader = DataLoader()
datasets = loader.list_nab_datasets()

print("Available NAB Datasets:\n")
for i, ds in enumerate(datasets, 1):
    print(f"{i}. {ds}")

# Filter artGaussian
print("\n\nArtGaussian datasets:")
art_datasets = [d for d in datasets if 'artGaussian' in d.lower() or 'art' in d.lower()]
for ds in art_datasets:
    print(f"  - {ds}")
