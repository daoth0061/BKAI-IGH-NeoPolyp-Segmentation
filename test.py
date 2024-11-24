try:
    from albumentations.pytorch.transforms import ToTensorV2
    print("ToTensorV2 imported successfully!")
except ImportError as e:
    print(f"Import failed: {e}")