import random
from pathlib import Path


SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def main():

    random.seed(SEED)

    shard_files = sorted(
        Path("data/tensorized").glob("shard_*.pt")
    )

    print(f"Found {len(shard_files)} shards")

    random.shuffle(shard_files)

    n = len(shard_files)

    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train = shard_files[:train_end]
    val = shard_files[train_end:val_end]
    test = shard_files[val_end:]

    out_dir = Path("data/splits")
    out_dir.mkdir(exist_ok=True)

    for name, split in [
        ("train.txt", train),
        ("val.txt", val),
        ("test.txt", test),
    ]:

        with open(out_dir / name, "w") as f:
            for x in split:
                f.write(str(x) + "\n")

    print("\nCreated splits:")
    print(f"Train shards: {len(train)}")
    print(f"Val shards:   {len(val)}")
    print(f"Test shards:  {len(test)}")
    print(f"Total:        {len(train)+len(val)+len(test)}")


if __name__ == "__main__":
    main()
