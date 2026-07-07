import evaluate
import run_experiments


def main():
    print("Step 1/2: evaluating the trained baseline model...")
    evaluate.main()

    print("\nStep 2/2: running controlled comparison experiments...")
    run_experiments.main()

    print("\nAll evaluations are complete.")
    print("Download:", evaluate.ARCHIVE_PATH)


if __name__ == "__main__":
    main()
