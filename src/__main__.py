from src.cli import RAG
import fire


def main() -> None:
    try:
        fire.Fire(RAG)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
