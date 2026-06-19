from query import ask_question


def main():

    print("=" * 60)
    print("DOCUMENT Q&A BOT")
    print("=" * 60)

    print("\nType 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        answer = ask_question(
            question
        )

        print("\nBot:")
        print(answer)
        print()


if __name__ == "__main__":
    main()