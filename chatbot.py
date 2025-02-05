import nltk
from nltk.chat.util import Chat, reflections

# Define pairs of patterns and responses
pairs = [
    (r"hi|hello", ["Hello!", "Hi there!"]),
    (r"how are you?", ["I'm doing well, thank you!", "I'm fine, how about you?"]),
    (r"what is your name?", ["I'm a chatbot!", "Call me Chatbot."]),
    (r"who are you?", ["I'm a simple chatbot created to chat with you!"]),
    (r"bye|quit", ["Goodbye!", "See you later!"])
]

# Create chatbot
chatbot = Chat(pairs, reflections)

# Start conversation
print("Chatbot: Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot.respond(user_input)
    print(f"Chatbot: {response}")
