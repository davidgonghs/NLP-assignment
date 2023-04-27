import spacy
import nltk
from nltk.tree import Tree
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load GPT-2 model and tokenizer
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load spaCy model for POS tagging and dependency parsing
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Variables to store the user's name and chat history
user_name = None
chat_history = []


def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="tf")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def analyze_input(text):
    # Tokenize input text
    tokens = nltk.word_tokenize(text)

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Dependency parsing
    doc = nlp(text)
    tree = [token for token in doc]
    tree_str = " ".join([f"({token.text}, {token.dep_})" for token in tree])

    # Find prepositional phrase-attachments
    pp_attachments = []
    for i in range(len(tree)):
        if tree[i].dep_ == "prep":
            pp_attachment = []
            pp_attachment.append((tree[i].text, tree[i].head.text))
            for j in range(i+1, len(tree)):
                if tree[j].head == tree[i]:
                    pp_attachment.append((tree[j].text, tree[j].dep_))
            pp_attachments.append(pp_attachment)

    # Find verb phrases
    vp_indices = []
    for i in range(len(tree)):
        if tree[i].pos_ == "VERB":
            vp_indices.append(i)
    verb_phrases = []
    for i in vp_indices:
        subtree = list(tree[i].subtree)
        verb_phrases.append([token.text for token in subtree])

    # Find resolved word ambiguities
    for i in range(len(pos_tags)):
        if pos_tags[i][1] == "NNP":
            for j in range(len(tree)):
                if tree[j].text == pos_tags[i][0] and tree[j].dep_ == "compound":
                    pos_tags[i] = (f"{tree[j].text} {tree[j+1].text}", "n")
                    break

    # Return analysis results
    return {
        "Prepositional Phrase-attachment": pp_attachments,
        "Verb phrases": verb_phrases,
        "Resolved word ambiguities": pos_tags
    }


print("Chatbot: Hi, I'm a chatbot. What's your name?")
while True:
    user_input = input("User: ").strip()
    chat_history.append(f"User: {user_input}")

    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Chatbot: Goodbye!")
        break

    if user_name is None:
        user_name = user_input
        response = f"Nice to meet you, {user_name}."
    else:
        # Analyze user input
        analysis = analyze_input(user_input)

        # Generate response with GPT-2
        response = generate_response(user_input)

        # Print analysis results
        print("Prepositional Phrase-attachment:", analysis["Prepositional Phrase-attachment"])
        print("Verb phrases:", analysis["Verb phrases"])
        print("Resolved word ambiguities:", analysis["Resolved word ambiguities"])

    chat_history.append(f"Chatbot: {response}")
    print(f"Chatbot: {response}")
