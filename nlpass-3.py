from datetime import datetime
import os
import spacy
import nltk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TFGPT2LMHeadModel

# Load GPT-2 model and tokenizer
model = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load GPT-Neo model and tokenizer
# model = TFGPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-350M", from_pt=True)
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-350M")

# Load spaCy model for anaphora resolution
nlp = spacy.load("en_core_web_sm")

# Variables to store the user's name and chat history
user_name = None
# chat history
chat_history = []
# Create a list of knowledge
knowledge = []

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





def anaphora_resolution(text):
    doc = nlp(text)
    for token in doc:
        if token.dep_ == "nsubj" and token.head.dep_ == "ROOT":
            if token.text.lower() == "i":
                return text.replace("I", user_name)
    return text



def generate_response(input_text):
    # input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = tokenizer.encode(input_text, return_tensors="tf")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Add generated_text to knowledge
    knowledge.append(generated_text)
    return generated_text



def save_chat_history():
    if not os.path.exists("chat_history"):
        os.makedirs("chat_history")

    # Create a file name with timestamp
    file_name = "chat_history_" + str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".txt"
    file_path = os.path.join("chat_history", file_name)

    # Write chat history and knowledge to file
    with open(file_path, "w") as f:
        f.write("Chat history:\n")
        for line in chat_history:
            f.write(line + "\n")
        f.write("\nKnowledge:\n")
        for knowledge_item in knowledge:
            f.write(knowledge_item + "\n")

    print(f"Chat history and knowledge saved to {file_path}")


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
    elif user_input.lower() == "get history":
        save_chat_history()
        response = "Chat history and knowledge have been saved."
    else:
        # Analyze user input
        analysis = analyze_input(user_input)
        # Print analysis results
        print("Prepositional Phrase-attachment:", analysis["Prepositional Phrase-attachment"])
        print("Verb phrases:", analysis["Verb phrases"])
        print("Resolved word ambiguities:", analysis["Resolved word ambiguities"])

        # Anaphora resolution
        user_input_resolved = anaphora_resolution(user_input)

        # Generate response with GPT-2
        response = generate_response(user_input_resolved)



    print(f"Chatbot: {response}")
    chat_history.append(f"Chatbot: {response}")
