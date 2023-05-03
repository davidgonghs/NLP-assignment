from datetime import datetime
import os
import spacy
import nltk
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import re
import json

#get msg with json format
def get_msg_json(sender, msg, analysis):
    return {
        "sender": sender,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": msg,
        "analysis": analysis
    }


class Chatbot:
    def __init__(self):
        self.model = TFGPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.nlp = spacy.load("en_core_web_sm")
        self.user_name = None
        self.chat_history = []
        self.knowledge = []

    def analyze_input(self, text):
        # Tokenize input text
        tokens = nltk.word_tokenize(text)

        # POS tagging
        pos_tags = nltk.pos_tag(tokens)

        # Dependency parsing
        doc = self.nlp(text)
        tree = [token for token in doc]
        tree_str = " ".join([f"({token.text}, {token.dep_})" for token in tree])

        # Find prepositional phrase-attachments
        pp_attachments = []
        for i in range(len(tree)):
            if tree[i].dep_ == "prep":
                pp_attachment = []
                pp_attachment.append((tree[i].text, tree[i].head.text))
                for j in range(i + 1, len(tree)):
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
                        pos_tags[i] = (f"{tree[j].text} {tree[j + 1].text}", "n")
                        break

        # Return analysis results
        return {
            "Prepositional Phrase-attachment": pp_attachments,
            "Verb phrases": verb_phrases,
            "Resolved word ambiguities": pos_tags
        }

    def anaphora_resolution(self, text):
        doc = self.nlp(text)
        antecedents = {}

        for token in doc:
            # Identify antecedents for pronouns
            if token.pos_ == "PROPN" and token.dep_ in ["nsubj", "nsubjpass"]:
                antecedents[token.lower_] = token.text

        # Replace pronouns with their antecedents, if available
        def replace_pronoun(match):
            pronoun = match.group(0).lower()
            return antecedents.get(pronoun, match.group(0))

        pronouns = r"\b(?:i|me|you|he|him|she|her|it|they|them)\b"
        resolved_text = re.sub(pronouns, replace_pronoun, text, flags=re.IGNORECASE)

        return resolved_text

    def generate_response(self, input_text):
        # input_ids = tokenizer.encode(input_text, return_tensors="pt")
        input_ids = self.tokenizer.encode(input_text, return_tensors="tf")
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Add generated_text to knowledge
        self.knowledge.append(generated_text)
        return generated_text

    def save_chat_history(self):
        if not os.path.exists("chat_history"):
            os.makedirs("chat_history")
        # Create a file name with timestamp
        json_file_name = "chat_history_" + str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".json"
        file_path = os.path.join("chat_history", json_file_name)
        # Write chat history and knowledge to file
        with open(file_path, "w") as f:
            f.write("Chat history:\n")
            for line in self.chat_history:
                f.write(json.dumps(line) + "\n")


        # create a text file to sve knowledge
        txt_file_name = "knowledge_" + str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".txt"
        file_path = os.path.join("chat_history", txt_file_name)
        with open(file_path, "w") as f:
            f.write("Knowledge:\n")
            for line in self.knowledge:
                f.write(line + "\n")

        print(f"Chat history and knowledge saved to {file_path}")

    def run(self):
        print("Chatbot: Hi, I'm a chatbot. What's your name?")
        while True:
            user_input = input("User: ").strip()
            # Add user input to chat history, should use json format include sender , timestamp, message
            # check if self.user_name is None,then sender is user, else sender is username
            # self.chat_history.append(f"User: {user_input}")


            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Chatbot: Goodbye!")
                break

            if not user_input:
                print("Chatbot: Please provide some input.")
                continue

            if self.user_name is None:
                self.user_name = user_input
                response = f"Nice to meet you, {self.user_name}."
            elif user_input.lower() == "get history":
                self.save_chat_history()
                response = "Chat history and knowledge have been saved."
            else:
                # Analyze user input
                analysis = self.analyze_input(user_input)
                # Print analysis results
                print("Prepositional Phrase-attachment:", analysis["Prepositional Phrase-attachment"])
                print("Verb phrases:", analysis["Verb phrases"])
                print("Resolved word ambiguities:", analysis["Resolved word ambiguities"])

                # Anaphora resolution
                user_input_resolved = self.anaphora_resolution(user_input)

                # Generate response with GPT-2
                response = self.generate_response(user_input_resolved)

                self.chat_history.append(get_msg_json("User" if self.user_name is None else self.user_name, user_input, analysis))

            print(f"Chatbot: {response}")

            # self.chat_history.append(f"Chatbot: {response}")
            self.chat_history.append(get_msg_json("Chatbot",response))

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.run()
