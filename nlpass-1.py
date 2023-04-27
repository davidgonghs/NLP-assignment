import spacy
import nltk
import tensorflow as tf
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.wsd import lesk
from nltk.parse.chart import ChartParser
from nltk.grammar import CFG

nlp = spacy.load("en_core_web_sm")

# Define grammar for parsing prepositional phrases
grammar = CFG.fromstring("""
S -> NP VP
NP -> NNP
VP -> VBD NP PP
PP -> IN NP
NNP -> 'user'
VBD -> 'went'
IN -> 'to'
""")

# Initialize a chart parser with the grammar
parser = ChartParser(grammar)

def remember_name(user_input):
    doc = nlp(user_input)
    for token in doc:
        if token.ent_type_ == "PERSON":
            return token.text
    return None


def resolve_anaphora(text):
    # A basic example of anaphora resolution, replace with a more advanced approach if needed
    return text.replace("he", "John").replace("she", "Jane").replace("it", "something")


def resolve_pp_attachments(text):
    doc = nlp(text)
    pp_attachments = []

    for token in doc:
        if token.dep_ == "prep":
            head = token.head.text
            dependent = [t.text for t in token.children if t.dep_ in ("pobj", "pcomp")]
            pp_attachments.append((head, token.text, dependent))

    return pp_attachments


def extract_verb_phrases(text):
    doc = nlp(text)
    verb_phrases = []

    for token in doc:
        if token.dep_ == "xcomp":
            verb = token.head.text
            dependent = token.text
            verb_phrases.append((verb, dependent))

    return verb_phrases


def resolve_ambiguity(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)

    resolved = []
    for word, pos in pos_tags:
        synset = lesk(words, word, pos[0].lower())
        if synset:
            resolved.append(f"{word} ({synset.pos()})")
        else:
            resolved.append(word)

    return " ".join(resolved)


# def update_knowledge(user_input):
#     pass  # Implement your knowledge update logic here


def keep_chat_history(user_input, bot_response, chat_history):
    chat_history.append({"user": user_input, "bot": bot_response})
    return chat_history


# Main chat function
def chat():
    chat_history = []
    user_name = None
    print("Chatbot: Hi, I'm a chatbot. What's your name?")
    while True:
        user_input = input("User: ").strip()
        if not user_name:
            user_name = remember_name(user_input)
            if user_name:
                chat_history = keep_chat_history(user_input, f"Nice to meet you, {user_name}.", chat_history)
                print(f"Chatbot: Nice to meet you, {user_name}.")
            else:
                chat_history = keep_chat_history(user_input, "I didn't get your name. Please tell me your name.", chat_history)
                print("Chatbot: I didn't get your name. Please tell me your name.")
        elif user_input.lower() in ["quit", "exit", "bye"]:
            chat_history = keep_chat_history(user_input, "Goodbye!", chat_history)
            print("Chatbot: Goodbye!")
            break
        else:
            # Example: Anaphoric resolution
            if "he" in user_input or "she" in user_input or "it" in user_input:
                user_input = resolve_anaphora(user_input)

            # Example: Prepositional Phrase-attachment resolution
            pp_attachments = resolve_pp_attachments(user_input)
            print("Prepositional Phrase-attachment:", pp_attachments)

            # Example: Extract verb phrases
            verb_phrases = extract_verb_phrases(user_input)
            print("Verb phrases:", verb_phrases)

            # Example: Resolve word ambiguities
            resolved_text = resolve_ambiguity(user_input)
            print("Resolved word ambiguities:", resolved_text)

            # Example: Update chatbot knowledge
            #update_knowledge(user_input)

            bot_response = "I understood your input. Let's continue our conversation."  # Placeholder response
            chat_history = keep_chat_history(user_input, bot_response, chat_history)
            print("Chatbot:", bot_response)

    print("Chat history:")
    for entry in chat_history:
        print(f"User: {entry['user']}\nChatbot: {entry['bot']}")


if __name__ == "__main__":
    chat()
