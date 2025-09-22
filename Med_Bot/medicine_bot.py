import streamlit as st
import random
import spacy
from collections import defaultdict, deque
import time

# --- Refined Language Model (Trigram Markov Chain) ---
class MedicalLanguageModel:
    """
    A more refined language model based on a trigram Markov chain.
    It learns word transition probabilities based on the two preceding words.
    """
    def __init__(self):
        # The key is now a tuple of two words, e.g., ('sore', 'throat')
        self.markov_chain = defaultdict(list)
        # Load the pre-trained spaCy model
        self.nlp = spacy.load("en_core_web_sm")

    def train(self, text):
        """Trains the model on text using lemmatization."""
        # Process the text with spaCy for accurate tokenization
        doc = self.nlp(text.lower())
        
        # Lemmatize and filter out punctuation/stopwords for cleaner data
        words = [
            token.lemma_ for token in doc 
            if token.is_alpha and not token.is_stop
        ]

        if len(words) < 3:
            st.warning("Training data is too short to build a trigram model.")
            return

        # Build the trigram Markov chain: (word1, word2) -> [next_word]
        for i in range(len(words) - 2):
            word1, word2 = words[i], words[i+1]
            next_word = words[i+2]
            self.markov_chain[(word1, word2)].append(next_word)

    def generate(self, seed_phrase, length=30):
        """Generates a new sentence from a starting two-word phrase."""
        if not self.markov_chain:
            return "The model has not been trained yet. Please check the training data file."
        
        # Ensure the seed phrase is a valid key in our model
        current_words = tuple(seed_phrase)
        if current_words not in self.markov_chain:
            return "I don't have enough information on that topic. Please rephrase your symptoms."

        generated_words = list(current_words)
        # Use a deque to keep track of recent words to avoid repetitive loops
        recent_words = deque(current_words, maxlen=5)

        for _ in range(length - 2):
            possible_next_words = self.markov_chain.get(current_words)
            if not possible_next_words:
                break # Stop if we reach a dead end

            # Logic to avoid simple repetition
            next_word = random.choice(possible_next_words)
            attempts = 0
            while next_word in recent_words and attempts < 5:
                # Try to find a different word if the chosen one was recent
                next_word = random.choice(possible_next_words)
                attempts += 1
            
            generated_words.append(next_word)
            
            # Slide the context window forward for the next word
            current_words = (current_words[1], next_word)
            recent_words.append(next_word)
            
        # Capitalize the first word and join into a sentence
        generated_words[0] = generated_words[0].capitalize()
        return " ".join(generated_words) + "."

# --- Streamlit UI and Application Logic ---

@st.cache_resource
def load_model():
    """Loads and trains the language model from the local text file."""
    try:
        with open(r"C:\Users\GeetanshChopra\Desktop\Projects\SLM\med_bot\medical_notes.txt", 'r', encoding='utf-8') as f:
            training_text = f.read()
        model = MedicalLanguageModel()
        model.train(training_text)
        return model
    except FileNotFoundError:
        st.error("Training data file ('medical_notes.txt') not found. Please ensure it is in the same directory as this script.")
        return None

# Load the trained model once when the app starts
model = load_model()

# If the model failed to load, stop the app to prevent errors.
if model is None:
    st.stop()

# --- UI Configuration and State ---
st.set_page_config(page_title="Refined Physician Bot", page_icon="👨‍⚕️")
st.title("General Physician Bot (Refined Model)")
st.warning("**Disclaimer:** This is an AI simulation for educational purposes and not real medical advice. Please consult a qualified healthcare professional.", icon="⚠️")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello. I am a simulated physician bot. Please describe your symptoms."}]

# Display past chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Logic ---

def generate_bot_response(user_message):
    """Finds a valid two-word seed phrase from user input and generates a response."""
    doc = model.nlp(user_message.lower())
    words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
    if len(words) < 2:
        return "Please describe your symptoms in more detail."
    
    # Find a valid two-word phrase from the user's input to start generation
    seed_phrase = None
    for i in range(len(words) - 1):
        phrase = (words[i], words[i+1])
        if phrase in model.markov_chain:
            seed_phrase = phrase
            break # Use the first valid phrase found
            
    # If no valid phrase is found, pick a random one from the model's knowledge base
    if not seed_phrase:
        seed_phrase = random.choice(list(model.markov_chain.keys()))

    return model.generate(seed_phrase)

# Handle user input from the chat box
if prompt := st.chat_input("Describe your symptoms..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display bot response
    with st.chat_message("assistant"):
        # Use a spinner for a better user experience
        with st.spinner("The doctor is thinking..."):
            time.sleep(random.uniform(0.5, 1.2)) # Simulate processing time
            bot_response = generate_bot_response(prompt)
            st.markdown(bot_response)
    
    # Add the bot's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
