import spacy
from spacy.tokens import DocBin

# Load the pre-trained spaCy model
nlp = spacy.load("fr_core_news_lg")

# Load the DocBin back and update the model
doc_bin = DocBin().from_disk("data/models/corpus.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

# Fine-tune the model
optimizer = nlp.resume_training()
for i in range(10):  # Number of training iterations
    losses = {}
    for doc in docs:
        nlp.update([doc], sgd=optimizer, losses=losses)
    print(f"Iteration {i+1}, Losses: {losses}")

# Save the fine-tuned model
nlp.to_disk("data/models/fine_tuned_spacy_embedding")

