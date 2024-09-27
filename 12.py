import nltk
import spacy #thư viện đồ thị
from nltk import word_tokenize, pos_tag, ne_chunk
from spacy import displacy
# Tải mô hình ngôn ngữ tiếng Anh của Spacy
nlp = spacy.load('en_core_web_sm')

# Câu ví dụ
vidu = "Come to the Maxley Heights Center for Horticulture and learn how to create a beautiful"

# Tách từ (Tokenization)
tokens = word_tokenize(vidu)
print("Tokens:", tokens)

# Gán nhãn từ loại (POS Tagging)
pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)

# Khám phá thực thể có tên (NER) với NLTK
ner_tree = ne_chunk(pos_tags)
print("\nNamed Entities (NLTK):")
for chunk in ner_tree:
    if hasattr(chunk, 'label'):
        print(chunk.label(), ' '.join(c[0] for c in chunk))

# Khám phá thực thể có tên (NER) và Dependency Parsing với SpaCy
doc = nlp(vidu)
print("\nNamed Entities (SpaCy):")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Tạo đồ thị phụ thuộc ngữ nghĩa (Dependency Parsing)
print("\nDependency Parsing Visualization:")
displacy.serve(doc, style="dep")