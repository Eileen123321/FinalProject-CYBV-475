from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline

model = DistilBertForSequenceClassification.from_pretrained("./emotion_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./emotion_model")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
text = input("Type a sentence: ")
result = classifier(text, return_all_scores=True)
print("\nEmotion scores:")
for emotion in result[0]:
    print(emotion['label'], ":", emotion['score'])
