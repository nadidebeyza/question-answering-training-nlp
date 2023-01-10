from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
import torch

# Gets .txt file as input
with open('input.txt', 'r') as file:
   lines = file.readlines()

for line in lines:
   context = line

# Keep the chat archive
chat_archive = []

def chat():
    var = input("What is your question?: ")
    #question = "How many programming languages does BLOOM support?"
    question = var

    chat_archive.append(question)

    my_trained_model = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-nl")

    question_answerer = pipeline("question-answering", model="deep_project_qa_model")
    question_answerer(question=question, context=context)

    tokenizer = AutoTokenizer.from_pretrained("deep_project_qa_model")
    inputs = tokenizer(question, context, return_tensors="pt")

    model = AutoModelForQuestionAnswering.from_pretrained("deep_project_qa_model")
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the highest probability from the model output for the start and end positions
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    # Decode the predicted tokens to get the answer
    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    tokenizer.decode(predict_answer_tokens)
    print("Answer:", tokenizer.decode(predict_answer_tokens))

    chat_archive.append(tokenizer.decode(predict_answer_tokens))

    var = input("Do you have another question? GO or BYE? ")
    question = var

    if question == "GO" or question == "go":
        chat()
    elif question == "BYE" or question == "bye":
        with open('output.txt', 'a') as f:
            f.write('\n'.join(chat_archive))

        print("You can find the chat archive in your files. Take care...")

chat()
