from tkinter import Button, Frame, Label, Text, Tk
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Load the model and vectorizer
vectorizer = pickle.load(open("vector.pkl", "rb"))
loaded_model = pickle.load(open("model.pkl", "rb"))

# Define stemming function
def stemming(content):
    con = re.sub("[^a-zA-Z]", " ", content)
    con = con.lower()
    con = con.split()
    con = [PorterStemmer().stem(word) for word in con if word not in stopwords.words("english")]
    con = " ".join(con)
    return con

def predict():
    user_input = text_input.get("1.0", "end-1c")  # Get the text from the Text widget

    user_input_stemmed = stemming(user_input)

    input_data = [user_input_stemmed]
    vector_form1 = vectorizer.transform(input_data)
    prediction = loaded_model.predict(vector_form1)

    if prediction[0] == 1:
        result_label.config(text='Suicidal', fg='red')
    else:
        result_label.config(text='Non-Suicidal', fg='green')

window = Tk()
window.title("Suicidal Thoughts Detection App")

main_frame = Frame(window, bg='#f2f2f2')  # Light gray background color for a calming effect
main_frame.pack(fill='both', expand=True)

title_label = Label(main_frame, text="Suicidal Thoughts Detection App", font=('Arial', 26, 'bold'), bg='#f2f2f2', fg='#333333')  # Light gray background color, dark text color
title_label.pack(pady=(20, 10))

subheader_text = (
    "It's okay to feel overwhelmed, and I want you to know that there's no judgment here. Your pain matters, and I'm here "
    "to listen. Sometimes, just having someone to talk to can make a difference."
    "I can't fully comprehend the depth of what you're going through, but I want to be a source of support for you."
    "Your strength is evident, even in times of darkness, and it's okay to ask for help. Together, we can explore strategies "
    "to cope and navigate through these emotions. You are not defined by your struggles, and there is hope for a brighter future."
    "You can share your thoughts and feelings with me."
)

subheader_label = Label(main_frame, text=subheader_text, font=('Arial', 15), wraplength=680, justify='left', bg='#f2f2f2', fg='#555555')  # Light gray background color, slightly darker text color
subheader_label.pack(pady=20) 

text_input = Text(main_frame, height=8, width=60, font=('Arial', 15), bg='#ffffff')  # White background color for the Text widget
text_input.pack(pady=20)

predict_button = Button(main_frame, text="Predict", height=1, width=6, font=('Arial', 13), command=predict, bg='#6a0572', fg='#ffffff')  # Purple button color with white text
predict_button.pack(pady=20)

result_label = Label(main_frame, text="", font=('Arial', 15, 'bold'), bg='#f2f2f2', fg='#333333')  # Light gray background color, dark text color
result_label.pack()

window.mainloop()