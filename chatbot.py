from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# Creating ChatBot Instance
chatbot = ChatBot(
    'Mental Health Bot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
) 

trainer = ListTrainer(chatbot)

training_data_depression = open('training_data/depression.txt').read().splitlines()
training_data_anxiety = open('training_data/anxiety.txt').read().splitlines()
training_data_ptsd = open('training_data/ptsd.txt').read().splitlines()
training_data_diagnosis = open('training_data/diagnosis.txt').read().splitlines()

training_data = training_data_depression + training_data_anxiety + training_data_ptsd + training_data_diagnosis

trainer.train(training_data)

# Training with English Corpus Data 
trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train(
    'chatterbot.corpus.english'
) 