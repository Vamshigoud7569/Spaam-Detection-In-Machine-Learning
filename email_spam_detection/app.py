from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import random

app = Flask(__name__)
pickle_in = open('model.pickle','rb')
pac = pickle.load(pickle_in)
tfid = open('tranform.pickle','rb')
tfidf_vectorizer = pickle.load(tfid)

train = pd.read_csv('Email spam.csv')
train=train.dropna()
train['spam'].unique()
train[train['spam']=='its termination would not  have such a phenomenal impact on the power situation .  however '].shape
df_x=train['text']
df_y=train['spam']

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3, random_state=9)

tfidf_vectorizer= TfidfVectorizer(min_df=1,stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 

clf=MultinomialNB()
clf.fit(tfidf_train,y_train)
acc = clf.score(tfidf_train,y_train)
tfidf_test = tfidf_vectorizer.transform(x_test) 
y_pred = clf.predict(tfidf_test)

f1  = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precsion = precision_score(y_test, y_pred, average='weighted')

# Define constants
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
NUM_GENERATIONS = 20


# Function to generate a random filter
def generate_filter():
    return [random.uniform(-1, 1) for _ in range(len(emails[0][0]))]  # Random weights

# Function to calculate the fitness of a filter
def calculate_fitness(filter):
    correct = 0
    for email, label in emails:
        score = sum(char_weight * filter[i] for i, char_weight in enumerate(email))
        predicted_label = "spam" if score > 0 else "ham"
        if predicted_label == label:
            correct += 1
    return correct

# Function to perform crossover between two filters
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Function to perform mutation on a filter
def mutate(filter):
    mutated_filter = filter[:]
    for i in range(len(mutated_filter)):
        if random.random() < MUTATION_RATE:
            mutated_filter[i] += random.uniform(-0.5, 0.5)
    return mutated_filter


def genetic_algorithm():
    # Generate initial population
    population = [generate_filter() for _ in range(POPULATION_SIZE)]

    for generation in range(NUM_GENERATIONS):
        # Calculate fitness for each filter
        fitness_scores = [calculate_fitness(filter) for filter in population]

        # Select top filters for reproduction
        top_filters_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:10]
        top_filters = [population[i] for i in top_filters_indices]

        # Create next generation through crossover and mutation
        new_population = top_filters[:]
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.choices(top_filters, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population
         
        # Display best filter in each generation
        best_filter_index = top_filters_indices[0]
        best_filter = population[best_filter_index]
        return best_filter_index, best_filter

# Harris Hawks Optimization Algorithm (not used in this version)
def hho_algorithm(objective_function, num_variables, num_hawks, max_iter, lb, ub):
    pass
    # Initialization
    positions = np.random.uniform(lb, ub, (num_hawks, num_variables))
    convergence_curve = []

    for iter in range(max_iter):
        # Calculate fitness values
        fitness_values = np.apply_along_axis(objective_function, 1, positions)

        # Sort positions based on fitness values
        sorted_indices = np.argsort(fitness_values)
        sorted_positions = positions[sorted_indices]

        # Update the top positions (based on the exploration and exploitation phase)
        for i in range(num_hawks):
            for j in range(num_variables):
                r1 = np.random.random() # Random number for evasion
                r2 = np.random.random() # Random number for attack

                # Evasion phase
                if r1 < 0.5:
                    positions[i, j] = sorted_positions[0, j] + np.random.uniform(-1, 1) * (sorted_positions[0, j] - sorted_positions[i, j])

                # Attack phase
                else:
                    positions[i, j] = sorted_positions[0, j] - r2 * (sorted_positions[0, j] - sorted_positions[i, j])

                # Boundary handling
                positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])

        # Update convergence curve
        convergence_curve.append(np.min(fitness_values))
    return sorted_positions[0], convergence_curve

@app.route('/')
@app.route('/index') 
def index():
    return render_template('index.html')

@app.route('/login') 
def login():
    return render_template('login.html') 
   
@app.route('/home') 
def home():
    return render_template('home.html') 

@app.route('/abstract') 
def abstract():
    return render_template('abstract.html') 
 
@app.route('/future') 
def future():
    return render_template('future.html')    

@app.route('/user') 
def user():
    return render_template('user.html')     

@app.route('/upload') 
def upload():
    return render_template('upload.html') 

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/chart')
def chart():    
    abc = request.args.get('news')    
    input_data = [abc.rstrip()]
    # Transforming input
    tfidf_test = tfidf_vectorizer.transform(input_data)
    # Predicting the input
    y_pred = pac.predict(tfidf_test)
    # For demonstration purposes, assuming y_test is your entire test dataset's labels
    y_test_pred = pac.predict(tfidf_vectorizer.transform(x_test))
     # Dummy call to HHO algorithm
    dummy_num_variables = 10
    dummy_num_hawks = 20
    dummy_max_iter = 100
    dummy_lb = np.zeros(dummy_num_variables)
    dummy_ub = np.ones(dummy_num_variables)
    accpred = accuracy_score(y_test, y_test_pred)
    if y_pred[0] == 1:         
        label = "Spam"
    elif y_pred[0] == 0:
        label = "No Spam"
    return render_template('prediction.html', prediction_text=label, val0=accpred, val1=acc, val2=f1, val3=recall, val4=precsion)

@app.route('/performance') 
def performance():
    return render_template('performance.html')    

def txtpred(text):  
    textn = [text.rstrip()]  
    # Transforming input
    tfidf_test = tfidf_vectorizer.transform(textn)
    # Predicting the input
    y_pred = pac.predict(tfidf_test)
    # For demonstration purposes, assuming y_test is your entire test dataset's labels
    y_test_pred = pac.predict(tfidf_vectorizer.transform(x_test))
     # Dummy call to HHO algorithm
    dummy_num_variables = 10
    dummy_num_hawks = 20
    dummy_max_iter = 100
    dummy_lb = np.zeros(dummy_num_variables)
    dummy_ub = np.ones(dummy_num_variables)
    accpred = accuracy_score(y_test, y_test_pred)
    if y_pred[0] == 1:         
        label = "Spam"
    elif y_pred[0] == 0:
        label = "No Spam"
    return label

@app.route('/read_file', methods=["POST"])
def read_file():
    if request.method == 'POST':
        file = request.files['datasetfile']  # Corrected the file name to match the HTML form
        text = file.read().decode("utf-8")
        print(text)
        # Perform processing on the text as needed
        label = txtpred(text)  # Assuming txtpred is a function defined elsewhere
        return render_template('upload.html', prediction_text=label)
    
if __name__ == '__main__':
    app.run(debug=True)
