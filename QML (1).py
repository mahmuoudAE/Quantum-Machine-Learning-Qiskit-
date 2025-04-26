# pip install qiskit qiskit-machine-learning pandas scikit-learn matplotlib ipython numpy pandas
import pandas as pd
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, EfficientSU2, TwoLocal
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler



### Changable variables
num_features = 5
reps = 4 
optimizer = COBYLA(maxiter=200)
# optimizer = SPSA(maxiter=40, learning_rate=0.03, perturbation=0.05) 
entanglement = 'full'
frac = 0.3              # Fraction of the data being used
# Also change ansatz 


# Load the dataset from CSV
df = pd.read_csv('./final_dataset.csv')



# Assuming the last column is the target variable
df = df.sample(frac=frac, random_state=algorithm_globals.random_seed)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values







X = PCA(n_components=num_features).fit_transform(X)   # Reducing the number of features

print("running train_test_split")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=algorithm_globals.random_seed)

m = len(X_train)


print("starting feature map function")
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)
ansatz = TwoLocal(num_qubits=num_features, rotation_blocks=['ry', 'rz'], entanglement_blocks='cz', entanglement=entanglement, reps=reps)
# ansatz = EfficientSU2(num_qubits=num_features, reps=reps, entanglement=entanglement)


sampler = Sampler()

objective_func_vals = []
# callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()





# construct variational quantum classifier
# Construct the Variational Quantum Classifier
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    loss="cross_entropy",                            # Can be changed
    optimizer=optimizer,
    callback=callback_graph,
)



# Loading previous model
print("Loading previous model")
vqc = vqc.load('QML_Team_H')
vqc.warm_start = True
vqc.neural_network.sampler = sampler
vqc.optimizer = optimizer


print("fitting data to model")
start = time.time()
vqc.fit(X_train, y_train)
elapsed = time.time() - start
print(f"Training time: {round(elapsed)} seconds")






# Save the VQC model to a file
print("saving to QML_Team_H")
vqc.save('QML_Team_H')

# Load the VQC model from the file



print("running prediction")


# Make predictions on the test set

# Evaluate the model


train_score_q4 = vqc.score(X_train, y_train) 
test_score_q4 = vqc.score(X_test, y_test)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

result = f"""
Quantum VQC on the training dataset: {train_score_q4:.2f}
Quantum VQC on the test dataset:     {test_score_q4:.2f}
time taken = {elapsed}
"""
with open('model_log.txt', 'a') as the_file:
    the_file.write(f'{result}\n')

