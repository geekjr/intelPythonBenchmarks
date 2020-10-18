from os import closerange
import pandas as pd
import matplotlib.pyplot as plt
import argparse
plt.style.use('fivethirtyeight')

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--show", required=True,
	help="to show or not")
args = vars(ap.parse_args())

df = pd.read_csv('./intelPythonBenchmarks.csv')


df_sklearn = df.loc[df['Test'] == 'Sklearn']

y_data = [float(df_sklearn['Intel']), float(df_sklearn['Normal'])]
plt.figure(figsize=(20, 9), dpi=100)
graph = plt.barh(['Intel Python', 'Python'], y_data)
for index, value in enumerate(y_data):
    plt.text(value, index, str(value))
graph[0].set_color('g')
graph[1].set_color('r')
plt.ylabel('Version of Python')
plt.xlabel('Time in seconds')
plt.title('Sklearn Benchmarks with Intel Python')
plt.savefig('sklearn.png')
if args['show'] == "true":
    plt.show()


df_tf = df.loc[df['Test'] == 'TensorFlow']

y_data = [float(df_tf['Intel']), float(df_tf['Normal'])]
plt.figure(figsize=(20, 9), dpi=100)
graph = plt.barh(['Intel Python', 'Python'], y_data)
for index, value in enumerate(y_data):
    plt.text(value, index, str(value))
graph[0].set_color('g')
graph[1].set_color('r')
plt.ylabel('Version of Python')
plt.xlabel('Time in seconds')
plt.title('TensorFlow Benchmarks with Intel Python')
plt.savefig('tensorflow.png')
if args['show'] == "true":
    plt.show()


df_pyt = df.loc[df['Test'] == 'PyTorch']

y_data = [float(df_pyt['Intel']), float(df_pyt['Normal'])]
plt.figure(figsize=(20, 9), dpi=100)
graph = plt.barh(['Intel Python', 'Python'], y_data)
for index, value in enumerate(y_data):
    plt.text(value, index, str(value))
graph[0].set_color('g')
graph[1].set_color('r')
plt.ylabel('Version of Python')
plt.xlabel('Time in seconds')
plt.title('PyTorch Benchmarks with Intel Python')
plt.savefig('pytorch.png')
if args['show'] == "true":
    plt.show()