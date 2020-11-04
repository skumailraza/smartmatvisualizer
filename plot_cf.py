import json
from collections import defaultdict
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse

import numpy as np


parser = argparse.ArgumentParser(description='SMARTMAT Data Visualizer')
parser.add_argument('--uid', type=int, required=True, help="Specific UID to load state")
parser.add_argument('--display', type=int, required=False, default=0, help="Display the confusion matrices")


opt = parser.parse_args()
uid = opt.uid


def load_state(uid):
    m_filename = "./saved/states/" + str(uid) +".json"
    print('Loading State... for UID: {}\nfrom file: {}'.format(uid,m_filename))
    try:
        with open(m_filename, 'r') as fp:
            fdata = fp.read()
            state = json.loads(fdata)
            ID = int(list(state.keys())[-1]) + 1
            prev_activity = (int(list(state.values())[-1][0][0]),int(list(state.values())[-1][0][1]))
            sample_count = int(state["count"])
            total_samples = int(state["total"])
            del state['count']
            del state['total']

            return state, ID, sample_count, prev_activity, total_samples
    except:
        print("*** No saved files found with this filename! ***")
        exit(0)

state, ID, sample_count, prev_activity, total_samples = load_state(uid)
values = np.asarray(list(state.values()))


x_9 = []
x_47 = []
y_9 = []
y_47 = []

for i in values:
    i = np.asarray(i)
    x_9.append(i[0][0])
    y_9.append(i[1][0])
    x_47.append(i[0][1])
    y_47.append(i[1][1])

cm_9 = confusion_matrix(y_9, x_9)
cm_47 = confusion_matrix(y_47, x_47)
print('CM_9: \n', cm_9)
print('CM_47: \n', cm_47)

disp_9 = ConfusionMatrixDisplay(confusion_matrix=cm_9,
                              display_labels=range(1,10))
disp_47 = ConfusionMatrixDisplay(confusion_matrix=cm_47,
                              display_labels=range(1,48))
accuracy_9 = accuracy_score(y_9, x_9)
accuracy_47 = accuracy_score(y_47, x_47)

fig, ax = plt.subplots()
disp = disp_9.plot(include_values=True,
                 cmap=plt.cm.Blues, ax=ax, xticks_rotation='horizontal')
disp.ax_.set_title('Accuracy_9:  {:.2f}'.format(accuracy_9))
fig.savefig('./saved/cm/'+ str(uid) + '_9.png', quality=50)

if opt.display == 1:
    plt.show()

fig, ax = plt.subplots(figsize=(30, 30))
disp = disp_47.plot(include_values=True,
                 cmap=plt.cm.Blues, ax=ax, xticks_rotation='horizontal')
disp.ax_.set_title('Accuracy_47:  {:.2f}'.format(accuracy_47))

fig.savefig('./saved/cm/'+ str(uid) + '_47.png', quality=60)

if opt.display == 1:
    plt.show()