import PySimpleGUI as sg
import cv2
import numpy as np
import os
import time
import pickle
import itertools
from collections import defaultdict
import argparse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
import json
import random
import cmapy
from myclasses import DataGenerator

    
'''Macros'''
matplotlib.use("TkAgg")


'''Argument Parser'''
parser = argparse.ArgumentParser(description='SMARTMAT Data Visualizer')
parser.add_argument('--data_path', type=str, default='./PY/datagen_3d/all/' ,required=False, help="data_path")
parser.add_argument('--sample', type=int, required=False, default=0, help="Specific Sample to Load")
parser.add_argument('--uid', type=int, required=True, help="Specific UID to load state")
parser.add_argument('--load', type=int, required=False, default=0, help="Set to 1 to load state for given UID. Use with UID")
parser.add_argument('--debug', type=int, required=False, default=0, help="Set 1 to display GT labels")
parser.add_argument('--train', type=int, required=False, default=0, help="Set 1 to change UI as training session")


def load_data(data_path, train_flag):
    if train_flag == 1:
        shuffle = False
    else:
        shuffle = True
    params = {'dim':  (128,64,50) ,  #time window size 50
            'batch_size': 1,
            'n_channels': 1,
            'shuffle': shuffle,
            'file_path': data_path}
    # leave session out mode
    labels0 = np.load(data_path + 'm_labels0.npy') -1
    labels1 = np.load(data_path + 'm_labels1.npy') -1

    #train_list = np.arange(10152,29234).tolist()
    train_list = np.arange(0,29233).tolist()

    train_gen = DataGenerator(train_list, labels0, labels1, **params)
    return train_gen

'''To load the data files'''
def load_data_train(leaveoutsession, bins):
    m_data_train = np.zeros( (0, 128, 64, 50, 1) )
    m_labels0_train = np.zeros( (0) )
    m_labels1_train = np.zeros( (0) )
    r = leaveoutsession
    m_filename = "PY/Sess"+str(r)+"_bin"+str(bins)+".txt"
    print(m_filename)
    with open(m_filename,"rb") as fp:   #read file
        m_datlist = pickle.load(fp)
        m_labels0 = pickle.load(fp)
        m_labels1 = pickle.load(fp)
        fp.close
    m_datlist = np.array(m_datlist)
    # norm = np.max(m_datlist)
    # m_datlist = m_datlist/norm
    m_labels0 = np.array(m_labels0)
    m_labels1 = np.array(m_labels1)
    print(m_datlist.shape, m_labels0.shape , m_labels0.shape)
    maxIndex = m_datlist.shape[0]
    return m_datlist, m_labels0, m_labels1, maxIndex


'''Auxilliary Functions'''
def update_count(label_count, activity):   
 
    label_count[activity[1]] += 1
    return label_count

def countFull(label_count, activity, train_flag):
    if train_flag == 1:
        if label_count[activity[1]] < 3:
            return False
        else:
            return True
    elif train_flag == 0:
        if label_count[activity[1]] < 5:
            return False
        else:
            return True

def load_state(uid):
    m_filename = "./saved/" + str(uid) +".json"
    print('Loading State... for UID: {}\nfrom file: {}'.format(uid,m_filename))
    try:
        with open(m_filename, 'r') as fp:
            fdata = fp.read()
            state = json.loads(fdata)
            ID = int(list(state.keys())[-1]) + 1
            prev_activity = (int(list(state.values())[-1][0][0]),int(list(state.values())[-1][0][1]))
            sample_count = int(state["count"])
            total_samples = int(state["total"])
            # if sample_count > 4:
            #     sample_count -=1
            #     total_samples-=1
            return state, ID, sample_count, prev_activity, total_samples
    except:
        print("*** No saved files found with this filename! ***")
        exit(0)

def getData(data_gen, index):
    # if index == train.shape[0]:
    #     return
    imgs = []
    avg = []
    train, l1, l0, id = data_gen.__getitem__(index)

    for frame in range(train.shape[3]):
        imgs.append(train[0,:,:,frame,0])
        avg.append(np.mean(train[0,:,:,frame,0]))
    return np.asarray(imgs), l0[0] + 1, l1[0] + 1, avg, id[0]


def colorFrame(f):
    f = f/f.max() 
    f = np.uint8(f*255)
    f = cv2.applyColorMap(f, cmapy.cmap('magma'))
    f = cv2.resize(f,(448,768), interpolation=cv2.INTER_LINEAR)
    return f

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)    
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def pressurePlot(avg):
    fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
    t = np.arange(0, 50, 1)
    fig.add_subplot(111).plot(t, avg, )
    fig.savefig("./assets/plot.png", bbox_inches='tight')
    plt.close(fig)
    plot = cv2.imread("./assets/plot.png")
    return cv2.imencode(".png", plot)[1].tobytes()
    # return plot.tobytes()

def update_state(state, index, lab0, lab1, in_lab0, in_lab1, count, total_samples,debug):
    state[index] = [(lab0, lab1), (in_lab0, in_lab1)]
    state['count'] = count
    state['total'] = total_samples
    if debug == 1:
        print(state)
    return state

def save_state(uid, state):
    with open("./saved/" + str(uid) + ".json", 'w') as fp:
        json.dump(state, fp)
        fp.close()


'''Main Routine'''
def main():
    
    '''Loading Assets'''
    sg.theme("LightGreen")
    ref_table = "./assets/table.png"

    '''Defining Macros'''
    opt = parser.parse_args()
    index = opt.sample
    width, height = 1024,786
    state = defaultdict(str)
    label_count = defaultdict(int)

    sample_count = 0
    uid = random.randint(1,101)
    prev_activity = None
    total_samples = 0

    '''Loading a state'''
    if opt.load == 1 and opt.uid != 0:
        uid = opt.uid
        state, index, sample_count, prev_activity, total_samples = load_state(opt.uid)
    elif opt.load == 0 and opt.uid != 0:
        uid = opt.uid
    else:
        print("Assigned User ID: ", uid)
    
    if opt.train == 1:
        opt.debug = 1
    # session = opt.session
    # bin = opt.bin
    data_gen = load_data(opt.data_path, opt.train)


    # data, labels1, labels0, maxIndex = load_data_train(session, bin)

    # print('Number of Samples: ', maxIndex)
    activity = None

    '''Define the window layout'''
    image_viewer_column = [
        [sg.Image(filename="", key="-IMAGE-", size=(128,64))],
        
        ]

    reference_view_column = [
        [sg.Image(filename="", key="-PLOT-", size=(64,64))],
        #  [sg.Canvas(key="-CANVAS-")],
         [
            sg.Slider(
                (0, 49),
                1,
                1,
                orientation="h",
                size=(55, 15),
                key="-THRESH SLIDER-",
                pad=((70,10),(1,1),)
            ),
        ],
        [sg.Radio("Frames", "Radio", size=(10, 1), key="-THRESH-")],
        
    ]

    button_view_column = [
        [sg.Button("Play ►", size=(10, 1))],
        [sg.Button("Next ->", size=(10, 1))],
        [sg.Button("Prev <-", size=(10, 1))],
        [sg.Button("Exit", size=(10, 1))],
        
    ]

    label_view_column = [
        [sg.Text('Enter Labels(0,1): ')],
        [sg.InputText('Enter Category (1-9)', size=(52,1), justification='right', text_color='red', background_color='white', key='in_label0')],
        [sg.InputText('Enter Exercise (1-47)', size=(52,1), justification='right', text_color='red', background_color='white', key='in_label1')],
        [sg.Text("", size=(60, 1), justification="center", key='labels', font='Any 24')],
        [sg.Button("Save", size=(10, 1), key="-SAVE-")],
        [sg.Text("", size=(60, 1), justification="right", key='new', text_color='green')],
        [sg.Text("", size=(60, 1), justification="right", key='save_prompt', text_color='red')],
        [sg.Text("", size=(60, 1), justification="right", key='count')]
        
    ]
    

    layout = [
        # [sg.Text("SmartMAT Data Visualizer", size=(60, 1), justification="center")],
        [sg.Text("", size=(60, 1), justification="center", key='session')],
        [sg.Text("", size=(60, 1), justification="center", key='frame')],
        [sg.Text("", size=(60, 1), justification="center", key='sid')],
        [sg.Text("", size=(60, 1), justification="center", key='samples')],
        [sg.Text("", size=(60, 1), justification="center", key='user')],

    

        [   sg.Column(image_viewer_column),
            sg.VSeparator(),
            sg.Column(reference_view_column, size=(512, 768)),

        ],
        [sg.HorizontalSeparator(),],
        [
            sg.Column(button_view_column),
            sg.VSeparator(),
            sg.Column(label_view_column),

        ]

        
    ]

    # Create the window and show it
    window = sg.Window('SmartMAT Data Visualizer', layout, location=(0,0), keep_on_top=True).Finalize()
    window.Maximize()
    
    # Populating the window
    frames, label0, label1, avg, ID = getData(data_gen, index)
    window["-PLOT-"].update(data=pressurePlot(avg))
    activity = (label0,label1)

    '''Event Loop'''
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            # sg.Popup('Max Frames reached!', keep_on_top=True)
            if bool(state) == False:
                break
            if state['count'] == '5':
                sg.Popup('Please annotate and save this sample to exit.', keep_on_top=True)
                continue
            if (sg.popup_yes_no('Save this session?')) == 'Yes':
                save_state(uid,state)
                break
            else:
                break
            
        frame = colorFrame(frames[0])

        if values["-THRESH-"]:
            frame = colorFrame(frames[int(values["-THRESH SLIDER-"])])
            window["frame"].update("Frame: {}".format(int(values["-THRESH SLIDER-"])))
            
        if event == "Play ►":
            for i in range(0,50):
                frame = colorFrame(frames[i])
                imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
                window["frame"].update("Frame: {}".format(i))
                window.refresh()
                time.sleep(0.03)

        elif event == "Play -►" and window.Element('-THRESH-'):
            for i in range(0,50):
                frame = colorFrame(frames[i])
                imgbytes = cv2.imencode(".png", frame)[1].tobytes()
                window["-IMAGE-"].update(data=imgbytes)
                window["frame"].update("Frame: {}".format(i))
                window.refresh()
                time.sleep(0.1)

        if event == 'Next ->':
            index += 1
            print('Index: ', index)
            frames, label0, label1, avg, ID = getData(data_gen,index)
            # prev_activity = activity
            activity = (label0, label1)
            
            while countFull(label_count, activity, opt.train):
                index += 1
                print('Index: ', index)
                frames, label0, label1, avg, ID = getData(data_gen,index)
                # prev_activity = activity
                activity = (label0, label1)    
            
            # if prev_activity != activity:
            #     sample_count = 0
            #     window["new"].update("New Sample!")

            # window["session"].update("Session: {}, Bin: {}".format(session, bin))
            window["sid"].update("Sample ID: {}".format(ID))

            if opt.debug == 1:
                window["labels"].update("Sample: {}, Category: {}, Exercise: {}".format(ID,label0,label1))
            # draw_figure(window["-CANVAS-"].TKCanvas, pressurePlot(avg))
            window["-PLOT-"].update(data=pressurePlot(avg))
            window["save_prompt"].update("")
            # window["-SAVE-"].disappear = False
            window.FindElement('-SAVE-').Update(disabled=False)
            window["new"].update("")

            window.refresh()



        if event == 'Prev <-':
            if index == 0:
                sg.Popup('No Previous Frame!', keep_on_top=True)
                continue
            index -= 1
            print('Index: ', index)
            frames, label0, label1, avg, ID = getData(data_gen, index)
            # prev_activity = activity
            activity = (label0, label1)
            window["sid"].update("Sample ID: {}".format(ID))

            if opt.debug == 1:
                window["labels"].update("Sample: {}, Label-0: {}, Label-1: {}".format(ID,label0,label1))
            # draw_figure(window["-CANVAS-"].TKCanvas, pressurePlot(avg))                
            window["-PLOT-"].update(data=pressurePlot(avg))
            window["save_prompt"].update("")
            window["new"].update("")
            window.refresh()

        if event == "-SAVE-":
            try:
                in_lab0 = int(values["in_label0"])
                in_lab1 = int(values["in_label1"])
            
                if in_lab0 < 1 or in_lab1 < 1 or in_lab0 > 9 or in_lab1 > 47:
                    sg.Popup('Invalid Label(s)!', keep_on_top=True)
                    continue
                
                if total_samples == 235:
                    sg.Popup('Annotation Samples Completed! You may save and exit.', keep_on_top=True)
                    continue
                # sample_count +=1
                len_state = len(list(state.values()))
                state = update_state(state,str(ID), str(label0), str(label1), str(in_lab0), str(in_lab1), str(sample_count), str(total_samples), opt.debug)
                
                # exit(0)
                if len_state < len(list(state.values())):
                    label_count = update_count(label_count, activity)
                    total_samples += 1

                window["save_prompt"].update("Saved!")
                # window["-SAVE-"].disappear = True
                window.FindElement('-SAVE-').Update(disabled=True)
                
            except:
                sg.Popup('Invalid Label(s)!', keep_on_top=True)
                continue

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)
        # window["session"].update("Session: {}, Bin: {}".format(session, bin))
        window["samples"].update("Samples: {}/235".format(total_samples))
        
        if opt.train == 1:
            window.FindElement("user").Update(text_color='red')
            window["user"].update("TRAINING MODE")            
        else:
            window["user"].update("UserID: {}".format(uid))

        window["sid"].update("Sample ID: {}".format(ID))

        if opt.debug == 1:
            # window["count"].update("Sample Count: " + str(sample_count))
            window["labels"].update("Sample: {}, Category: {}, Exercise: {}".format(ID,label0,label1))

    window.close()

main()