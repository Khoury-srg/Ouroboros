import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.font_manager
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

def categorical_color_list(nc, nsc, cmap="tab10", continuous=False):
    """ Return a color matrix. Each row is a categor. The colors are similar in the same category.
    """
    cmap = categorical_cmap(nc, nsc, cmap, continuous)
    clist = cmap(np.arange(nc*nsc))
    clist = np.reshape(clist, (nc, nsc, 4))
    return clist

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc="lower right")
  
def plot_acc_loss(file_path, save_path):
    colors = categorical_color_list(5,1)
    data = np.load(file_path)
    print(file_path)
    # print(data)
    # print(data["timestamps"])
    verification_time = sum([float(x[1]) for x in data["timestamps"] if x[0] == "verification"])
    training_time = sum([float(x[1]) for x in data["timestamps"] if x[0] == "training"])
    accs = data["accs"]
    losses = data["losses"]
    verify_steps = data["verified_steps"]
    timestamps = data["timestamps"]

    fig, ax = plt.subplots(figsize=(6,2.5))

    # print(timestamps)
    # ax.axvline(x = -100, c='g', linestyle=":", label = 'verified safe')
    # ax.axvline(x = -100, c='r', linestyle=":", label = 'found counter examples')
    time_ticks = [float(t[3]) for t in timestamps]
    ax.plot(time_ticks, accs, label="accuracy", color='k', linewidth=2)

    for i in range(1,len(timestamps)):
        if timestamps[i][0] == "training":
            ax.fill_between(time_ticks[i-1:i+1], -1, 2, facecolor=colors[0], alpha=0.2, label="training")
        else:
            ax.fill_between(time_ticks[i-1:i+1], -1, 2, facecolor=colors[1], alpha=0.5, label="verification")
    # for t in verify_steps:
    #     ax.axvline(x=t[1], c='g' if t[2] else 'r', linestyle=":")
    ax.set_xlim([0,time_ticks[-1]+1])
    ax.set_ylim([-0.05,1.05])
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accuracy')
    # ax.set_title('Accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.legend()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "12"
    legend_without_duplicate_labels(ax)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def merge_db_results(time_out = 300, v2_num=10):
    finetune_epoch = 5
    accs = []
    losses = []
    verified_steps = []
    timestamps = []
    encodings = ["00000-1", "01000-1", "11000-1", "11100-1", "11110-1"]
    for encoding in encodings:
        verification_time = 0
        training_time = 0
        for task_index in range(v2_num+1):
            file_path =  "../results/DBIndexTask_" + str(task_index) + "_" + encoding + "_" + str(time_out) + ".npz"
            data = np.load(file_path)
            timestamps.append(data["timestamps"])
            verification_time += sum([float(x[2]) for x in data["timestamps"] if x[0] == "verification"])
            training_time += sum([float(x[2]) for x in data["timestamps"] if x[0] == "training"])
            
        save_name = "../results/DBIndexTask" + encoding + "_" + str(time_out) + ".npz"
        np.savez("../results/" + save_name, 
                timestamps = [("training", training_time, training_time), ("verification", verification_time, verification_time)],
                )
        
def cluster_bar_graph(task_and_time_outs, encoding_and_labels, save_path):
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "12"
    
    running_times = {p[0]: [] for p in encoding_and_labels}
    training_times = {p[0]: [] for p in encoding_and_labels}
    verification_times = {p[0]: [] for p in encoding_and_labels}
    max_running_times = np.zeros(len(task_and_time_outs))
    hatches = [['', ''], ['\\\\\\', '\\\\\\\\'], ['..', '...'], ['///', '////'], ['xxx', 'xxxx'], ['+++', '++++'], ['---', '----']]
    for i, (task_prefix, task_name, time_out) in enumerate(task_and_time_outs):
        for encoding,label in encoding_and_labels:
            file_path =  "../results/" + task_prefix + encoding + "_" + str(time_out) + ".npz"
            print(file_path)
            data = np.load(file_path)
            # print(data["timestamps"])
            verification_time = sum([float(x[2]) for x in data["timestamps"] if x[0] == "verification"])
            training_time = sum([float(x[2]) for x in data["timestamps"] if x[0] == "training"])
            running_time = verification_time + training_time
            running_times[encoding].append(running_time)
            training_times[encoding].append(training_time)
            verification_times[encoding].append(verification_time)
            # print(verification_time)
            # print(training_time)
            max_running_times[i] = max(max_running_times[i], running_time)

    print("loaded")

    width = 0.8  # the width of the bars
    n = len(encoding_and_labels)
    x = np.arange(len(task_and_time_outs)) # the label locations
    font_size=9
    
    colors = categorical_color_list(len(encoding_and_labels), 3)
    fig, ax = plt.subplots(figsize=(12,2.5))
    for i,(encoding,label) in enumerate(encoding_and_labels):
        # print(running_times[encoding], running_times[baseline_encoding])
        normalized_training_time = np.array(training_times[encoding]) / max_running_times
        normalized_verification_time = np.array(verification_times[encoding]) / max_running_times
        ret = ax.bar(x - width/2 + width/2/n + i/n*width, normalized_training_time, width/n, label=label+" "+"(training)", edgecolor = 'black', color=colors[i][0], hatch = hatches[i][0], linewidth = 1)
        # color=colors[i][0],
        if i != 0:
            ret = ax.bar(x - width/2 + width/2/n + i/n*width, normalized_verification_time, width/n, bottom=normalized_training_time, label=label+" "+"(verification)", edgecolor = 'black', color=colors[i][1], hatch = hatches[i][1], linewidth = 1)
            ax.bar_label(ret, padding=3, fontsize=font_size, labels = ["{0:.0f}".format(math.ceil(x)) if x < 590 else "T/O" for x in running_times[encoding]])
        else:
            ax.bar_label(ret, padding=3, fontsize=font_size, labels = ["{0:.0f}".format(math.ceil(x)) for x in training_times[encoding]])
    
    print('ploted')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalized Running Time')
    # ax.set_title('Running time of different methods')
    ax.set_xticks(x, [p[1] for p in task_and_time_outs])
    ax.legend(bbox_to_anchor=(1.05, 1.09))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    fig.tight_layout()
    print("saving")
    plt.savefig(save_path, dpi=300)
    # plt.show()

def accuracy_comparison(task_classes):
    for task_class, time_out in task_classes:
    # for task_class in task_classes:
        counter_task = task_class(add_counterexample = True, incremental_training = True, batch_counterexample = True, early_rejection = True, incremental_verification = False, time_out=time_out) # wo finetune
        counter_task.load_model("../model/"+counter_task.save_name+".pth")
        counter_task.load_data("../results/"+counter_task.save_name+"_counterexamples.pickle")
        # print(len(counter_task.counterexample_data))
        normal_task = task_class(time_out = time_out) # normal
        normal_task.load_model("../model/"+normal_task.save_name+".pth")

        normal_normal_acc, loss = normal_task.test(normal_task.model, normal_task.testing_data, normal_task.compute_loss)
        normal_counter_acc, loss = normal_task.test(normal_task.model, counter_task.counterexample_data, normal_task.compute_loss)
        counter_normal_acc, loss = counter_task.test(counter_task.model, normal_task.testing_data, counter_task.compute_loss)
        counter_counter_acc, loss = counter_task.test(counter_task.model, counter_task.counterexample_data, counter_task.compute_loss)
        print(counter_task.save_name)
        print("   normal model acc:", normal_normal_acc, "\t counterexamples acc:", normal_counter_acc)
        print("ouroboros model acc:", counter_normal_acc, "\t counterexamples acc:", counter_counter_acc)
