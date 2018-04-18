import matplotlib.pyplot as plt


def save_loss_plot(path_name, title, x_data, y_data):

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.grid(True)

    ax.plot(x_data, y_data, 'r')

    plt.savefig(path_name)


def save_2loss_plot(path_name, title, labels, x_data, y_data1, y_data2):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title(title)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    ax.grid(True)

    line1,  = ax.plot(x_data, y_data1, 'r')
    line2,  = ax.plot(x_data, y_data2, 'b')

    ax.legend((line1, line2), (labels[0], labels[1]))

    plt.savefig(path_name)

'''
if __name__ == '__main__':
    #save_2loss_plot('.pipppo','pippo',range(5), range(5), range(5, 10))
'''
