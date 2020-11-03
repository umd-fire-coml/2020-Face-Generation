import matplotlib.pyplot as plt


def loss_plot(epochs):
    plt.xlabel('epochs')
    plt.ylabel('loss')
    for (epoch_num, loss) in epochs:
        plt.plot(epoch_num, loss)
    plt.show()
