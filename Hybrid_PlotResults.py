import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def hybrid_plot_results():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'PSO-TEL [28]', 'GWO-TEL [29]', 'KHOA-TEL [26]', 'SOA-TEL [27]', 'HKHSSO-TEL']
    Classifier = ['TERMS', 'VGG16 [29]', 'InceptionV3 [28]', 'Xception [30]', 'ResNet [33]', 'MobileNet [34]', 'TEL [28],[29],[30],[33],[34]', 'HKHSSO-TEL']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - Hybrid-Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - Hybrid-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [35, 45, 55, 65, 75, 85]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="PSO-TEL [28]")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="GWO-TEL [29]")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="KHOA-TEL [26]")
            plt.plot(learnper, Graph[:, 3], color='c', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="SOA-TEL [27]")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
                     label="HKHSSO-TEL")
            plt.xlabel('Learning Percentage (%)')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_Hybrid_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="VGG16 [29]")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="Inception [28]")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="Xception [30]")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="ResNet [33]")
            ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="MobileNet [34]")
            ax.bar(X + 0.50, Graph[:, 10], color='y', width=0.10, label="TEL [28],[29],[30],[33],[34]")
            ax.bar(X + 0.60, Graph[:, 4], color='k', width=0.10, label="HKHSSO-TEL")
            plt.xticks(X + 0.10, ('35', '45', '55', '65', '75', '85'))
            plt.xlabel('Learning Percentage (%)')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_Hybrid_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()



def KFOLD_PlotResults():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_KFold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Algorithm = ['TERMS', 'PSO-TEL [28]', 'GWO-TEL [29]', 'KHOA-TEL [26]', 'SOA-TEL [27]', 'HKHSSO-TEL']
    Classifier = ['TERMS', 'VGG16 [29]', 'InceptionV3 [28]', 'Xception [30]', 'ResNet [33]', 'MobileNet [34]', 'TEL [28],[29],[30],[33],[34]', 'HKHSSO-TEL']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - Hybrid-Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset - ', i + 1, ' - Hybrid-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="PSO-TEL [28]")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="GWO-TEL [29]")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="KHOA-TEL [26]")
            plt.plot(learnper, Graph[:, 3], color='c', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="SOA-TEL [27]")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
                     label="HKHSSO-TEL")
            plt.xticks(learnper, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFOLD')
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([80, 100])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_KFOLD_line.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="VGG16 [29]")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="Inception [28]")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="Xception [30]")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="ResNet [33]")
            ax.bar(X + 0.40, Graph[:, 9], color='c', width=0.10, label="MobileNet [34]")
            ax.bar(X + 0.50, Graph[:, 10], color='y', width=0.10, label="TEL [28],[29],[30],[33],[34]")
            ax.bar(X + 0.60, Graph[:, 4], color='k', width=0.10, label="HKHSSO-TEL")
            plt.xticks(X + 0.10, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFOLD')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_KFOLD_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    # hybrid_plot_results()
    KFOLD_PlotResults()
