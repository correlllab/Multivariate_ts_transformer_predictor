import sys, os, pickle, json
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import matplotlib.pyplot as plt


# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, f'{y[i]:.2f}%', ha = 'center',
                 bbox = dict(facecolor = 'white', alpha = .5))


if __name__ == '__main__':
    model_names = ['FCN', 'RNN', 'VanillaTransformer', 'OOP_Transformer']
    confusion_matrices = dict.fromkeys(model_names)

    counter = 0
    values = []
    names =[]
    for model_name in model_names:
        try:
            with open(f'../saved_data/evaluation_results/{model_name}_conf_matrix.json', 'r') as f:
                confusion_matrices[model_name] = json.load(f)
            counter += 1
            values.append(confusion_matrices[model_name]['NC'] * 100)
            names.append(model_name)
        except FileNotFoundError as e:
            print(f'\nERROR: {e} (confusion matrix does not exist for this model)\n')

    print(confusion_matrices)

    plt.figure(figsize = (10,5))
    plt.bar(x=names, height=values)
    addlabels(names, values)

    plt.xlabel('Model')
    plt.ylabel('Percentage of NC')
    plt.title('NC rate of the models with the preemptive dataset (i.e. unseen data during training)')
    
    plt.savefig('../saved_data/imgs/evaluation/models_NC_rates.png')