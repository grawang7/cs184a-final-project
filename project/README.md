### About our project

project.ipynb is a Jupyter notebook that can be run directly and demonstrates our project. (Please run this one. (To run in less than 1 minute, may require GPU, e.g., please run on Google Colab.))

cs184a-project.ipynb is a Jupyter notebook that contains the entirety of our project and models' code and output. It is the basis for our final report. (Model training takes longer than 1 minute. Please run project.ipynb. (Example output illustrated.))

project.html is project.ipynb exported as HTML, showing the outputs of all cells in the notebook.

cs184a-project.html is cs184a-project.ipynb exported as HTML, showing the outputs of all cells in the notebook. Our model evaluation was based on these results.

archive.zip contains the dataset needed to run our project.ipynb and cs184a-project.ipynb. The first code cell in project.ipynb (and cs184a-project.ipynb) will extract the data files into a folder called dataset so they can be accessed and used.

trained_custom_cnn.pth saves our trained custom CNN model.

trained_resnet_50.pth saves our trained ResNet-50 model.

trained_alexnet_SGD.pth saves our trained AlexNet model with SGD optimizer.

trained_alexnet_Adam.pth saves our trained AlexNet model with Adam optimizer.

trained_vit_b.pth saves our trained Visual Transformer model.

trained_deit.pth saves our trained Data-Efficient Image Transformer model.

src/ is a directory that contains all the individual code for Python that our team wrote/adapted. These scripts and modules are not called by the project.ipynb notebook. (Scripts and modules consist of code that is in cs184a-project.ipynb, just extracted/separated/modularized into individual files.)

src/data.py is our code for extracting the dataset from archive.zip, importing Python libraries and packages used in our project, and augmenting, visualizing, and exploring the data.

src/train.py is our code for training all our models (a general train_model function, as well as a train_model_with_acc function that outputs the model's accuracy as it's training).

src/evaluation.py is our code for evaluating all our models (a general calculate_accuracy and general confusion_matrix_and_metrics functions).

src/cnn.py is our code for our custom CNN model, including our implementation of/experimentation with Bayesian optimization.

src/resnet.py is our code for transfer learning with the ResNet-50 model in PyTorch.

src/alexnet.py is our code for transfer learning with the AlexNet model in PyTorch.

src/vit.py is our code for transfer learning with the base Vision Transformer model in PyTorch.

src/deit.py is our code for transfer learning with the Data-Efficient Image Transformer model in PyTorch.
