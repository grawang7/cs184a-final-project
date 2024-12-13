### About our project

project.ipynb is a Jupyter notebook that can be run directly and demonstrates our project, containing the entirety of our project and models' code and output. (Model training takes more than 1 minute. To run notebook in 1 minute, please do not train models. (Example output is illustrated.))

project.html is project.ipynb exported as HTML, showing the outputs of all cells in the notebook.

archive.zip contains the dataset needed to run our project.ipynb (without zipping, data files would exceed size limit). The first code cell in project.ipynb will extract the data files into a folder called dataset so they can be accessed and used.

src/ is a directory that contains all the individual code for Python that our team wrote/adapted. These scripts and modules are not called by the project.ipynb notebook. (Scripts and modules consist of code that is in project.ipynb, just extracted/separated/modularized into individual files.)

src/data.py is our code for extracting the dataset from archive.zip, importing Python libraries and packages used in our project, and augmenting, visualizing, and exploring the data.

src/train.py is our code for training all our models (a general train_model function, as well as a train_model_with_acc function that outputs the model's accuracy as it's training).

src/evaluation.py is our code for evaluating all our models (a general calculate_accuracy and general confusion_matrix_and_metrics functions).

src/cnn.py is our code for our custom CNN model, including our implementation of/experimentation with Bayesian optimization.

src/resnet.py is our code for transfer learning with the ResNet-50 model in PyTorch.

src/alexnet.py is our code for transfer learning with the AlexNet model in PyTorch.

src/vit.py is our code for transfer learning with the base Vision Transformer model in PyTorch.

src/deit.py is our code for transfer learning with the Data-Efficient Image Transformer model in PyTorch.
