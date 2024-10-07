import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
import matplotlib.pyplot as plt


def get_matrix_beloved_color(model, name):
    iris = pd.read_csv('model/data (2).csv')
    iris.drop_duplicates(inplace=True)
    X = iris.drop(["beloved_color"], axis=1)
    Y = iris["beloved_color"]
    label_encoder = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    y_test = label_encoder.fit_transform(y_test)
    y_pred_logic = model.predict(X_test)
    y_actu = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(y_pred_logic, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    heatmap = sn.heatmap(df_confusion, annot=True)
    fig = heatmap.get_figure()
    fig.savefig(f'static/media/{name}.png')
    plt.close(heatmap.get_figure())


def get_matrix_iris(model, name):
    iris = pd.read_csv('model/IRISS.csv')
    iris.drop_duplicates(inplace=True)
    X = iris.drop(["species"], axis=1)
    Y = iris["species"]
    label_encoder = LabelEncoder()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
    y_test = label_encoder.fit_transform(y_test)
    y_pred_logic = model.predict(X_test)
    y_actu = pd.Series(y_test, name='Actual')
    y_pred = pd.Series(y_pred_logic, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    heatmap = sn.heatmap(df_confusion, annot=True)
    fig = heatmap.get_figure()
    fig.savefig(f'static/media/{name}.png')
    plt.close(heatmap.get_figure())


