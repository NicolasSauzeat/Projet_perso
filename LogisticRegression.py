# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importation des librairies et modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # to avoid deprecation warnings
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt 
plt.rc("font", size=14)


def main():
    path= input("Veuillez entrer le chemin du fichier à analyser :")
    data=import_BDD(path)
    fig2= stat_desc(data)
    features,target= preprocess(data)
    model,score,X_train,X_test,Y_train,Y_test = entrainement(features,target)
    reg_final = cross_validate(X_train,Y_train)
    Y_train_pred,Y_test_pred = prediction(X_train,X_test,reg_final)
    accuracy_train_set, accuracy_test_set = print_accuracy(Y_train,Y_train_pred,Y_test,Y_test_pred)
    print("Précision sur l'échantillon d'entraînement en % :", accuracy_train_set* 100)
    print("Précision sur l'échantillon de test en % :", accuracy_test_set* 100)
    print("f1-score on training set : ", f1_score(Y_train, Y_train_pred))
    print("f1-score on test set : ", f1_score(Y_test, Y_test_pred))
    fig = roc_visualisation(Y_train,Y_train_pred, Y_test,Y_test_pred,X_train,X_test,reg_final)
    display_graph(fig,fig2)
    print("Done")
    
    
def import_BDD(path):
# Importation de la BDD
    data = pd.read_excel(path,sheet_name="Hoja1")
# Statistiques descriptives
    print("number of rows :")
    display(data.shape[0])  
    print("number of columns :")
    display(data.shape[1])
    print("Pourcentage de valeurs manquantes :")
    display(100*data.isnull().sum()/data.shape[0])
    return data

def stat_desc(data):
    # Nombre de femmes et d'hommes
    sns.countplot(data["Genre"])
    #  Nombre d'individus par tranche d'âge 
    sns.countplot(data["TA"])
    # Nombre d'individus par zone 
    sns.countplot(data["Zone"])
    # Nombre d'individus par année d'enregistrement 
    sns.countplot(data["AÑO"])
    # Nombre d'individus par type de visa 
    sns.countplot(data["BENEFICIO"])
    # Nombre d'individus actifs et inactifs 
    sns.countplot(data["ACTIF"])
    # Tableau croisé entre âge et actif
    table=pd.crosstab(data["TA"],data["ACTIF"])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of TA vs Actif')
    plt.xlabel("Tranche d'âge")
    plt.ylabel("Proportion d'actifs")
    fig2=plt.savefig('Tranche-age_vs_actif')
    return fig2


def preprocess(data):
    list_features = ["GENRE","TA","ACT","PAYS","LANGUE","REG","ZONE","AÑO"]
    features = data.loc[:,list_features]
    target =data.loc[:,"ACTIF"]
    # Encodage des variables de type string 
    from sklearn.preprocessing import OrdinalEncoder
    ord_enc = OrdinalEncoder()
    features["TA"]= ord_enc.fit_transform(features[["TA"]])
    target= ord_enc.fit_transform(target.values.reshape(-1,1)).flatten()
    # Remplacement des valeurs manquantes 
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    features["ACT"]= imputer.fit_transform(pd.DataFrame(features["ACT"])).flatten()
    return features, target

def entrainement(features,target):
    # On sépare les données en données d'entraînement et de test 
    X_train, X_test, Y_train, Y_test = train_test_split(features,target,test_size=0.25, random_state=0,stratify=target)
    classifier = LogisticRegression(multi_class="auto",solver="liblinear")
    classifier.fit(X_train,Y_train)
    score= classifier.score(X_test,Y_test)
    print("Ceci est le score de la première régression logistique :")
    display(score*100)
    return classifier, score, X_train, X_test,Y_train, Y_test

# On cherche à améliorer le score avec une Cross Validation
def cross_validate(X_train,Y_train):
    classifier_bis = LogisticRegression()
    from sklearn.model_selection import GridSearchCV
    grid = { "C": [0.9,1,1.1],"penalty":["l1","l2"]}
    reg_cv = GridSearchCV(classifier_bis,grid, cv=10)
    reg_cv.fit(X_train,Y_train)
    return reg_cv
# Affichage des meilleurs paramètres
    print("Meilleurs hyperparamètres :",reg_cv.best_params_)
    print("Meilleure taux de précision :", reg_cv.best_score_)
# Utilisation des meilleurs paramètres pour la régression finale 
    final_reg = LogisticRegression(C=0.9,penalty="l2")
    final_reg.fit(X_train,Y_train)
    return final_reg


def prediction(X_train,X_test,final_reg):
# On calcule les valeurs prévues par le modèle pour le jeu d'entraînement
    Y_train_pred= final_reg.predict(X_train)
    Y_test_pred = final_reg.predict(X_test)
    return Y_train_pred, Y_test_pred





def print_accuracy(Y_train,Y_train_pred,Y_test,Y_test_pred):
# Score de précision du modèle
    accuracy_train_set = accuracy_score(Y_train, Y_train_pred)
    accuracy_test_set = accuracy_score(Y_test, Y_test_pred)
    return accuracy_train_set, accuracy_test_set



def roc_visualisation(Y_train,Y_train_pred, Y_test,Y_test_pred,X_train, X_test,classifier):

    # Visualisation des matrices de confusion 

    from plotly.subplots import make_subplots
    cm_train = confusion_matrix(Y_train, Y_train_pred)
    cm_test = confusion_matrix(Y_test, Y_test_pred)

    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ("train", "test"), 
                    x_title = 'Prediction', y_title = 'True label')
    fig.update_layout(
        title = go.layout.Title(text = "Confusion matrices", x = 0.5))
    fig.update_yaxes(autorange='reversed')
    fig.add_trace(
    go.Heatmap(
        name = 'train',
        x = ['0', '1'], 
        y = ['0', '1'], 
        z = cm_train,
        colorscale = 'gnbu',
        zmin = 0,
        zmax = max(cm_train.max(), cm_test.max())
    ),
    row = 1,
    col = 1
)  
    fig.add_trace(
        go.Heatmap(
        name = 'test',
        x = ['0', '1'], 
        y = ['0', '1'], 
        z = cm_test,
        colorscale = 'gnbu',
        zmin = 0,
        zmax = max(cm_train.max(), cm_test.max())
    ),
    row = 1,
    col = 2
)


# Visualize ROC curves
    probas_train = classifier.predict_proba(X_train)[:,1]
    precisions, recalls, thresholds = roc_curve(Y_train, probas_train)
    fig = go.Figure(
        data = go.Scatter(
            name = 'train',
            x = recalls, 
            y = precisions, 
            mode = 'lines'
            ),
        layout = go.Layout(
            title = go.layout.Title(text = "ROC curve", x = 0.5),
            xaxis = go.layout.XAxis(title = 'False Positive Rate'),
            yaxis = go.layout.YAxis(title = 'True Positive Rate')
            )
        )

    probas_test = classifier.predict_proba(X_test)[:,1]
    precisions, recalls, thresholds = roc_curve(Y_test, probas_test)
    fig.add_trace(go.Scatter(
        name = 'test',
        x = recalls, 
        y = precisions, 
        mode = 'lines'
        )
        )
    return fig


def display_graph(fig,fig2):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot
    plot(fig)
    plot(fig2)


if __name__=="__main__":
    main()