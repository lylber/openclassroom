import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dash_bootstrap_components as dbc
from ressources.dico_features import *
from app import app

# Initialisation de l'application Dash
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model = joblib.load(r'dashbord_dash\model\regression_logistique_model_v2.pkl')
scaler = MinMaxScaler()
data = pd.read_csv(r'datasets\test.csv')

colonnes = data.columns[:-1]

# Layout pour l'application Dash
layout = html.Div(children=[ dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Prédition modèle", className="text-center",style={'fontFamily': 'Roboto'}), className="mb-0 mt-3")
        ]),
        html.Br(),
    *[dcc.Input(id=f'feature-{col}', type='number', placeholder=col) if col not in colonnes_catégoriques
      else dcc.Dropdown(id=f'feature-{col}', placeholder=col, options=[{'label': str(key), 'value': val} for key, val in
                                                       globals()[f'd_{col}'].items()], value=list(globals()[f'd_{col}'].keys())[0])
      for i, col in enumerate(colonnes)],
    html.Br(),
    html.Button('Prédiction', id='predict-button'),
    html.Div(id='prediction-output')
])
])

# Callback pour mettre à jour la sortie de la prédiction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State(f'feature-{col}', 'value') for col in colonnes]
)
def update_prediction(n_clicks, *features):
    try:
        if n_clicks is not None:
            # Créer un DataFrame avec les fonctionnalités et les noms de colonnes
            features_df = pd.DataFrame([features], columns=colonnes)

            # Remplacer les valeurs des colonnes catégoriques par les nombres associés dans les dictionnaires
            for col in colonnes_catégoriques:
                features_df[col] = features_df[col].map(globals()[f'd_{col}'])

            # Convertir le DataFrame en tableau NumPy
            features_array = features_df.to_numpy()

            # Mettre à l'échelle les fonctionnalités avec MinMaxScaler
            scaled_features = scaler.fit_transform(features_array[:, :len(colonnes_catégoriques)])

            # Concaténer les fonctionnalités catégoriques avec les fonctionnalités numériques mises à l'échelle
            features_array_scaled = np.concatenate((scaled_features, features_array[:, len(colonnes_catégoriques):]), axis=1)

            # Faire la prédiction avec le modèle
            prediction = model.predict(features_array_scaled)[0]
            probabilities = model.predict_proba(features_array_scaled)[0]
            positive_probability = probabilities[1]

            # Convertir la prédiction en texte significatif
            prediction_text = 'non solvable' if prediction == 1 else 'solvable'

            return (
                f"La prédiction du modèle est : {prediction_text}, "
                f"Probabilité : {positive_probability:.2f}"
            )
    except Exception as e:
        return f"Erreur : {str(e)}"

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)
