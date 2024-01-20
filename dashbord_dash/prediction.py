import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import InconsistentVersionWarning
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_table
import warnings
from app import app

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Move loading outside the main script
def load_data():
    scaler = MinMaxScaler()
    model = joblib.load(r'dashbord_dash\model\regression_logistique_model_v2.pkl')
    data = pd.read_csv(r'datasets\test.csv')
    data_brut=pd.read_csv(r'dashbord_dash\ressources\brut_test.csv')
    data_brut=data_brut[data_brut['SK_ID_CURR'].isin(data.SK_ID_CURR)]
    d = data.drop(['SK_ID_CURR'], axis=1)
    scaled_features = scaler.fit_transform(d)
    df_scaled = pd.DataFrame(scaled_features, columns=d.columns)
    df_scaled['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)
    data_brut['AGE']=-data['AGE']
    data_brut['AGE']=data_brut['AGE'].astype(int)
    return model, df_scaled, data_brut

model, data, data_brut = load_data()

param = {'color': '#8badda'}

# Define the get_heat_color function to generate a gradient color
def get_heat_color(value):
    # Ensure the value is within the valid range [0, 1]
    value = max(0, min(1, value))
    
    # Calculate color based on the importance value
    red = int((1 - value) * 255)
    green = int(value * 255)
    return f'rgb({red}, {green}, 0)'

# Create a sample layout for the DataTable
table_layout = dash_table.DataTable(
    id='data-table',
    columns=[
        {'name': 'SK_ID_CURR', 'id': 'SK_ID_CURR'},
        {'name': 'CODE_GENDER', 'id': 'CODE_GENDER'},
        {'name': 'AGE', 'id': 'AGE'},
        {'name': 'NAME_FAMILY_STATUS', 'id': 'NAME_FAMILY_STATUS'},
        {'name': 'CNT_CHILDREN', 'id': 'CNT_CHILDREN'}

    ],
    style_table={'height': '300px', 'overflowY': 'auto'}
)

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

layout = html.Div(children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Prédiction du modèle", className="text-center",style={'fontFamily': 'Roboto'}), className="mb-0 mt-3")
        ]),
        html.Label("Entrez l'ID du client :"),
        dcc.Dropdown(
            id='input-customer-id',
            options=[{'label': str(i), 'value': i} for i in data['SK_ID_CURR']],
            value=data['SK_ID_CURR'].iloc[0],  # Set the default value to the first item in the list
            style={
                'backgroundColor': param['color'],
                'fontWeight': 'bold',
                'width':'150px'
            }
        ),
        html.Br(),

        html.Div(id='output-prediction'),

        dbc.Progress(id='probability-progress-bar', value=50, color="success", style={"height": "20px"}, className="mb-3"),
        html.Br(),
        dcc.Graph(id='prediction-chart1'),
        html.Br(),
        table_layout
    ])
])

# ...

# ...

@app.callback(
    [Output('output-prediction', 'children'),
     Output('prediction-chart1', 'figure'),
     Output('probability-progress-bar', 'value'),
     Output('probability-progress-bar', 'style'),
     Output('data-table', 'data')],
    [Input('input-customer-id', 'value')],
    allow_duplicate=True
)
def update_prediction(customer_id):
    
    try:
        customer_id = int(customer_id)
        input_features = data[data['SK_ID_CURR'] == customer_id].iloc[:, :-1].values

        # Normalize input features before prediction
        scaler = MinMaxScaler()
        input_features_scaled = scaler.fit_transform(input_features)

        prediction = model.predict(input_features_scaled)[0]
        probabilities = model.predict_proba(input_features_scaled)[0]

        positive_probability = probabilities[1]

        logistic_classifier = model.named_steps['m']
        coefficients = logistic_classifier.coef_[0]

        df = pd.DataFrame({'Variable': data.iloc[:, :-1].columns, 'Importance': coefficients})
        df = df.sort_values(by='Importance', ascending=False)

        print(f"DEBUG: Prediction: {prediction}, Positive Probability: {positive_probability}")

        # Use go.Figure directly for the bar chart
        fig = go.Figure()

        # Add a horizontal bar for each variable with a gradient color
        for i, (variable, importance) in enumerate(zip(df['Variable'], df['Importance'])):
            color = get_heat_color(importance)
            fig.add_trace(go.Bar(x=[importance], y=[variable], orientation='h', marker=dict(color=color), name=f'Variable {i + 1}'))

        fig.update_layout(title='Importance des variables dans le modèle de régression logistique', xaxis_title='Importance', yaxis_title='Variables')

        # Update the progress bar value based on the positive probability
        progress_value = int(positive_probability * 100)

        # Update the progress bar style with the dynamic color gradient
        color = get_heat_color(positive_probability)
        progress_style = {"height": "20px", "backgroundColor": color}

        # Fetch AGE and NAME_FAMILY_STATUS for the selected SK_ID_CURR
        selected_data = data_brut.loc[data_brut['SK_ID_CURR'] == customer_id,  ['SK_ID_CURR', 'AGE','CODE_GENDER', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN',]]
        table_data = selected_data.to_dict(orient='records')

        if prediction == 1:
            prediction='non solvable'
        else:
            prediction='solvable'

        print(f"DEBUG: Table Data: {table_data}")

        return (
            f"La prédiction du modèle pour le client {customer_id} est : {prediction}, "
            f"Probabilité  : {positive_probability:.2f}",
            fig,
            progress_value,
            progress_style,
            table_data
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return f"Erreur : ceci ne correspond pas à un numéro client connu", None, 0, {"height": "20px"}, []

# ...



if __name__ == '__main__':
    app.run_server(debug=True)
