# Documentation de l'Application de Prévision de Séries Temporelles

---

---

## Introduction

Cette application Streamlit permet de réaliser des prévisions de séries temporelles à partir de fichiers Parquet téléchargés par l'utilisateur. Elle intègre également des événements PredictHQ pour ajuster les prévisions en fonction des événements futurs.

## Importation des Bibliothèques Nécessaires

Le code commence par l'importation des bibliothèques nécessaires :

```python
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from predicthq import Client

```

- `streamlit` : pour créer l'interface web.
- `pandas` : pour manipuler les données.
- `Prophet` : pour les prévisions de séries temporelles.
- `plotly` : pour visualiser les données.
- `Client` de PredictHQ : pour intégrer les événements PredictHQ.

## Configuration de l'Interface Streamlit

Le titre de l'application est défini comme suit :

```python
st.title('Streaming Forecast App')

```

## Téléchargement de Fichiers Parquet

L'utilisateur peut télécharger plusieurs fichiers Parquet :

```python
uploaded_files = st.file_uploader("Upload Parquet files", type=["parquet"], accept_multiple_files=True)

```

## Traitement des Fichiers Téléchargés

Si des fichiers sont téléchargés, ils sont lus et combinés :

```python
if uploaded_files:
    df_list = []

    for file in uploaded_files:
        df = pd.read_parquet(file)
        if pd.api.types.is_numeric_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'], unit='s')
        df_list.append(df)

    data = pd.concat(df_list, ignore_index=True)
    data.sort_values(by='ts', inplace=True)

```

1. Les fichiers Parquet sont lus en DataFrames pandas.
2. La colonne 'ts' est convertie en format datetime si nécessaire.
3. Les DataFrames sont combinés en un seul DataFrame et triés par la colonne 'ts'.

## Affichage des Données Combinées et Triées

Les données combinées et triées sont affichées :

```python
st.subheader('Combined and Sorted Data')
st.write(data)

```

## Tracé des Données Brutes

Une fonction est définie pour tracer les données brutes :

```python
def plot_raw_data():
    for col in data.columns:
        if col != 'ts' and pd.api.types.is_numeric_dtype(data[col]):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['ts'], y=data[col], name=col))
            fig.layout.update(title_text=f'Time Series data for {col}', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

plot_raw_data()

```

1. La fonction `plot_raw_data` trace les colonnes numériques des données brutes.
2. Un graphique est créé pour chaque colonne numérique, à l'exception de 'ts'.

## Prévisions avec Prophet

Les prévisions sont réalisées à l'aide de Prophet :

```python
st.subheader('Forecast data')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

combined_forecast_df = pd.DataFrame()

for col in data.columns:
    if col != 'ts' and pd.api.types.is_numeric_dtype(data[col]):
        df_train = data[['ts', col]].rename(columns={"ts": "ds", col: "y"})
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)
        forecast_new = forecast[forecast['ds'] > data['ts'].max()]
        forecast_col = forecast_new[['ds', 'yhat']].rename(columns={'ds': 'ts', 'yhat': f'Forecast_{col}'})
        if combined_forecast_df.empty:
            combined_forecast_df = forecast_col
        else:
            combined_forecast_df = pd.merge(combined_forecast_df, forecast_col, on='ts')

        st.write(f'Forecast plot for {col}')
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig)

        st.write(f"Forecast components for {col}")
        fig_components = m.plot_components(forecast)
        st.write(fig_components)

```

1. L'utilisateur sélectionne le nombre d'années de prévision via un curseur.
2. Pour chaque colonne numérique, Prophet est utilisé pour créer des prévisions.
3. Les prévisions sont ajoutées à un DataFrame combiné.
4. Les graphiques de prévision et les composants de prévision sont affichés.

## Sauvegarde des Prévisions Combinées

Les prévisions combinées sont sauvegardées dans un fichier CSV :

```python
combined_forecast_df.to_csv('combined_forecast.csv', index=False)
st.write("Forecasts saved to CSV file")

```

## Intégration avec PredictHQ

Les événements PredictHQ sont intégrés pour ajuster les prévisions :

```python
st.subheader('PredictHQ Events')

phq = Client(access_token="Access-Token")

if not combined_forecast_df.empty:
    gte_date = combined_forecast_df['ts'].iloc[0].strftime('%Y-%m-%d')
    lte_date = combined_forecast_df['ts'].iloc[-1].strftime('%Y-%m-%d')
    categories = ["sports", "public-holidays"]

    events = phq.events.search(category="public-holidays", rank={'gte': 90, 'lt': 100}, active={
        'lte': f'{lte_date}',
        'gte': f'{gte_date}',
        'tz': 'UTC'
    })
    sports = phq.events.search(category="sports", rank={'gte': 90, 'lt': 100}, active={
        'lte': f'{lte_date}',
        'gte': f'{gte_date}',
        'tz': 'UTC'
    })

    events_list = []
    for event in events:
        events_list.append({
            "Title": event.title,
            "Category": event.category,
            "Rank": event.rank,
            "Start Date": event.start.strftime('%Y-%m-%d'),
            "End Date": event.end.strftime('%Y-%m-%d')
        })
    for event in sports:
        events_list.append({
            "Title": event.title,
            "Category": event.category,
            "Rank": event.rank,
            "Start Date": event.start.strftime('%Y-%m-%d'),
            "End Date": event.end.strftime('%Y-%m-%d')
        })

    events_df = pd.DataFrame(events_list)
    st.subheader('Events DataFrame')
    st.write(events_df)

    for event_date in events_df['Start Date']:
        combined_forecast_df.loc[combined_forecast_df['ts'] == event_date, combined_forecast_df.columns != 'ts'] *= 1.10

    st.subheader('Adjusted Forecast DataFrame')
    st.write(combined_forecast_df)

    combined_forecast_df.to_csv('adjusted_combined_forecast.csv', index=False)
    st.write("Adjusted forecasts saved to CSV file")

```

1. Le client PredictHQ est initialisé avec un jeton d'accès.
2. Les événements PredictHQ pour les catégories "sports" et "public-holidays" sont récupérés pour la période des prévisions.
3. Les événements sont stockés dans un DataFrame.
4. Les prévisions sont ajustées en augmentant de 10 % les valeurs des dates correspondant aux événements.

## Tracé des Prévisions Ajustées

Les prévisions ajustées sont tracées :

```python
fig = go.Figure()

for column in combined_forecast_df.columns:
    if column != 'ts':
        fig.add_trace(go.Scatter(x=combined_forecast_df['ts'], y=combined_forecast_df[column], mode='lines+markers', name=column))

fig.update_layout(
    title='Adjusted Combined Forecast',
    xaxis_title='Date',
    yaxis_title='Forecast Values',
    hovermode='x'
)

st.plotly_chart(fig)

```

1. Un graphique est créé pour chaque colonne des prévisions ajustées.
2. Les prévisions ajustées sont affichées avec Plotly.

## Résumé

Cette application Streamlit permet de télécharger des fichiers Parquet, de les combiner et de réaliser des prévisions de séries temporelles avec Prophet. Elle intègre également des événements PredictHQ pour ajuster les prévisions en fonction des événements futurs. Les prévisions sont affichées et peuvent être téléchargées en tant que fichiers CSV.

---