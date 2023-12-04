import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class Process:
    def __init__(self):
        self.loaded_model = joblib.load('archivos/modelo_decision_tree.joblib')
        self.features = ['FAMI_EDUCACIONPADRE','FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR','FAMI_ESTRATOVIVIENDA','FAMI_PERSONASHOGAR','ESTU_HORASSEMANATRABAJA','ESTU_GENERO_-','ESTU_GENERO_F','ESTU_GENERO_M']

    def get_model_result_one(self,request):
        datos_formulario = request.form.to_dict()
        # Crear un DataFrame de pandas a partir de los datos del formulario
        df = pd.DataFrame.from_dict(datos_formulario, orient='index')
        df_transpuesto = df.transpose()
        print(df_transpuesto)
        # Hacer algo con el DataFrame, por ejemplo, imprimirlo
        if (request.form['ESTU_GENERO'] == "M"):
            df_transpuesto['ESTU_GENERO_M'] = "1"
            df_transpuesto['ESTU_GENERO_F'] = "0"
            df_transpuesto['ESTU_GENERO_-'] = "0"
        else:
            df_transpuesto['ESTU_GENERO_M'] = "0"
            df_transpuesto['ESTU_GENERO_F'] = "1"
            df_transpuesto['ESTU_GENERO_-'] = "0" 
        
        df_transpuesto = df_transpuesto[['FAMI_EDUCACIONPADRE',	'FAMI_EDUCACIONMADRE','FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR','FAMI_ESTRATOVIVIENDA','FAMI_PERSONASHOGAR','ESTU_HORASSEMANATRABAJA','ESTU_GENERO_-','ESTU_GENERO_F','ESTU_GENERO_M']]
        return self.predict_model(df_transpuesto)['POR_ENCIMA_MEDIA'].iloc[0],request.form

    def get_model_result_multiple(self,request):
        if 'file' in request.files:
            archivo_cargado = request.files['file']
            if archivo_cargado.filename != '':
                df = pd.read_csv(archivo_cargado)
                df_filtrado = df[['ESTU_GENERO','FAMI_EDUCACIONPADRE','FAMI_EDUCACIONMADRE','FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR','FAMI_ESTRATOVIVIENDA','FAMI_PERSONASHOGAR','ESTU_HORASSEMANATRABAJA']]
                df_filtrado = pd.get_dummies(df_filtrado, columns=['ESTU_GENERO'])
                if not 'ESTU_GENERO_-' in df_filtrado:
                    df_filtrado['ESTU_GENERO_-'] = "0" 
                df_filtrado = df_filtrado[['FAMI_EDUCACIONPADRE',	'FAMI_EDUCACIONMADRE','FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR','FAMI_ESTRATOVIVIENDA','FAMI_PERSONASHOGAR','ESTU_HORASSEMANATRABAJA','ESTU_GENERO_-','ESTU_GENERO_F','ESTU_GENERO_M']]
        return self.predict_model(df_filtrado), df_filtrado

    def predict_model(self, df_filtrado):
        new_predictions = self.loaded_model.predict(df_filtrado)
        df_filtrado['PREDICCION_PUNT_GLOBAL'] = new_predictions
        df_filtrado['POR_ENCIMA_MEDIA'] = (df_filtrado['PREDICCION_PUNT_GLOBAL'] == 1)
        print(df_filtrado[['PREDICCION_PUNT_GLOBAL', 'POR_ENCIMA_MEDIA']])
        return df_filtrado[['PREDICCION_PUNT_GLOBAL', 'POR_ENCIMA_MEDIA']]
    
    def get_torta(self, new_data):
        # Supongamos que tienes un DataFrame llamado new_data con las columnas 'POR_ENCIMA_MEDIA' y 'PUNT_GLOBAL'
        # Asegúrate de haber agregado estas columnas después de las predicciones del modelo

        # Contar la cantidad de registros por encima y por debajo de la media
        counts = new_data['POR_ENCIMA_MEDIA'].value_counts()

        # Configurar el estilo de matplotlib con tonos pastel
        plt.style.use('seaborn-pastel')

        # Crear una gráfica de torta
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=['Encima','Debajo'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
        plt.title('Registros por Encima y por Debajo de la Media')
        plt.savefig("static/torta.jpg")

    def get_distribution(self):

        # Obtener las importancias de las características del modelo
        feature_importances = self.loaded_model.feature_importances_
        # Supongamos que tienes una lista de nombres de características en features
        #features = ['ESTU_GENERO', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'PUNT_GLOBAL']

        # Configurar el estilo de matplotlib con tonos pastel
        plt.style.use('seaborn-pastel')

        # Crear una paleta de colores pastel con la misma longitud que features
        colores_pastel = plt.cm.Paired(np.arange(len(self.features )))

        # Crear un gráfico de barras de las importancias de las características
        plt.figure(figsize=(10, 6))
        barras = plt.bar(self.features , feature_importances, color=colores_pastel[:len(self.features )])  # Ajustar la paleta de colores

        # Configurar la leyenda con los nombres de las características y sus colores
        plt.legend(barras, self.features , loc='upper right')

        plt.title('Importancia de las Características')
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.xticks([])
        plt.savefig("static/distribution.jpg")

    def get_multiple_graphic(self, df):
        plt.style.use('seaborn-pastel')
        df = df[['FAMI_EDUCACIONPADRE',	'FAMI_EDUCACIONMADRE','FAMI_TIENEINTERNET','FAMI_TIENECOMPUTADOR','FAMI_ESTRATOVIVIENDA','FAMI_PERSONASHOGAR','ESTU_HORASSEMANATRABAJA','ESTU_GENERO_F','ESTU_GENERO_M']]
        num_columnas = df.shape[1]
        # Calcular el número de filas y columnas para el diseño de subgráficos
        num_filas = (num_columnas // 2) + (1 if num_columnas % 2 != 0 else 0)
        num_columnas_subplot = min(2, num_columnas)

        # Crear un gráfico de barras para cada columna
        plt.figure(figsize=(15, 5 * num_filas))
        for i, columna in enumerate(df.columns):
            plt.subplot(num_filas, num_columnas_subplot, i + 1)
            df[columna].value_counts().plot(kind='bar', color=plt.cm.Paired(range(len(df[columna].unique()))))
            plt.title(columna)
            plt.xlabel(columna)
            plt.ylabel('Frecuencia')

        plt.tight_layout()
        plt.savefig("static/total.jpg")


    def get_initial_data(self):
        info_table = self.df.describe(include='all').transpose().reset_index()
        df_head = self.df.head()
        column_names = df_head.columns.tolist()
        data_rows = df_head.values.tolist()
        na_counts = self.df.isna().sum()
        an_counts = self.df.isin(["-"]).sum()
        na_counts_list = na_counts.reset_index().rename(columns={0: 'Count'}).to_dict(orient='records')
        an_counts_list = an_counts.reset_index().rename(columns={0: 'Count'}).to_dict(orient='records')
        return {'info_table':info_table,
                            'column_names':column_names, 
                            'data_rows':data_rows,
                            'count_colums':len(column_names),
                            'count_rows':len(self.df),
                            'na_counts':na_counts_list,
                            'an_counts':an_counts_list
        }
    
    def get_matrix(self):
        df = sns.load_dataset("iris")
        numeric_columns = df.select_dtypes(include=[float, int]).columns
        correlation_matrix = df[numeric_columns].corr()
        sns.set(style="white")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, cmap="Pastel1", annot=True, linewidths=.5, ax=ax)
        plt.savefig("static/matriz_correlacion_pastel.jpg")
