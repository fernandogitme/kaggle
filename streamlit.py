import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data

def load_data():
    return pd.read_csv("archive/KAG_conversion_data.csv")


def encoder_one_hot(df,columns_to_encode):
    encoder = OneHotEncoder()

    encoded_cols = encoder.fit_transform(df[columns_to_encode])

    new_columns = encoder.get_feature_names_out(columns_to_encode)

    # Nuevo DataFrame con las columnas codificadas
    encoded_df = pd.DataFrame(encoded_cols.toarray(), columns=new_columns, index=df.index)

    # Concatenar el DataFrame original con las nuevas columnas codificadas
    result_df = pd.concat([df.drop(columns=columns_to_encode), encoded_df], axis=1)

    return result_df

def detect_outliers_mad(df, column, threshold=3.5):

    numeric_data = pd.to_numeric(df[column], errors='coerce')
    
    
    if len(numeric_data) == 0:
        return df, np.nan  # Retorna el DataFrame original si no hay datos válidos
    
    median = numeric_data.median()
    
    # Calculate MAD
    mad = np.median(np.abs(numeric_data - median))
    
    if mad == 0:
        return df, 0  # Retorna el DataFrame original si MAD es 0
    
    # Calculate the modified Z-score
    modified_z_score = 0.6745 * (numeric_data - median) / mad
    
    # Filter out outliers
    mask = abs(modified_z_score) < threshold
    df_no_outliers = df[mask]
    
    return df_no_outliers, mad

def crear_heatmap_correlaciones(df, variables_x=None, variables_y=None, height=600):
    # Calcular la matriz de correlación completa
    corr_matrix = df.corr()
    
    # Si no se especifican variables, usar todas
    if variables_x is None:
        variables_x = corr_matrix.columns
    if variables_y is None:
        variables_y = corr_matrix.columns
    
    # Filtrar la matriz de correlación
    corr_filtered = corr_matrix.loc[variables_y, variables_x]
    
    # Crear el heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_filtered.values,
        x=corr_filtered.columns,
        y=corr_filtered.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Correlación: %{z}<extra></extra>'
    ))

    # Personalización del layout
    fig.update_layout(
        title='Heatmap de Correlaciones',
        xaxis_title='Variables X',
        yaxis_title='Variables Y',
        height=height,  # Ajusta la altura aquí
        coloraxis_colorbar=dict(
            title="Correlación"
        )
    )
    
    return fig

def main():
    st.set_page_config(page_title="Visualizador de campañas", layout="wide")
    
    df = load_data()

    # Coste por mil impresiones (CPM)
    df['CPM'] = np.where(df['Impressions'] != 0, (df['Spent'] / df['Impressions']) * 1000, 0)

    # Coste por lead (CPL)
    df['CPL'] = np.where(df['Total_Conversion'] != 0, df['Spent'] / df['Total_Conversion'], 0)

    # Coste por venta (CPA)
    df['CPA'] = np.where(df['Approved_Conversion'] != 0, df['Spent'] / df['Approved_Conversion'], 0)

    # Tasa de clics (CTR)
    df['CTR'] = np.where(df['Impressions'] != 0, (df['Clicks'] / df['Impressions']) * 100, 0)

    # Coste por clic (CPC)
    df['CPC'] = np.where(df['Clicks'] != 0, df['Spent'] / df['Clicks'], 0)

    # Tasa de conversión (Conversion Rate) para leads y ventas
    df['Conversion_Rate_Lead'] = np.where(df['Impressions'] != 0, (df['Total_Conversion'] / df['Impressions']) * 100, 0)
    df['Conversion_Rate_Sale'] = np.where(df['Impressions'] != 0, (df['Approved_Conversion'] / df['Impressions']) * 100, 0)

    # Porcentaje de Total_Conversion en base a Spent
    df['Pct_Spent_per_Conversion'] = np.where(df['Total_Conversion'] != 0, df['Spent'] / df['Total_Conversion'], 0)

    # Porcentaje de Total_Conversion en base a Clicks
    df['Pct_Conversion_per_Click'] = np.where(df['Clicks'] != 0, df['Total_Conversion'] / df['Clicks'], 0)

    # Tasa de conversión aprobada
    df['Approved_Conversion_Rate'] = np.where(df['Clicks'] != 0, df['Approved_Conversion'] / df['Clicks'], 0)

    # Porcentaje de Approved_Conversion en base a Total_Conversion
    df['Pct_Approved_Conversion'] = np.where(df['Total_Conversion'] != 0, df['Approved_Conversion'] / df['Total_Conversion'], 0)

    st.subheader(":dart: **Visualizador de campañas.**")
    st.write("Aquí podrás ver una serie de filtros donde puedes ver las métricas más importantes.\nDespués puedes meter las métricas y ver cómo ha ido.")

    tab1, tab2, tab3, tab4 = st.tabs(["Visualizador", "Benchmark", "Modelo", "Recomendador"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            campañas = df["xyz_campaign_id"].unique().tolist()
            campañas_seleccionadas = st.multiselect("Campañas", campañas, campañas)

            df_filtrado = df[df["xyz_campaign_id"].isin(campañas_seleccionadas)]

            generos = df_filtrado["gender"].unique().tolist()
            generos_seleccionados = st.multiselect("Géneros", generos, generos)

            df_filtrado = df_filtrado[df_filtrado["gender"].isin(generos_seleccionados)]

        with col2:
            edades = df_filtrado["age"].unique().tolist()
            edades_seleccionadas = st.multiselect("Edades", edades, edades)

            df_filtrado = df_filtrado[df_filtrado["age"].isin(edades_seleccionadas)]

            intereses = df_filtrado["interest"].unique().tolist()
            intereses_seleccionados = st.multiselect("Intereses", intereses, intereses)

            df_filtrado = df_filtrado[df_filtrado["interest"].isin(intereses_seleccionados)]

        _, col_central, _ = st.columns([1, 18, 1])  # Crear tres columnas, la del medio más ancha. No puede ser más de 20 la suma.
        
        with col_central:
            st.divider()
            col_izq, col_der = st.columns([2, 2])
            with col_izq:
                st.markdown('''*Podemos decidir si quitar o no la **ventana de atribución** para que las estadísticas sean más realistas.  
                            Hay que tener en cuenta que esto solo se puede hacer con las ventanas de atribución donde el gasto es 0 y hay conversiones.  Las ventanas donde hay gasto y conversiones no las podemos quitar por que no están los datos etiquetados.*''')
                on = st.toggle("¿Quitar ventana de atribución?")
            with col_der:
                 st.markdown('''*Esta funcion es para quitar los valores que son muy inusuales. Puedes definir el umbral de filtrado 
                            si quieres pero está ajustado a 3.5 automaticamente. No hay funcionalidad de aplicar a más de una columna. 
                            Ten en cuenta que si aplicas también la ventana de atribución los datos filtrados serán distintos que si aplicas 
                            solo los outliers.*''')

                 outliers = st.toggle("¿Quitar outliers?")
                 if outliers:
                     with st.expander("Opciones configuracion de valores atipicos"):
                         columna = st.selectbox("Selecciona una de las columnas para filtrar valores atipicos",("Impressions","Clicks","Spent"))
                         valor = st.slider("Define el umbral(treshold) del filtrado. Mientras mas grande sea el valor, menos valores atipicos detectará.",min_value=0.0, max_value=7.0, value=3.5, step=0.1)

                #st.selectbox("Filtros outliers",("Email", "Home phone", "Mobile phone"))
            st.divider()


            if on:
                df_filtrado = df_filtrado[~((df_filtrado["Total_Conversion"] >= 1) & (df_filtrado["Spent"] == 0))]
            if outliers:
                # Uso de funcion outliers (Zscore, IQR3 y otro menos conocido: Metodo de Desviación Absoluta Mediana (MAD))
                # Usaremos el ultimo que es el que mejor se comporta con el distribuciones que no son normales y valores atipicos.
                df_filtrado, nada = detect_outliers_mad(df_filtrado,str(columna),valor)
            st.write("")


            with st.expander("Datos brutos de filtro"):
                st.dataframe(df_filtrado, use_container_width=True) 
            
            
            with st.expander("Datos estadisticos del filtro:",expanded=False):
                st.dataframe(df_filtrado.iloc[:,6:].describe().T.drop("min",axis=1), use_container_width=True) 

            columns_to_encode = ['xyz_campaign_id','age', 'gender', 'interest']

            df_to_encode = df_filtrado.drop(["ad_id","fb_campaign_id"],axis=1)

            df_result = encoder_one_hot(df_to_encode,columns_to_encode)
            with st.expander("Datos con conversion a columnas categoricas",expanded=False):
                st.dataframe(df_result, use_container_width=True)
            st.divider()

            columnas_x = df_result.columns.tolist()
            COLUMNAS_Y = ['Impressions', 'Clicks', 'Spent', 'Total_Conversion',
                        'Approved_Conversion', 'CPM', 'CPL', 'CPA', 'CTR', 'CPC',
                        'Conversion_Rate_Lead', 'Conversion_Rate_Sale',
                        'Pct_Spent_per_Conversion', 'Pct_Conversion_per_Click',
                        'Approved_Conversion_Rate', 'Pct_Approved_Conversion']
            
            
            cols  = sorted(list(set(columnas_x)- set(COLUMNAS_Y)))
            

            cols_interes_x = st.multiselect("Selecciona las variables que te gustaría revisar. Tenga en cuenta que se **usarán los filtros que seleccionó anteriormente** para ver las correlaciones.", cols, None)
            
            if cols_interes_x:
                altura_figura = st.slider("Altura de la figura", min_value=400, max_value=2000, value=800, step=50)

                # Limpiar el espacio para el gráfico anterior
                graph_placeholder = st.empty()

                # Mostrar el gráfico basado en la selección de columnas X
                fig = crear_heatmap_correlaciones(df_result, variables_x=cols_interes_x, variables_y=COLUMNAS_Y, height=altura_figura)
                graph_placeholder.plotly_chart(fig, use_container_width=True, height=altura_figura)

                # Permitir al usuario seleccionar variables Y específicas
                st.write("Ahora puedes seleccionar variables Y específicas:")
                cols_interes_y = st.multiselect("Selecciona las variables Y que te gustaría revisar", columnas_x, default=COLUMNAS_Y)

                if cols_interes_y:
                    # Actualizar el gráfico con la selección de columnas Y
                    fig = crear_heatmap_correlaciones(df_result, variables_x=cols_interes_x, variables_y=cols_interes_y, height=altura_figura)
                    graph_placeholder.plotly_chart(fig, use_container_width=True, height=altura_figura)
                   

    # Contenido para las otras pestañas
    with tab2:
        st.write("Contenido de la pestaña Benchmark")

    with tab3:
        st.write("Contenido de la pestaña Modelo")

    with tab4:
        st.write("Contenido de la pestaña Recomendador")

if __name__ == "__main__":
    main()