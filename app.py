import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import requests
import unicodedata

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="AgroRecomienda IA", page_icon="🌱", layout="wide")
st.title("🌱 Asistente Agrónomo con Random forest")
st.write("Ingresa las propiedades fisicoquímicas y geográficas de tu suelo para recibir las 3 mejores recomendaciones.")

# --- 2. CARGAR MODELOS Y DATOS ---
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_cultivos.pkl')
    columnas = joblib.load('columnas_modelo.pkl')
    clases = joblib.load('clases_modelo.pkl')
    return modelo, columnas, clases

@st.cache_data
def cargar_geojson():
    url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json"
    respuesta = requests.get(url)
    return respuesta.json()

modelo_rf, columnas_modelo, clases_modelo = cargar_modelo()
geojson_colombia = cargar_geojson()

descripciones = {
    "Café": "Ideal para suelos con pH ligeramente ácido. Alto valor comercial.",
    "Cacao": "Requiere suelos profundos y ricos en potasio. Excelente para climas cálidos.",
    "Aguacate": "Exigente en buen drenaje. Alta demanda de exportación.",
    "Caña Panelera": "Muy rústica, requiere buen fósforo para desarrollo de raíces.",
    "Plátano": "Extrae muchos nutrientes, especialmente potasio.",
    "Arroz": "Se adapta a suelos más pesados y soporta variaciones en la acidez.",
    "Maíz": "Exigente en nitrógeno y materia orgánica. Cultivo de ciclo corto.",
    "Mora": "Prefiere suelos francos. Gran oportunidad en mercados locales.",
    "Papa de año": "Requiere suelos sueltos y bien fertilizados.",
    "Caucho": "Se adapta a suelos ácidos, ideal para proyectos a largo plazo."
}

def normalizar_texto(texto):
    texto = str(texto).upper()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto.strip()

# --- 3. INTERFAZ DIVIDIDA EN DOS COLUMNAS ---
col_form, col_map = st.columns([1.2, 1])

with col_form:
    st.subheader("📍 Datos de Ubicación y Terreno")
    col1, col2, col3 = st.columns(3)

    deptos = [c.replace('Departamento_', '') for c in columnas_modelo if c.startswith('Departamento_')]
    topos = [c.replace('Topografia_', '') for c in columnas_modelo if c.startswith('Topografia_')]
    riegos = [c.replace('Riego_', '') for c in columnas_modelo if c.startswith('Riego_')]

    with col1:
        departamento = st.selectbox("Departamento", deptos)
    with col2:
        topografia = st.selectbox("Topografía", topos)
    with col3:
        riego = st.selectbox("Tipo de Riego", riegos)

    st.subheader("🧪 Análisis Fisicoquímico del Suelo")
    col4, col5 = st.columns(2)

    with col4:
        ph = st.number_input("pH (escala 0-14)", min_value=0.0, max_value=14.0, value=5.5, step=0.1)
        mo = st.number_input("Materia Orgánica (%)", min_value=0.0, value=4.0, step=0.1)
        fosforo = st.number_input("Fósforo Bray II (ppm o mg/kg)", min_value=0.0, value=15.0, step=1.0)

    with col5:
        calcio = st.number_input("Calcio interc. (cmol(+)/kg)", min_value=0.0, value=3.0, step=0.1)
        magnesio = st.number_input("Magnesio interc. (cmol(+)/kg)", min_value=0.0, value=1.0, step=0.1)
        potasio = st.number_input("Potasio interc. (cmol(+)/kg)", min_value=0.0, value=0.5, step=0.1)

    btn_recomendar = st.button("🔍 Generar Recomendación", use_container_width=True)

with col_map:
    st.subheader("🗺️ Ubicación del Terreno")
    m = folium.Map(location=[4.5709, -74.2973], zoom_start=5, tiles="cartodbpositron")
    depto_norm = normalizar_texto(departamento)
    
    def estilo_mapa(feature):
        nombre_mapa = normalizar_texto(feature['properties'].get('NOMBRE_DPT', ''))
        if depto_norm in nombre_mapa or nombre_mapa in depto_norm:
            return {'fillColor': '#2E7D32', 'color': 'black', 'weight': 2, 'fillOpacity': 0.8} # Cambié el rojo por un verde oscuro elegante
        else:
            return {'fillColor': '#cccccc', 'color': 'white', 'weight': 1, 'fillOpacity': 0.4}

    folium.GeoJson(geojson_colombia, style_function=estilo_mapa).add_to(m)
    st_folium(m, width=500, height=450, returned_objects=[])

# --- 4. RESULTADOS DE LA PREDICCIÓN ---
st.markdown("---")

if btn_recomendar:
    input_data = pd.DataFrame(np.zeros((1, len(columnas_modelo))), columns=columnas_modelo)
    
    input_data['pH agua:suelo'] = ph
    input_data['Materia organica'] = mo
    input_data['Fósforo Bray II'] = fosforo
    input_data['Calcio intercambiable'] = calcio
    input_data['Magnesio intercambiable'] = magnesio
    input_data['Potasio intercambiable'] = potasio
    
    if f'Departamento_{departamento}' in columnas_modelo:
        input_data[f'Departamento_{departamento}'] = 1
    if f'Topografia_{topografia}' in columnas_modelo:
        input_data[f'Topografia_{topografia}'] = 1
    if f'Riego_{riego}' in columnas_modelo:
        input_data[f'Riego_{riego}'] = 1

    probabilidades = modelo_rf.predict_proba(input_data)[0]
    resultados = list(zip(clases_modelo, probabilidades))
    resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)

    st.success("¡Análisis completado! Aquí tienes las mejores opciones para tu terreno:")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    cols_resultados = [col_res1, col_res2, col_res3]
    
    for i in range(3):
        cultivo = resultados_ordenados[i][0]
        prob = resultados_ordenados[i][1] * 100
        desc = descripciones.get(cultivo, "Excelente opción comercial para estas condiciones.")
        
        with cols_resultados[i]:
            st.info(f"### #{i+1} {cultivo}\n**Coincidencia:** {prob:.1f}%\n\n💡 *{desc}*")

# --- 5. FIRMA Y CONTACTO ---
st.markdown("<br><br>", unsafe_allow_html=True) # Espacio en blanco
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #555;'>
        <p>Desarrollado con Machine Learning 🤖 basado en datos de suelos de Colombia.</p>
        <p><strong>Desarrollado por:</strong> Joan Giraldo | <strong>Contacto:</strong> <a href='mailto:giraldojoan@gmail.com'>giraldojoan@gmail.com</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)
