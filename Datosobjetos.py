import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import sweetviz as sv
import streamlit.components.v1 as components

# Configuración de la página
st.set_page_config(page_title="Sensor Óptico de Objetos Reales", layout="wide")

st.title("📷 Sensor Óptico IoT - Captura de Objetos Reales")
st.markdown("""
Esta aplicación utiliza la **Cámara Web como un sensor óptico de visión artificial**. 
Coloca cualquier objeto que tengas a la mano frente a la cámara. 
El sistema extraerá las propiedades físicas, geométricas y de color del objeto para estructurarlas en un DataFrame.
""")

# --- 1. ADQUISICIÓN DE DATOS MEDIANTE EL SENSOR ÓPTICO ---
st.sidebar.header("🔌 Captura del Hardware")
st.sidebar.write("Coloca un objeto frente a tu cámara web en un fondo despejado.")

foto_sensor = st.sidebar.camera_input("📸 Capturar objeto real")

# Inicializar la variable en el estado de la sesión si no existe
if 'datos_objeto_real' not in st.session_state:
    st.session_state['datos_objeto_real'] = None

if foto_sensor is not None:
    with st.spinner("Sensor analizando el objeto físico..."):
        img_pil = Image.open(foto_sensor)
        img_np = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        _, umbral = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lista_areas, lista_perimetros, lista_rojo, lista_verde, lista_azul, lista_brillo = [], [], [], [], [], []
        
        for idx, c in enumerate(contornos):
            area = cv2.contourArea(c)
            if area > 100:
                perimetro = cv2.arcLength(c, True)
                mascara = np.zeros(gris.shape, dtype=np.uint8)
                cv2.drawContours(mascara, [c], -1, 255, -1)
                media_colores = cv2.mean(img_bgr, mask=mascara)
                
                lista_areas.append(area)
                lista_perimetros.append(perimetro)
                lista_azul.append(round(media_colores[0], 2))
                lista_verde.append(round(media_colores[1], 2))
                lista_rojo.append(round(media_colores[2], 2))
                lista_brillo.append(round(cv2.mean(gris, mask=mascara)[0], 2))
        
        if len(lista_areas) > 0:
            df_capturado = pd.DataFrame({
                "ID_Segmento": [f"Parte_{i+1}" for i in range(len(lista_areas))],
                "Area_Pixeles": lista_areas,
                "Perimetro_Pixeles": lista_perimetros,
                "Intensidad_Rojo": lista_rojo,
                "Intensidad_Verde": lista_verde,
                "Intensidad_Azul": lista_azul,
                "Brillo_Promedio": lista_brillo
            })
            # Guardamos rigurosamente en el session_state
            st.session_state['datos_objeto_real'] = df_capturado

# Asignamos la variable principal SIEMPRE desde el session_state
df = st.session_state['datos_objeto_real']

# --- DESARROLLO DE LAS ACTIVIDADES REQUERIDAS ---
if df is not None:
    st.success("✔️ ¡Propiedades físicas del objeto capturadas con éxito!")
    
    # 2. TABLA INTERACTIVA
    st.header("📋 1. Tabla Interactiva de Datos del Objeto")
    st.dataframe(df, use_container_width=True)
    st.subheader("🔍 Detección de Registros Duplicados")
    
    cantidad_duplicados = df.duplicated().sum()
    
    if cantidad_duplicados > 0:
        st.warning(f"⚠️ Se han detectado {cantidad_duplicados} filas idénticas en la captura.")
        st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
    else:
        st.success("✔️ No se encontraron filas duplicadas. Cada segmento detectado tiene propiedades únicas.")
    
    # 3. INFORMACIÓN DESCRIPTIVA
    st.header("📉 2. Información Descriptiva Básica")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Segmentos Detectados (Filas)", df.shape[0])
    with col2:
        st.metric("Variables Físicas (Columnas)", df.shape[1])
    with col3:
        st.metric("Valores Nulos", df.isnull().sum().sum())
    with col4:
        st.metric("Área Máxima Detectada", f"{int(df['Area_Pixeles'].max())} px")
        
    st.subheader("Tipos de Datos y Validación de Nulos")
    info_df = pd.DataFrame({
        "Tipo de Dato": df.dtypes.astype(str),
        "Valores Faltantes": df.isnull().sum(),
        "％ de Nulos": (df.isnull().sum() / len(df) * 100).round(2).astype(str) + " ％"
    })
    st.table(info_df)
    
    st.subheader("Estadísticas Descriptivas de las Propiedades del Objeto")
    st.dataframe(df.describe().T, use_container_width=True)
    
    # 4. REPORTE DE PERFILADO CON SWEETVIZ
    st.header("🧬 3. Reporte de Perfilado Automatizado (Sweetviz)")
    
    if st.button("Generar Reporte Sweetviz"):
        with st.spinner("Generando reporte HTML interactivo..."):
            reporte = sv.analyze(df)
            reporte.show_html(filepath="reporte_sweetviz.html", open_browser=False)
            
            with open("reporte_sweetviz.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            
            components.html(html_content, height=700, scrolling=True)
            st.success("¡Reporte interactivo desplegado!")

    # --- 5. ÚLTIMO PASO: GUARDAR DATOS LIMPIOS ---
    st.header("💾 4. Guardar Datos Verificados y Limpios")
    st.markdown("Una vez auditadas las dimensiones (`df.shape`) y el resumen estadístico (`df.describe()`), descarga el dataset final:")
    
    # Conversión segura garantizada
    csv_limpio = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Descargar datos_limpios.csv",
        data=csv_limpio,
        file_name="datos_limpios.csv",
        mime="text/csv",
        help="Haz clic aquí para descargar el archivo CSV validado con las propiedades del objeto."
    )
    
else:
    st.info("💡 Para comenzar, enciende la cámara en el panel izquierdo y captura el objeto.")