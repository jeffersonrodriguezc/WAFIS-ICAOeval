import os
import csv
import random
import shutil
from pathlib import Path

def reorganizar_dataset(ruta_csv, carpeta_dataset, carpeta_templates, carpeta_test):
    """
    Reorganiza un dataset de imágenes basado en un archivo CSV.

    Args:
        ruta_csv (str): La ruta al archivo 'people.csv'.
        carpeta_dataset (str): La ruta a la carpeta que contiene las carpetas de cada persona.
        carpeta_templates (str): La ruta a la carpeta donde se guardarán las imágenes de template.
        carpeta_test (str): La ruta a la carpeta donde se guardarán las imágenes de prueba.
    """

    # --- 1. Crear las carpetas de destino si no existen ---
    os.makedirs(carpeta_templates, exist_ok=True)
    os.makedirs(carpeta_test, exist_ok=True)
    print(f"Carpetas '{carpeta_templates}' y '{carpeta_test}' aseguradas.")

    # --- 2. Leer y filtrar los datos del archivo CSV ---
    personas_filtradas = {}
    counter = 0
    with open(ruta_csv, 'r') as f:
        lector_csv = csv.reader(f)
        next(lector_csv)  # Omitir la cabecera
        for i, row in enumerate(lector_csv):
            # --- CORRECCIÓN ---
            # Nos aseguramos de que la fila no esté vacía y tenga 2 elementos
            if not row or len(row) < 2:
                continue # Si está vacía, la saltamos y vamos a la siguiente

            nombre, num_imagenes = row
            # También nos aseguramos de que 'num_imagenes' no sea una cadena vacía
            if num_imagenes and int(num_imagenes) >= 2: # <-- LÍNEA MODIFICADA
                # --- 3. Generar un ID único ---
                iniciales = "".join([n[0] for n in nombre.split('_')])
                id_unico = f"{iniciales}{counter}"
                personas_filtradas[nombre] = id_unico
                counter += 1
    
    print(f"Se procesarán {len(personas_filtradas)} personas con 2 o más imágenes.")

    # --- 4. Recorrer las carpetas del dataset ---
    for nombre_persona in os.listdir(carpeta_dataset):
        if nombre_persona in personas_filtradas:
            id_persona = personas_filtradas[nombre_persona]
            ruta_persona = os.path.join(carpeta_dataset, nombre_persona)
            
            if os.path.isdir(ruta_persona):
                imagenes = [img for img in os.listdir(ruta_persona) if img.endswith(('.jpg', '.jpeg', '.png'))]
                
                if len(imagenes) >= 2:
                    # --- 5. Seleccionar 2 imágenes diferentes al azar ---
                    imagen_template, imagen_test = random.sample(imagenes, 2)

                    # --- 6. Renombrar y copiar las imágenes ---
                    # Extraer la extensión y el número de las imágenes originales
                    num_template = os.path.splitext(imagen_template)[0].split('_')[-1]
                    ext_template = os.path.splitext(imagen_template)[1]

                    #print(f"Imagen template seleccionada: {imagen_template} (Número: {num_template}, Extensión: {ext_template})")
                    
                    num_test = os.path.splitext(imagen_test)[0].split('_')[-1]
                    ext_test = os.path.splitext(imagen_test)[1]

                    #print(f"Imagen test seleccionada: {imagen_test} (Número: {num_test}, Extensión: {ext_test})")

                    # Crear los nuevos nombres de archivo
                    nuevo_nombre_template = f"{id_persona}{ext_template}"
                    nuevo_nombre_test = f"{id_persona}_{num_test}{ext_test}"

                    if nuevo_nombre_template == 'A1161.jpg':
                        print("ENCONTRADO")
                        print(nombre_persona, ruta_persona, imagenes)

                        print(f"Renombrando '{imagen_template}' a '{nuevo_nombre_template}'")
                        print(f"Renombrando '{imagen_test}' a '{nuevo_nombre_test}'")

                    # Copiar las imágenes a las carpetas de destino
                    shutil.copy2(os.path.join(ruta_persona, imagen_template), os.path.join(carpeta_templates, nuevo_nombre_template))
                    shutil.copy2(os.path.join(ruta_persona, imagen_test), os.path.join(carpeta_test, nuevo_nombre_test))
                    
                    #print(f"Procesado: {nombre_persona} -> ID: {id_persona}")

    print("\n¡Proceso de reorganización completado! ✅")


# --- CONFIGURACIÓN ---
# Por favor, reemplaza estas rutas con las tuyas

# 1. Ruta al archivo people.csv (Asegúrate de que esta sea la ruta correcta)
ruta_csv_personas = Path(r'C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\datasets\LFW\raw\people.csv')

# 2. Ruta a la carpeta principal que contiene las carpetas de cada persona
ruta_carpeta_dataset = Path(r'C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\datasets\LFW\raw\lfw-deepfunneled\lfw-deepfunneled')

# 3. Ruta a la carpeta donde guardarás las imágenes "template"
ruta_carpeta_templates = Path(r'C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\datasets\LFW\processed\templates')

# 4. Ruta a la carpeta donde guardarás las imágenes "test"
ruta_carpeta_test = Path(r'C:\Users\JotaR\Documents\Github\WAFIS-ICAOeval\datasets\LFW\processed\test')


# --- EJECUTAR EL SCRIPT ---
if __name__ == '__main__':
    # Validar que el archivo CSV y la carpeta del dataset existan antes de ejecutar
    if not os.path.exists(ruta_csv_personas):
        print(f"Error: El archivo '{ruta_csv_personas}' no se encuentra. Por favor, verifica la ruta.")
    elif not os.path.exists(ruta_carpeta_dataset):
        print(f"Error: La carpeta del dataset '{ruta_carpeta_dataset}' no se encuentra. Por favor, verifica la ruta.")
    else:
        reorganizar_dataset(ruta_csv_personas, ruta_carpeta_dataset, ruta_carpeta_templates, ruta_carpeta_test)