################################################################################
# TÍTULO: 01_data_cleaning.R                                                   #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Limpieza de datos y creación de nuevas variables                #
# FECHA: 21 de mayo de 2025                                                    #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Subir un nivel directorio para acceder a la estructura principal del proyecto
setwd("../")

# Cargar librerías
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,  # Manipulación de datos
  lubridate,  # Manejo de fechas
  dplyr,      # Manipulación de datos
  readr       # Lectura de archivos
)

# Fijar semilla
set.seed(123)

###########################################
# 1. CARGAR BASES DE DATOS               #
###########################################

cat("\n-------------------------------------------\n")
cat("INICIANDO PROCESO DE LIMPIEZA DE DATOS\n")
cat("-------------------------------------------\n")

# Cargar datos originales
data_dir <- "stores/raw"
train_properties <- read.csv(file.path(data_dir, "train.csv"))
test_properties <- read.csv(file.path(data_dir, "test.csv"))

cat("Dimensiones originales:\n")
cat("Train:", dim(train_properties), "\n")
cat("Test:", dim(test_properties), "\n")

###########################################
# 2. FUNCIONES AUXILIARES                #
###########################################

# Función para calcular antigüedad en meses desde diciembre 2021
calculate_antiguedad <- function(month, year) {
  # Fecha de referencia: diciembre 2021
  ref_date <- as.Date("2021-12-01")
  
  # Crear fecha de publicación usando el primer día del mes
  pub_date <- as.Date(paste(year, month, "01", sep = "-"))
  
  # Calcular diferencia en meses
  antiguedad <- interval(pub_date, ref_date) %/% months(1)
  
  # Asegurar que la antigüedad no sea negativa
  antiguedad <- pmax(antiguedad, 0)
  
  return(antiguedad)
}

# Función para transformar property_type
transform_property_type <- function(property_type) {
  # 1 = casa, 0 = apartamento
  case_when(
    tolower(property_type) %in% c("house", "casa") ~ 1,
    tolower(property_type) %in% c("apartment", "apartamento") ~ 0,
    TRUE ~ 0  # Por defecto asignar 0 (apartamento) para casos no identificados
  )
}

# Función principal de limpieza
clean_dataset <- function(data, dataset_name) {
  cat(paste0("\n\n==============================================================\n"))
  cat(paste0("LIMPIANDO DATASET: ", dataset_name, "\n"))
  cat(paste0("==============================================================\n\n"))
  
  cat("Variables originales:", ncol(data), "\n")
  cat("Observaciones originales:", nrow(data), "\n\n")
  
  # Mostrar variables que serán eliminadas (removiendo lat y lon de la lista)
  variables_a_eliminar <- c("city", "surface_total", "surface_covered", "rooms", 
                            "bathrooms", "operation_type", "title", "description")
  
  cat("Variables que serán eliminadas:\n")
  for(var in variables_a_eliminar) {
    if(var %in% names(data)) {
      cat(paste0("- ", var, "\n"))
    }
  }
  
  # Definir variables a mantener según si existe price o no (incluyendo lat y lon)
  variables_base <- c("property_id", "month", "year", "bedrooms", "property_type", "lat", "lon")
  if("price" %in% names(data)) {
    variables_a_mantener <- c("property_id", "price", "month", "year", "bedrooms", "property_type", "lat", "lon")
  } else {
    variables_a_mantener <- variables_base
  }
  
  # Crear dataset limpio con las variables seleccionadas
  cleaned_data <- data %>%
    select(all_of(variables_a_mantener)) %>%
    # Crear variable antigüedad
    mutate(
      # Calcular antigüedad en meses desde diciembre 2021
      antiguedad = calculate_antiguedad(month, year),
      
      # Transformar property_type (1 = casa, 0 = apartamento)
      property_type_binary = transform_property_type(property_type)
    ) %>%
    # Eliminar variables month, year y property_type original
    select(-month, -year, -property_type) %>%
    # Renombrar para claridad
    rename(is_house = property_type_binary)
  
  cat("\nVariables en el dataset limpio:\n")
  cat(paste0("- ", names(cleaned_data), collapse = "\n- "), "\n")
  
  cat("\nDimensiones del dataset limpio:", dim(cleaned_data), "\n")
  
  return(cleaned_data)
}

# En esta sección se definen las funciones auxiliares que automatizan y encapsulan
# tareas clave del proceso de limpieza:

# 1. `calculate_antiguedad()`:
#    - Calcula la antigüedad de publicación de cada propiedad (en meses) tomando
#      como referencia diciembre de 2021. Esto permite capturar el tiempo en mercado
#      como una variable relevante para el análisis sin usar directamente `month` y `year`.

# 2. `transform_property_type()`:
#    - Convierte la variable categórica `property_type` a formato binario, donde
#      1 representa propiedades tipo casa y 0 representa apartamentos u otros.
#      Esta transformación simplifica el uso de esta variable en modelos de regresión
#      o árboles de decisión.

# 3. `clean_dataset()`:
#    - Es la función central de limpieza. Aplica transformaciones al dataset de entrada:
#      a. Elimina variables irrelevantes, constantes o con alta proporción de valores faltantes.
#      b. Calcula la antigüedad.
#      c. Codifica el tipo de propiedad.
#      d. Retorna un dataset limpio, reducido y estructurado, listo para ingeniería de variables
#         o modelado.

# Estas funciones permiten estandarizar la limpieza tanto del conjunto de entrenamiento como
# del conjunto de prueba, garantizando consistencia y reproducibilidad en el pipeline.



###########################################
# 3. APLICAR LIMPIEZA A AMBOS DATASETS   #
###########################################

# Limpiar dataset de entrenamiento
train_clean <- clean_dataset(train_properties, "ENTRENAMIENTO")

# Limpiar dataset de prueba
test_clean <- clean_dataset(test_properties, "PRUEBA")

###########################################
# 4. VERIFICACIÓN DE CALIDAD             #
###########################################

cat("\n\n==============================================================\n")
cat("VERIFICACIÓN DE CALIDAD DE LOS DATOS LIMPIOS\n")
cat("==============================================================\n\n")

# Función para verificar calidad de datos
verify_data_quality <- function(data, dataset_name) {
  cat(paste0("--- VERIFICACIÓN: ", dataset_name, " ---\n"))
  
  # Verificar valores faltantes
  missing_summary <- data %>%
    summarise_all(~ sum(is.na(.))) %>%
    gather(variable, missing_count) %>%
    mutate(missing_pct = round(missing_count / nrow(data) * 100, 2)) %>%
    arrange(desc(missing_count))
  
  cat("Resumen de valores faltantes:\n")
  print(missing_summary)
  
  # Verificar rangos de variables numéricas
  if("bedrooms" %in% names(data)) {
    cat("\nRango de bedrooms:", range(data$bedrooms, na.rm = TRUE), "\n")
  }
  
  if("antiguedad" %in% names(data)) {
    cat("Rango de antigüedad:", range(data$antiguedad, na.rm = TRUE), "\n")
  }
  
  if("lat" %in% names(data)) {
    cat("Rango de lat:", range(data$lat, na.rm = TRUE), "\n")
  }
  
  if("lon" %in% names(data)) {
    cat("Rango de lon:", range(data$lon, na.rm = TRUE), "\n")
  }
  
  if("is_house" %in% names(data)) {
    cat("Distribución de is_house:\n")
    print(table(data$is_house, useNA = "ifany"))
  }
  
  # Verificar estadísticas de price solo si existe y tiene valores válidos
  if("price" %in% names(data) && sum(!is.na(data$price)) > 0) {
    cat("Estadísticas de price:\n")
    price_values <- data$price[!is.na(data$price)]
    cat("Min:", min(price_values), "\n")
    cat("Max:", max(price_values), "\n")
    cat("Media:", round(mean(price_values), 2), "\n")
    cat("Mediana:", round(median(price_values), 2), "\n")
    cat("Valores válidos:", length(price_values), "de", nrow(data), "\n")
  } else if("price" %in% names(data)) {
    cat("Variable price existe pero no tiene valores válidos.\n")
  } else {
    cat("Variable price no existe en este dataset (esperado para dataset de test).\n")
  }
  
  cat("\n")
}

verify_data_quality(train_clean, "TRAIN")
verify_data_quality(test_clean, "TEST")

# ==============================================================
# LIMPIEZA Y VERIFICACIÓN DE DATOS
# ==============================================================

# El proceso de limpieza aplicado a los conjuntos de entrenamiento y prueba permitió depurar 
# las variables más relevantes para el modelado, manteniendo una estructura homogénea, libre de 
# valores faltantes y adecuada para el pipeline de predicción.

# Para ambos conjuntos:
# - Se eliminaron variables con alta proporción de valores ausentes o sin valor predictivo 
#   ('surface_total', 'surface_covered', 'rooms', 'bathrooms', 'operation_type', 'title', 'description').
# - Se eliminaron variables constantes ('city').
# - Se crearon dos nuevas variables derivadas:
#     * 'antiguedad': número de meses transcurridos desde diciembre de 2021.
#     * 'is_house': codificación binaria de 'property_type' (1 = casa, 0 = apartamento).

# Dataset limpio de entrenamiento:
# - Dimensiones finales: 38,644 observaciones × 7 variables.
# - No se presentan valores faltantes en ninguna variable.
# - 'bedrooms' varía entre 0 y 11, mientras que 'antiguedad' cubre de 4 a 32 meses.
# - Las coordenadas geográficas ('lat' y 'lon') están completas y en rangos esperados para Bogotá.
# - La distribución de 'is_house' muestra que el 24.5% de las observaciones corresponden a casas.
# - La variable objetivo 'price' tiene un rango amplio (300M a 1650M COP), con una media de ~654M y mediana de ~560M.

# Dataset limpio de prueba:
# - Dimensiones finales: 10,286 observaciones × 7 variables.
# - Solo la variable 'price' contiene valores faltantes (como es esperado en la base de prueba).
# - El resto de variables se encuentra completamente disponible.
# - El patrón geográfico y de distribución de 'bedrooms' y 'is_house' es coherente con el conjunto de entrenamiento.

# En resumen, los datasets han sido estandarizados y preparados exitosamente, eliminando ruido 
# y redundancia. Las transformaciones aplicadas garantizan consistencia entre conjuntos, y 
# las variables resultantes están listas para el siguiente paso: ingeniería de características 
# y entrenamiento del modelo.


###########################################
# 5. ESTADÍSTICAS DESCRIPTIVAS BÁSICAS   #
###########################################

cat("\n==============================================================\n")
cat("ESTADÍSTICAS DESCRIPTIVAS DE LOS DATOS LIMPIOS\n")
cat("==============================================================\n\n")

cat("--- DATASET DE ENTRENAMIENTO ---\n")
print(summary(train_clean))

cat("\n--- DATASET DE PRUEBA ---\n")
print(summary(test_clean))

# ==============================================================
# ESTADÍSTICAS DESCRIPTIVAS DE LOS DATOS LIMPIOS
# ==============================================================

# --- CONJUNTO DE ENTRENAMIENTO ---

# La base contiene 38,644 observaciones y 7 variables limpias.
# La variable objetivo `price` presenta una distribución asimétrica a la derecha, 
# con un valor mínimo de 300 millones COP y máximo de 1.650 millones. 
# La mediana es de 560 millones y la media de aproximadamente 654 millones,
# lo cual sugiere presencia de valores extremos que podrían justificar una transformación logarítmica.

# La variable `bedrooms` varía entre 0 y 11, con una mediana de 3 habitaciones,
# reflejando una mayoría de propiedades de tamaño mediano. La media es 3.1.

# Las coordenadas `lat` y `lon` se distribuyen consistentemente con la geografía de Bogotá:
# latitudes entre 4.577 y 4.765, longitudes entre -74.17 y -74.03.

# La variable `antiguedad` (medida en meses desde diciembre de 2021) va de 4 a 32 meses,
# con una mediana de 12 meses, lo que indica que la mayoría de propiedades llevan
# alrededor de un año publicadas.

# La variable `is_house` indica que solo el 24.5% de las propiedades son casas, 
# mientras que el 75.5% son apartamentos, lo que puede reflejar la oferta urbana predominante.

# --- CONJUNTO DE PRUEBA ---

# Contiene 10,286 observaciones con las mismas 7 variables.
# La variable `price` no tiene valores observados (NA en el 100%), como es esperado para predicción.

# `bedrooms` tiene una mediana de 2 habitaciones, ligeramente menor que en entrenamiento (3),
# y un valor máximo también de 11.

# Las coordenadas `lat` y `lon` se encuentran en rangos coherentes con el conjunto de entrenamiento:
# latitudes entre 4.589 y 4.733, y longitudes entre -74.10 y -74.03.

# `antiguedad` presenta valores similares (mínimo 4, máximo 32 meses),
# con mediana de 11 meses, lo cual sugiere que ambas muestras están temporalmente alineadas.

# La proporción de casas (`is_house = 1`) en el conjunto de prueba es muy baja (2.66%),
# lo que sugiere una subrepresentación de este tipo de propiedad respecto al entrenamiento.
# Esto podría requerir atención durante la validación para evitar sesgos de predicción.

###########################################
# 6. GUARDAR DATOS LIMPIOS               #
###########################################

cat("\n==============================================================\n")
cat("GUARDANDO DATOS LIMPIOS\n")
cat("==============================================================\n\n")

# Crear directorio si no existe
output_dir <- "stores/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Guardar en formato CSV
train_file <- file.path(output_dir, "train_cleaned.csv")
test_file <- file.path(output_dir, "test_cleaned.csv")

write.csv(train_clean, train_file, row.names = FALSE)
write.csv(test_clean, test_file, row.names = FALSE)

cat("Archivos guardados exitosamente:\n")
cat(paste0("- ", train_file, "\n"))
cat(paste0("- ", test_file, "\n"))

# Verificar que los archivos se guardaron correctamente
cat("\nVerificando archivos guardados:\n")
train_verification <- read.csv(train_file)
test_verification <- read.csv(test_file)

cat("Train verificado - dimensiones:", dim(train_verification), "\n")
cat("Test verificado - dimensiones:", dim(test_verification), "\n")

###########################################
# 7. RESUMEN FINAL                       #
###########################################

cat("\n\n==============================================================\n")
cat("RESUMEN FINAL DEL PROCESO DE LIMPIEZA\n")
cat("==============================================================\n\n")

cat("TRANSFORMACIONES APLICADAS:\n\n")

cat("1. VARIABLES ELIMINADAS:\n")
eliminadas <- c("city", "surface_total", "surface_covered", "rooms", 
                "bathrooms", "operation_type", "title", "description")
cat(paste0("   - ", eliminadas, collapse = "\n   - "), "\n\n")

cat("2. VARIABLES MANTENIDAS:\n")
cat("   - property_id: Identificador único\n")
cat("   - price: Variable dependiente (solo en train)\n") 
cat("   - bedrooms: Variable predictora\n")
cat("   - lat: Latitud (para validación cruzada espacial)\n")
cat("   - lon: Longitud (para validación cruzada espacial)\n\n")

cat("3. VARIABLES TRANSFORMADAS:\n")
cat("   - property_type → is_house: 1 = casa, 0 = apartamento\n\n")

cat("4. VARIABLES CREADAS:\n")
cat("   - antiguedad: Diferencia en meses desde diciembre 2021\n\n")

cat("5. ESTRUCTURA FINAL:\n")
cat("   Train:", ncol(train_clean), "variables,", nrow(train_clean), "observaciones\n")
cat("   Test:", ncol(test_clean), "variables,", nrow(test_clean), "observaciones\n\n")

cat("Proceso de limpieza completado exitosamente.\n")
cat("Los archivos CSV están listos para el siguiente paso del pipeline.\n")


################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################
