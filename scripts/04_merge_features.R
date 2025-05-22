################################################################################
# TÍTULO: 04_merge_features.R                                                 #
# PROYECTO: Making Money with ML                                              #
# DESCRIPCIÓN: Combina datasets con variables limpias, espaciales y de texto  #
# FECHA: 21 de mayo de 2025                                                   #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../")

# Cargar librerías
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,
  readr
)

# Fijar semilla
set.seed(123)

################################################################################
# 1. CARGA DE LOS DATOS A COMBINAR                                            #
################################################################################

cat("Cargando datasets procesados...\n")

train_clean <- read_csv("stores/processed/train_cleaned.csv")
test_clean  <- read_csv("stores/processed/test_cleaned.csv")

train_spatial <- read_csv("stores/processed/train_vars_espacial.csv")
test_spatial  <- read_csv("stores/processed/test_vars_espacial.csv")

train_text <- read_csv("stores/processed/train_vars_texto.csv")
test_text  <- read_csv("stores/processed/test_vars_texto.csv")

################################################################################
# 2. COMBINACIÓN DE DATASETS                                                  #
################################################################################

cat("Combinando datasets de entrenamiento...\n")

train_final <- train_clean %>%
  left_join(train_spatial, by = "property_id") %>%
  left_join(train_text, by = "property_id")

cat("Combinando datasets de prueba...\n")

test_final <- test_clean %>%
  left_join(test_spatial, by = "property_id") %>%
  left_join(test_text, by = "property_id")

################################################################################
# 3. VERIFICACIÓN Y RESUMEN                                                   #
################################################################################

cat("Dimensiones finales:\n")
cat(" - Train:", dim(train_final), "\n")
cat(" - Test :", dim(test_final), "\n")

cat("Verificando identificadores únicos...\n")
cat(" - IDs únicos en train:", n_distinct(train_final$property_id), "\n")
cat(" - IDs únicos en test :", n_distinct(test_final$property_id), "\n")

################################################################################
# 4. GUARDAR RESULTADOS                                                       #
################################################################################

output_dir <- "stores/processed"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

write_csv(train_final, file.path(output_dir, "train_merged.csv"))
write_csv(test_final,  file.path(output_dir, "test_merged.csv"))

cat("Archivos combinados guardados exitosamente en:\n")
cat(" - stores/processed/train_merged.csv\n")
cat(" - stores/processed/test_merged.csv\n")

################################################################################
# 5. DICCIONARIO DE VARIABLES DEL DATASET COMBINADO                           #
################################################################################

# VARIABLES PROVENIENTES DE LA LIMPIEZA INICIAL (01_data_cleaning.R)
# -----------------------------------------------------------------------------
# property_id              | Identificador único de la propiedad (categórica nominal)
# price                    | Precio de oferta (numérica continua, solo en train)
# bedrooms                 | Número de habitaciones (numérica discreta)
# antiguedad              | Meses desde publicación hasta diciembre de 2021 (numérica discreta)
# is_house                 | Indicador binario: 1 si es casa, 0 si es apartamento (binaria)

# VARIABLES ESPACIALES (02_spatial_vars.R)
# -----------------------------------------------------------------------------
# distancia_parque               | Distancia en metros al parque más cercano (numérica continua)
# distancia_universidad         | Distancia en metros a la universidad más cercana (numérica continua)
# distancia_estacion_transporte | Distancia en metros a estación de Transmilenio más cercana (numérica continua)
# distancia_zona_comercial      | Distancia en metros a la zona comercial más cercana (numérica continua)

# VARIABLES DE TEXTO (03_text_vars.R)
# -----------------------------------------------------------------------------
# nivel_premium           | Nivel de lujo percibido según términos en la descripción (ordinal discreta: 0–5)
# nivel_completitud       | Proporción de espacios funcionales mencionados (continua: 0–1)
# nivel_venta_inmediata   | Indicador de urgencia en venta según términos clave (ordinal discreta: 0–3)

# NOTAS METODOLÓGICAS
# -----------------------------------------------------------------------------
# - Las variables espaciales se expresan en metros y provienen de OpenStreetMap.
# - Las variables de texto fueron derivadas del campo `description` original mediante procesamiento léxico.
# - La variable `price` solo está disponible en el conjunto de entrenamiento y es la variable objetivo.
# - El identificador `property_id` debe conservarse sin transformaciones para garantizar consistencia en el pipeline.

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################
