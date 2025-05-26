################################################################################
# TÍTULO: 06_linear_regression.R                                               #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo Linear Regression para predicción     #
# FECHA: 22 de mayo de 2025                                                   #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Subir un nivel directorio para acceder a la estructura principal del proyecto
setwd("../")

# Cargar librerías usando pacman
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,      # Manipulación de datos
  caret,          # Para entrenamiento de modelos
  sf,             # Para datos espaciales
  spatialsample,  # Para validación cruzada espacial
  stargazer       # Para mostrar resultados
)

# Fijar semilla para reproducibilidad
set.seed(123)

###########################################
# 1. CARGA Y PREPARACIÓN DE DATOS        #
###########################################

# Cargar datasets procesados
train <- read_csv("stores/processed/train_merged.csv")
test <- read_csv("stores/processed/test_merged.csv")

cat("Dimensiones train:", dim(train), "\n")
cat("Dimensiones test:", dim(test), "\n")

# Verificar que existe la variable precio
if(!"price" %in% names(train)) {
  stop("La variable 'price' no se encontró en el dataset de entrenamiento")
}

# Verificar variables disponibles
variables_esperadas <- c(
  "property_id", "price", "bedrooms", "antiguedad", "is_house",
  "distancia_parque", "distancia_universidad", "distancia_estacion_transporte", 
  "distancia_zona_comercial", "nivel_premium", "nivel_completitud", "nivel_venta_inmediata",
  "lat", "lon"
)

cat("\nVerificando variables esperadas:\n")
for(var in variables_esperadas) {
  if(var %in% names(train)) {
    cat("✓", var, "\n")
  } else {
    cat("✗", var, "(faltante)\n")
  }
}

# Mostrar estadísticas básicas de la variable objetivo
cat("\nEstadísticas de price:\n")
summary(train$price)

###########################################
# 2. IMPUTACIÓN DE VALORES FALTANTES     #
###########################################

# Variables numéricas continuas - imputar con mediana
variables_numericas_continuas <- c("bedrooms", "antiguedad", "distancia_parque", 
                                   "distancia_universidad", "distancia_estacion_transporte", 
                                   "distancia_zona_comercial", "nivel_completitud")

for(var in variables_numericas_continuas) {
  if(var %in% names(train)) {
    if(sum(is.na(train[[var]])) > 0) {
      mediana <- median(train[[var]], na.rm = TRUE)
      train[[var]] <- ifelse(is.na(train[[var]]), mediana, train[[var]])
      test[[var]] <- ifelse(is.na(test[[var]]), mediana, test[[var]])
      cat("Imputada", var, "con mediana:", mediana, "\n")
    }
  }
}

# Variables ordinales discretas - imputar con moda
variables_ordinales <- c("nivel_premium", "nivel_venta_inmediata")

for(var in variables_ordinales) {
  if(var %in% names(train)) {
    if(sum(is.na(train[[var]])) > 0) {
      moda <- as.numeric(names(sort(table(train[[var]]), decreasing = TRUE))[1])
      train[[var]] <- ifelse(is.na(train[[var]]), moda, train[[var]])
      test[[var]] <- ifelse(is.na(test[[var]]), moda, test[[var]])
      cat("Imputada", var, "con moda:", moda, "\n")
    }
  }
}

# Variable binaria - imputar con moda
if("is_house" %in% names(train)) {
  if(sum(is.na(train$is_house)) > 0) {
    moda <- as.numeric(names(sort(table(train$is_house), decreasing = TRUE))[1])
    train$is_house <- ifelse(is.na(train$is_house), moda, train$is_house)
    test$is_house <- ifelse(is.na(test$is_house), moda, test$is_house)
    cat("Imputada is_house con moda:", moda, "\n")
  }
}

###########################################
# 3. ESPECIFICACIÓN DEL MODELO           #
###########################################

# Definir fórmula del modelo usando todas las variables predictoras disponibles
# (excluyendo property_id, price, lat y lon)
model_form <- price ~ bedrooms + antiguedad + is_house + 
  distancia_parque + distancia_universidad + distancia_estacion_transporte + 
  distancia_zona_comercial + nivel_premium + nivel_completitud + nivel_venta_inmediata

cat("\nFórmula del modelo:\n")
print(model_form)

###########################################
# 4. VALIDACIÓN CRUZADA ESPACIAL         #
###########################################

# Convertir datos a formato sf siguiendo el patrón de los cuadernos
train_sf <- st_as_sf(
  train,
  coords = c("lon", "lat"),  # "coords" is in x/y order -- so longitude goes first!
  crs = 4326  # Set our coordinate reference system to EPSG:4326
)

cat("Datos convertidos a formato espacial sf\n")

# Crear bloques espaciales siguiendo el patrón de los cuadernos
set.seed(123)
block_folds <- spatial_block_cv(train_sf, v = 5)

cat("Bloques espaciales creados:\n")
print(block_folds)

# Extraer índices de los folds espaciales para usar con caret
folds <- list()
for(i in 1:5) {
  folds[[i]] <- block_folds$splits[[i]]$in_id
}

# Configurar validación cruzada espacial
ctrl <- trainControl(
  method = "cv",
  index = folds  # Usar índices espaciales
)

###########################################
# 5. ENTRENAMIENTO DEL MODELO            #
###########################################

cat("Iniciando entrenamiento del modelo Linear Regression con validación cruzada espacial...\n")

# Entrenar modelo usando caret con validación espacial
set.seed(123)
modelo_lm <- train(
  model_form,           # Fórmula del modelo
  data = train,         # Datos de entrenamiento (dataset original, no sf)
  method = 'lm',        # Linear regression
  trControl = ctrl      # Configuración de CV espacial
)

# Mostrar resultados del modelo
cat("Resultados del modelo:\n")
print(modelo_lm)

# Mostrar métricas de validación cruzada
cat("Métricas de validación cruzada espacial:\n")
print(modelo_lm$results)

# Mostrar coeficientes del modelo final
cat("Coeficientes del modelo:\n")
print(summary(modelo_lm$finalModel))

###########################################
# 6. PREDICCIONES                        #
###########################################

# Realizar predicciones en el conjunto de test
cat("Generando predicciones...\n")
predictions <- predict(modelo_lm, newdata = test)

# Verificar predicciones
cat("Estadísticas de las predicciones:\n")
cat("Min:", min(predictions, na.rm = TRUE), "\n")
cat("Max:", max(predictions, na.rm = TRUE), "\n")
cat("Mean:", mean(predictions, na.rm = TRUE), "\n")
cat("Median:", median(predictions, na.rm = TRUE), "\n")
cat("NAs:", sum(is.na(predictions)), "\n")

###########################################
# 7. EXPORTACIÓN DE RESULTADOS           #
###########################################

# Crear submission
submission <- data.frame(
  property_id = test$property_id,
  price = predictions
)

# Verificar submission
cat("Dimensiones del submission:", dim(submission), "\n")
cat("NAs en submission:", sum(is.na(submission$price)), "\n")

# Mostrar preview del submission
cat("Preview del submission:\n")
print(head(submission))

# Exportar submission
write_csv(submission, "stores/submissions/submission_LR1.csv")

cat("Submission exportado exitosamente: stores/submissions/submission_1.csv\n")

###########################################
# 8. ANÁLISIS DE IMPORTANCIA DE VARIABLES#
###########################################

# Calcular importancia de variables
importance <- varImp(modelo_lm, scale = FALSE)
cat("Importancia de variables:\n")
print(importance)

###########################################
# 9. GUARDADO DE INFORMACIÓN DEL MODELO  #
###########################################

# Crear directorio models si no existe
if (!dir.exists("stores/models")) {
  dir.create("stores/models", recursive = TRUE)
}

# Guardar información del modelo
model_info <- list(
  formula = model_form,
  cv_results = modelo_lm$results,
  final_model = modelo_lm,
  variable_importance = importance,
  model_summary = summary(modelo_lm$finalModel),
  spatial_folds = block_folds,
  date_created = Sys.time()
)

saveRDS(model_info, "stores/models/linear_regression_model_info.rds")

cat("Información del modelo guardada en: stores/models/linear_regression_model_info.rds\n")

###########################################
# 10. RESUMEN FINAL                      #
###########################################

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESUMEN FINAL - LINEAR REGRESSION CON VALIDACIÓN ESPACIAL\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Variables predictoras:\n")
cat("- Estructurales: bedrooms, antiguedad, is_house\n")
cat("- Espaciales: distancia_parque, distancia_universidad,\n")
cat("             distancia_estacion_transporte, distancia_zona_comercial\n")
cat("- De texto: nivel_premium, nivel_completitud, nivel_venta_inmediata\n")
cat("\nValidación cruzada: Espacial (5 folds con bloques geográficos)\n")
cat("Beneficio: Evita optimismo por autocorrelación espacial\n")
cat("\nMétricas de validación cruzada espacial:\n")
cat("RMSE:", round(modelo_lm$results$RMSE, 2), "\n")
cat("R-squared:", round(modelo_lm$results$Rsquared, 4), "\n")
cat("MAE:", round(modelo_lm$results$MAE, 2), "\n")
cat("\nSubmission generado: submission_1.csv\n")
cat("Observaciones procesadas:", nrow(test), "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################