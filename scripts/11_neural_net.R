################################################################################
# TÍTULO: 11_neural_network.R                                                 #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo Neural Network para predicción        #
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
  tidyverse,  # Manipulación de datos
  caret,      # Para entrenamiento de modelos
  nnet        # Para neural networks
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
  "distancia_zona_comercial", "nivel_premium", "nivel_completitud", "nivel_venta_inmediata"
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
# 3. PREPROCESAMIENTO PARA NEURAL NETWORK#
###########################################

# Para neural networks necesitamos normalizar todas las variables
# Crear objeto de preprocesamiento
preprocess_params <- preProcess(
  train[, variables_numericas_continuas], 
  method = c("center", "scale")
)

# Aplicar normalización a datos de entrenamiento y test
train_normalized <- predict(preprocess_params, train)
test_normalized <- predict(preprocess_params, test)

# También normalizar la variable objetivo para mejorar convergencia
price_params <- preProcess(train["price"], method = c("center", "scale"))
train_normalized$price <- predict(price_params, train["price"])$price

cat("Variables normalizadas exitosamente\n")

###########################################
# 4. ESPECIFICACIÓN DEL MODELO           #
###########################################

# Definir fórmula del modelo
model_form <- price ~ bedrooms + antiguedad + is_house + 
  distancia_parque + distancia_universidad + distancia_estacion_transporte + 
  distancia_zona_comercial + nivel_premium + nivel_completitud + nivel_venta_inmediata

cat("\nFórmula del modelo:\n")
print(model_form)

###########################################
# 5. CONFIGURACIÓN DE VALIDACIÓN CRUZADA #
###########################################

# Configurar validación cruzada con grid search para hiperparámetros
ctrl <- trainControl(
  method = "cv",        # Cross-validation
  number = 5,           # 5 folds
  verboseIter = TRUE    # Mostrar progreso
)

# Definir grilla de hiperparámetros para neural network
# size: número de unidades en la capa oculta
# decay: parámetro de regularización
nnet_grid <- expand.grid(
  size = c(5, 10, 15),     # Número de neuronas en capa oculta
  decay = c(0, 0.01, 0.1)  # Parámetro de decay
)

cat("Grilla de hiperparámetros:\n")
print(nnet_grid)

###########################################
# 6. ENTRENAMIENTO DEL MODELO            #
###########################################

cat("Iniciando entrenamiento del modelo Neural Network...\n")

# Entrenar modelo usando caret con grid search
set.seed(123)
modelo_nnet <- train(
  model_form,                    # Fórmula del modelo
  data = train_normalized,       # Datos normalizados
  method = 'nnet',              # Neural network
  trControl = ctrl,             # Configuración de CV
  tuneGrid = nnet_grid,         # Grilla de hiperparámetros
  linout = TRUE,                # Para regresión (salida lineal)
  trace = FALSE,                # No mostrar detalles de entrenamiento
  maxit = 1000                  # Máximo número de iteraciones
)

# Mostrar resultados del modelo
cat("Resultados del modelo:\n")
print(modelo_nnet)

# Mostrar mejores hiperparámetros
cat("Mejores hiperparámetros:\n")
print(modelo_nnet$bestTune)

# Mostrar métricas de validación cruzada
cat("Métricas de validación cruzada:\n")
print(modelo_nnet$results)

###########################################
# 7. PREDICCIONES                        #
###########################################

# Realizar predicciones en el conjunto de test normalizado
cat("Generando predicciones...\n")
predictions_normalized <- predict(modelo_nnet, newdata = test_normalized)

# Desnormalizar las predicciones para obtener valores en escala original
# Usando los parámetros de normalización de la variable objetivo
predictions_original <- predictions_normalized * attr(price_params$std, "scaled:scale") + 
  attr(price_params$mean, "scaled:center")

# Verificar predicciones
cat("Estadísticas de las predicciones:\n")
cat("Min:", min(predictions_original, na.rm = TRUE), "\n")
cat("Max:", max(predictions_original, na.rm = TRUE), "\n")
cat("Mean:", mean(predictions_original, na.rm = TRUE), "\n")
cat("Median:", median(predictions_original, na.rm = TRUE), "\n")
cat("NAs:", sum(is.na(predictions_original)), "\n")

###########################################
# 8. EXPORTACIÓN DE RESULTADOS           #
###########################################

# Crear directorio submissions si no existe
if (!dir.exists("stores/submissions")) {
  dir.create("stores/submissions", recursive = TRUE)
}

# Crear submission
submission <- data.frame(
  property_id = test$property_id,
  price = predictions_original
)

# Verificar submission
cat("Dimensiones del submission:", dim(submission), "\n")
cat("NAs en submission:", sum(is.na(submission$price)), "\n")

# Mostrar preview del submission
cat("Preview del submission:\n")
print(head(submission))

# Exportar submission
write_csv(submission, "stores/submissions/submission_7.csv")

cat("Submission exportado exitosamente: stores/submissions/submission_7.csv\n")

###########################################
# 9. ANÁLISIS DE IMPORTANCIA DE VARIABLES#
###########################################

# Calcular importancia de variables
importance <- varImp(modelo_nnet, scale = FALSE)
cat("Importancia de variables:\n")
print(importance)

###########################################
# 10. GUARDADO DE INFORMACIÓN DEL MODELO #
###########################################

# Crear directorio models si no existe
if (!dir.exists("stores/models")) {
  dir.create("stores/models", recursive = TRUE)
}

# Guardar información del modelo
model_info <- list(
  formula = model_form,
  best_tune = modelo_nnet$bestTune,
  cv_results = modelo_nnet$results,
  final_model = modelo_nnet,
  variable_importance = importance,
  preprocess_params = preprocess_params,
  price_params = price_params,
  date_created = Sys.time()
)

saveRDS(model_info, "stores/models/neural_network_model_info.rds")

cat("Información del modelo guardada en: stores/models/neural_network_model_info.rds\n")

###########################################
# 11. RESUMEN FINAL                      #
###########################################

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESUMEN FINAL - NEURAL NETWORK\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Variables predictoras:\n")
cat("- Estructurales: bedrooms, antiguedad, is_house\n")
cat("- Espaciales: distancia_parque, distancia_universidad,\n")
cat("             distancia_estacion_transporte, distancia_zona_comercial\n")
cat("- De texto: nivel_premium, nivel_completitud, nivel_venta_inmediata\n")
cat("\nHiperparámetros óptimos:\n")
cat("Size (neuronas en capa oculta):", modelo_nnet$bestTune$size, "\n")
cat("Decay (regularización):", modelo_nnet$bestTune$decay, "\n")
cat("\nMétricas de validación cruzada:\n")
best_results <- modelo_nnet$results[
  modelo_nnet$results$size == modelo_nnet$bestTune$size & 
    modelo_nnet$results$decay == modelo_nnet$bestTune$decay, ]
cat("RMSE:", round(best_results$RMSE, 2), "\n")
cat("R-squared:", round(best_results$Rsquared, 4), "\n")
cat("MAE:", round(best_results$MAE, 2), "\n")
cat("\nSubmission generado: submission_7.csv\n")
cat("Observaciones procesadas:", nrow(test), "\n")
cat("Preprocesamiento: Variables normalizadas (center + scale)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################