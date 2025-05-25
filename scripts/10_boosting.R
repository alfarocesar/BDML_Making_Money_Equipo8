################################################################################
# TÍTULO: 08_boosting.R                                                        #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo Boosting para predicción de precios   #
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
  gbm         # Gradient Boosting Machine
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
# 3. ESPECIFICACIÓN DEL MODELO           #
###########################################

# Definir fórmula del modelo usando todas las variables predictoras disponibles
# (excluyendo property_id y price)
model_form <- price ~ bedrooms + antiguedad + is_house + 
  distancia_parque + distancia_universidad + distancia_estacion_transporte + 
  distancia_zona_comercial + nivel_premium + nivel_completitud + nivel_venta_inmediata

cat("\nFórmula del modelo:\n")
print(model_form)

###########################################
# 4. CONFIGURACIÓN DE VALIDACIÓN CRUZADA #
###########################################

# Configurar validación cruzada siguiendo el patrón de los cuadernos
ctrl <- trainControl(
  method = "cv",        # Cross-validation
  number = 5,           # 5 folds
  verboseIter = TRUE    # Mostrar progreso
)

###########################################
# 5. ESPECIFICACIÓN DE HIPERPARÁMETROS   #
###########################################

# Definir grilla de hiperparámetros para GBM siguiendo los ejemplos de los cuadernos
# Parámetros basados en CuadernoModulo05_Boosting.pdf
grid_gbm <- expand.grid(
  n.trees = c(200, 300, 500),           # Número de árboles (M)
  interaction.depth = c(4, 6),          # Profundidad máxima de árboles (J)
  shrinkage = c(0.001, 0.01),          # Tasa de aprendizaje (λ)
  n.minobsinnode = c(10, 30)           # Tamaño mínimo de nodo terminal
)

cat("Número total de combinaciones de hiperparámetros:", nrow(grid_gbm), "\n")
cat("Grilla de hiperparámetros:\n")
print(grid_gbm)

###########################################
# 6. ENTRENAMIENTO DEL MODELO            #
###########################################

cat("Iniciando entrenamiento del modelo Boosting (GBM)...\n")

# Entrenar modelo usando caret (siguiendo el patrón de los cuadernos)
set.seed(123)
modelo_gbm <- train(
  model_form,           # Fórmula del modelo
  data = train,         # Datos de entrenamiento
  method = 'gbm',       # Gradient Boosting Machine
  trControl = ctrl,     # Configuración de CV
  tuneGrid = grid_gbm,  # Grilla de hiperparámetros
  verbose = FALSE       # No mostrar detalles durante entrenamiento
)

# Mostrar resultados del modelo
cat("Resultados del modelo:\n")
print(modelo_gbm)

# Mostrar mejores hiperparámetros
cat("Mejores hiperparámetros:\n")
print(modelo_gbm$bestTune)

# Mostrar métricas de validación cruzada del mejor modelo
cat("Métricas de validación cruzada del mejor modelo:\n")
best_results <- modelo_gbm$results[
  which(modelo_gbm$results$n.trees == modelo_gbm$bestTune$n.trees &
          modelo_gbm$results$interaction.depth == modelo_gbm$bestTune$interaction.depth &
          modelo_gbm$results$shrinkage == modelo_gbm$bestTune$shrinkage &
          modelo_gbm$results$n.minobsinnode == modelo_gbm$bestTune$n.minobsinnode), 
]
print(best_results)

###########################################
# 7. PREDICCIONES                        #
###########################################

# Realizar predicciones en el conjunto de test
cat("Generando predicciones...\n")
predictions <- predict(modelo_gbm, newdata = test)

# Verificar predicciones
cat("Estadísticas de las predicciones:\n")
cat("Min:", min(predictions, na.rm = TRUE), "\n")
cat("Max:", max(predictions, na.rm = TRUE), "\n")
cat("Mean:", mean(predictions, na.rm = TRUE), "\n")
cat("Median:", median(predictions, na.rm = TRUE), "\n")
cat("NAs:", sum(is.na(predictions)), "\n")

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
  price = predictions
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
importance <- varImp(modelo_gbm, scale = FALSE)
cat("Importancia de variables:\n")
print(importance)

# Mostrar las variables más importantes
cat("Top 5 variables más importantes:\n")
top_vars <- importance$importance
top_vars$Variable <- rownames(top_vars)
top_vars <- top_vars[order(top_vars$Overall, decreasing = TRUE), ]
print(head(top_vars, 5))

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
  best_tune = modelo_gbm$bestTune,
  cv_results = modelo_gbm$results,
  final_model = modelo_gbm,
  variable_importance = importance,
  hyperparameter_grid = grid_gbm,
  date_created = Sys.time()
)

saveRDS(model_info, "stores/models/boosting_model_info.rds")

cat("Información del modelo guardada en: stores/models/boosting_model_info.rds\n")

###########################################
# 11. RESUMEN FINAL                      #
###########################################

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESUMEN FINAL - BOOSTING (GBM)\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Variables predictoras:\n")
cat("- Estructurales: bedrooms, antiguedad, is_house\n")
cat("- Espaciales: distancia_parque, distancia_universidad,\n")
cat("             distancia_estacion_transporte, distancia_zona_comercial\n")
cat("- De texto: nivel_premium, nivel_completitud, nivel_venta_inmediata\n")
cat("\nMejores hiperparámetros encontrados:\n")
cat("- Número de árboles (n.trees):", modelo_gbm$bestTune$n.trees, "\n")
cat("- Profundidad máxima (interaction.depth):", modelo_gbm$bestTune$interaction.depth, "\n")
cat("- Tasa de aprendizaje (shrinkage):", modelo_gbm$bestTune$shrinkage, "\n")
cat("- Mínimo en nodo (n.minobsinnode):", modelo_gbm$bestTune$n.minobsinnode, "\n")
cat("\nMétricas de validación cruzada:\n")
cat("RMSE:", round(best_results$RMSE, 2), "\n")
cat("R-squared:", round(best_results$Rsquared, 4), "\n")
cat("MAE:", round(best_results$MAE, 2), "\n")
cat("\nSubmission generado: submission_7.csv\n")
cat("Observaciones procesadas:", nrow(test), "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################