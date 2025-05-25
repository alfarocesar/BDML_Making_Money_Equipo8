################################################################################
# TÍTULO: 08_random_forest.R                                                   #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo Random Forest para predicción         #
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
  ranger      # Implementación eficiente de Random Forest
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
# 5. CONFIGURACIÓN DE HIPERPARÁMETROS    #
###########################################

# Definir grilla de hiperparámetros para Random Forest
# mtry: número de variables consideradas en cada división
# min.node.size: tamaño mínimo de nodos terminales
# splitrule: criterio de división (para regresión típicamente "variance")

# Número total de predictores (sin contar property_id y price)
n_predictors <- ncol(train) - 2

# Definir grilla de mtry basada en reglas heurísticas
mtry_values <- c(
  floor(n_predictors/3),     # Valor conservador
  floor(sqrt(n_predictors)), # Valor por defecto para clasificación (adaptado)
  floor(n_predictors/2)      # Valor más liberal
)

# Asegurar que los valores estén en el rango válido
mtry_values <- pmax(1, pmin(n_predictors, mtry_values))
mtry_values <- unique(mtry_values)

# Grilla de parámetros
tune_grid <- expand.grid(
  mtry = mtry_values,
  splitrule = "variance",        # Para regresión
  min.node.size = c(5, 10, 20)   # Diferentes tamaños mínimos de nodo
)

cat("Número de predictores:", n_predictors, "\n")
cat("Valores de mtry a probar:", mtry_values, "\n")
cat("Combinaciones de hiperparámetros:", nrow(tune_grid), "\n")
print(tune_grid)

###########################################
# 6. ENTRENAMIENTO DEL MODELO            #
###########################################

cat("Iniciando entrenamiento del modelo Random Forest...\n")

# Entrenar modelo usando caret con ranger
set.seed(123)
modelo_rf <- train(
  model_form,           # Fórmula del modelo
  data = train,         # Datos de entrenamiento
  method = 'ranger',    # Random Forest usando ranger
  trControl = ctrl,     # Configuración de CV
  tuneGrid = tune_grid, # Grilla de hiperparámetros
  num.trees = 500,      # Número de árboles
  importance = 'permutation' # Calcular importancia de variables
)

# Mostrar resultados del modelo
cat("Resultados del modelo:\n")
print(modelo_rf)

# Mostrar mejores hiperparámetros
cat("Mejores hiperparámetros:\n")
print(modelo_rf$bestTune)

# Mostrar métricas de validación cruzada
cat("Métricas de validación cruzada:\n")
print(modelo_rf$results)

###########################################
# 7. PREDICCIONES                        #
###########################################

# Realizar predicciones en el conjunto de test
cat("Generando predicciones...\n")
predictions <- predict(modelo_rf, newdata = test)

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
importance <- varImp(modelo_rf, scale = FALSE)
cat("Importancia de variables:\n")
print(importance)

# Crear gráfico de importancia si es posible
if("ggplot2" %in% loadedNamespaces()) {
  importance_plot <- plot(importance, main = "Importancia de Variables - Random Forest")
  print(importance_plot)
}

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
  best_tune = modelo_rf$bestTune,
  cv_results = modelo_rf$results,
  final_model = modelo_rf,
  variable_importance = importance,
  hyperparameter_grid = tune_grid,
  model_type = "Random Forest",
  num_trees = 500,
  date_created = Sys.time()
)

saveRDS(model_info, "stores/models/random_forest_model_info.rds")

cat("Información del modelo guardada en: stores/models/random_forest_model_info.rds\n")

###########################################
# 11. RESUMEN FINAL                      #
###########################################

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESUMEN FINAL - RANDOM FOREST\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Variables predictoras:\n")
cat("- Estructurales: bedrooms, antiguedad, is_house\n")
cat("- Espaciales: distancia_parque, distancia_universidad,\n")
cat("             distancia_estacion_transporte, distancia_zona_comercial\n")
cat("- De texto: nivel_premium, nivel_completitud, nivel_venta_inmediata\n")
cat("\nHiperparámetros óptimos:\n")
cat("mtry:", modelo_rf$bestTune$mtry, "\n")
cat("min.node.size:", modelo_rf$bestTune$min.node.size, "\n")
cat("splitrule:", modelo_rf$bestTune$splitrule, "\n")
cat("num.trees: 500\n")
cat("\nMétricas de validación cruzada:\n")
best_idx <- which(modelo_rf$results$mtry == modelo_rf$bestTune$mtry & 
                    modelo_rf$results$min.node.size == modelo_rf$bestTune$min.node.size)
cat("RMSE:", round(modelo_rf$results$RMSE[best_idx], 2), "\n")
cat("R-squared:", round(modelo_rf$results$Rsquared[best_idx], 4), "\n")
cat("MAE:", round(modelo_rf$results$MAE[best_idx], 2), "\n")
cat("\nSubmission generado: submission_7.csv\n")
cat("Observaciones procesadas:", nrow(test), "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################