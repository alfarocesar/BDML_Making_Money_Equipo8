################################################################################
# TÍTULO: 12_super_learner.R                                                   #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo SuperLearner para predicción          #
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
  tidyverse,    # Manipulación de datos
  SuperLearner, # Para implementar SuperLearner
  caret,        # Para algunos algoritmos
  randomForest, # Para Random Forest
  glmnet,       # Para Elastic Net
  rpart,        # Para árboles de decisión
  nnls          # Para Non-Negative Least Squares
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
# 3. PREPARACIÓN DE DATOS PARA SL        #
###########################################

# Preparar variable dependiente y matriz de predictores para SuperLearner
Y <- train$price

# Crear matriz de predictores (excluyendo property_id y price)
predictores <- c("bedrooms", "antiguedad", "is_house", 
                 "distancia_parque", "distancia_universidad", "distancia_estacion_transporte", 
                 "distancia_zona_comercial", "nivel_premium", "nivel_completitud", "nivel_venta_inmediata")

# Verificar que todas las variables predictoras existen
predictores_disponibles <- predictores[predictores %in% names(train)]
cat("Variables predictoras disponibles:\n")
cat(paste(predictores_disponibles, collapse = ", "), "\n")

# Crear matrices para SuperLearner
X <- train[, predictores_disponibles, drop = FALSE]
newX <- test[, predictores_disponibles, drop = FALSE]

# Convertir a data.frame para asegurar compatibilidad
X <- data.frame(X)
newX <- data.frame(newX)

cat("Dimensiones de X:", dim(X), "\n")
cat("Dimensiones de newX:", dim(newX), "\n")

###########################################
# 4. CONFIGURACIÓN DE ALGORITMOS BASE     #
###########################################

# Verificar algoritmos disponibles en SuperLearner
cat("Algoritmos disponibles en SuperLearner:\n")
listWrappers()

# Definir biblioteca de algoritmos siguiendo los ejemplos de los cuadernos
sl.lib <- c(
  "SL.lm",          # Linear regression
  "SL.glmnet",      # Elastic Net
  "SL.randomForest", # Random Forest
  "SL.rpart"        # Classification and Regression Trees
)

cat("Algoritmos seleccionados para SuperLearner:\n")
cat(paste(sl.lib, collapse = ", "), "\n")

###########################################
# 5. CONFIGURACIÓN DE VALIDACIÓN CRUZADA #
###########################################

# Configurar validación cruzada de 5 folds (siguiendo patrón de los cuadernos)
folds <- 5
set.seed(123)
index <- split(sample(1:length(Y)), rep(1:folds, length = length(Y)))

cat("Configuración de validación cruzada:\n")
cat("Número de folds:", folds, "\n")
cat("Tamaño de cada fold:", sapply(index, length), "\n")

###########################################
# 6. ENTRENAMIENTO DE SUPERLEARNER       #
###########################################

cat("Iniciando entrenamiento de SuperLearner...\n")
cat("Esto puede tomar varios minutos...\n")

# Entrenar SuperLearner con validación cruzada
set.seed(123)
fitY <- SuperLearner(
  Y = Y,                          # Variable dependiente
  X = X,                          # Variables predictoras
  method = "method.NNLS",         # Método de combinación (Non-Negative Least Squares)
  SL.library = sl.lib,            # Biblioteca de algoritmos
  cvControl = list(V = folds, validRows = index)  # Control de validación cruzada
)

# Mostrar resultados del SuperLearner
cat("Resultados del SuperLearner:\n")
print(fitY)

# Mostrar coeficientes de combinación
cat("Coeficientes de combinación de algoritmos:\n")
print(fitY$coef)

# Mostrar riesgo (error) de cada algoritmo
cat("Riesgo de validación cruzada por algoritmo:\n")
print(fitY$cvRisk)

###########################################
# 7. PREDICCIONES                        #
###########################################

cat("Generando predicciones con SuperLearner...\n")

# Realizar predicciones usando solo el SuperLearner (ensemble completo)
predictions <- predict(fitY, newdata = newX, onlySL = TRUE)

# Extraer las predicciones
pred_values <- predictions$pred

# Verificar predicciones
cat("Estadísticas de las predicciones:\n")
cat("Min:", min(pred_values, na.rm = TRUE), "\n")
cat("Max:", max(pred_values, na.rm = TRUE), "\n")
cat("Mean:", mean(pred_values, na.rm = TRUE), "\n")
cat("Median:", median(pred_values, na.rm = TRUE), "\n")
cat("NAs:", sum(is.na(pred_values)), "\n")

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
  price = pred_values
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
# 9. ANÁLISIS DE RENDIMIENTO INDIVIDUAL  #
###########################################

# Obtener predicciones de algoritmos individuales para análisis
cat("Analizando rendimiento de algoritmos individuales...\n")

# Predicciones de cada algoritmo base
predictions_all <- predict(fitY, newdata = newX, onlySL = FALSE)

# Mostrar estructura de predicciones
cat("Estructura de predicciones por algoritmo:\n")
str(predictions_all$library.predict)

# Crear submissions adicionales para los mejores algoritmos individuales
if(ncol(predictions_all$library.predict) >= 2) {
  
  # Encontrar el algoritmo individual con menor riesgo de CV
  best_algo_idx <- which.min(fitY$cvRisk)
  best_algo_name <- names(fitY$cvRisk)[best_algo_idx]
  
  cat("Mejor algoritmo individual:", best_algo_name, "con CV Risk:", fitY$cvRisk[best_algo_idx], "\n")
  
  # Crear submission del mejor algoritmo individual
  submission_best_individual <- data.frame(
    property_id = test$property_id,
    price = predictions_all$library.predict[, best_algo_idx]
  )
  
  write_csv(submission_best_individual, "stores/submissions/submission_8.csv")
  cat("Submission del mejor algoritmo individual exportado: submission_8.csv\n")
}

###########################################
# 10. GUARDADO DE INFORMACIÓN DEL MODELO #
###########################################

# Crear directorio models si no existe
if (!dir.exists("stores/models")) {
  dir.create("stores/models", recursive = TRUE)
}

# Guardar información completa del SuperLearner
model_info <- list(
  superlearner_fit = fitY,
  algorithms_used = sl.lib,
  cv_risks = fitY$cvRisk,
  coefficients = fitY$coef,
  cv_folds = folds,
  variables_used = predictores_disponibles,
  date_created = Sys.time()
)

saveRDS(model_info, "stores/models/super_learner_model_info.rds")

cat("Información del modelo guardada en: stores/models/super_learner_model_info.rds\n")

###########################################
# 11. RESUMEN FINAL                      #
###########################################

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESUMEN FINAL - SUPERLEARNER\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Algoritmos incluidos en el ensemble:\n")
for(i in 1:length(sl.lib)) {
  cat("-", sl.lib[i], "| Peso:", round(fitY$coef[i], 4), 
      "| CV Risk:", round(fitY$cvRisk[i], 2), "\n")
}
cat("\nMejor algoritmo individual:", names(which.min(fitY$cvRisk)), "\n")
cat("SuperLearner CV Risk:", round(min(fitY$cvRisk), 2), "\n")
cat("\nSubmissions generados:\n")
cat("- submission_7.csv (SuperLearner ensemble)\n")
if(exists("submission_best_individual")) {
  cat("- submission_8.csv (Mejor algoritmo individual)\n")
}
cat("Observaciones procesadas:", nrow(test), "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################