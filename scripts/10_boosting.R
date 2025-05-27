################################################################################
# TÍTULO: 10_boosting.R                                                        #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo Boosting para predicción de precios    #
# FECHA: 22 de mayo de 2025                                                    #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Subir un nivel directorio para acceder a la estructura principal del proyecto
setwd("../")

# Cargar librerías usando pacman
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,     # Manipulación de datos
  caret,         # Para entrenamiento de modelos
  gbm,           # Gradient Boosting Machine
  spatialsample, # Validación cruzada espacial
  sf,             # Datos espaciales
  xgboost,  # Para el modelo XGBoost
  Matrix,    # Para matrices dispersas (necesario para one-hot encoding en xgb)
)

# Fijar semilla para reproducibilidad
set.seed(123)

###########################################
# 1. CARGA Y PREPARACIÓN DE DATOS         #
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

###########################################################
# 2. IMPUTACIÓN DE VALORES FALTANTES y PREPROCESAMIENTO   #
###########################################################

# Transformación de la variable objetivo (price)
train_df <- train %>%
  mutate(price_log = log1p(price))

# Identificar características numéricas y categóricas
numerical_features <- c(
  "bedrooms","antiguedad","lat", "lon","distancia_parque", 
  "distancia_universidad", "distancia_estacion_transporte",
  "distancia_zona_comercial"
)

# Definimos todas categóricas
# Luego filtraremos las que tienen un solo nivel
possible_categorical_features <- c()


# Añadir 'description_length' a las características numéricas
numerical_features <- c(numerical_features)


# Identificación de Chapinero usando lat/lon
chapinero_lat_min <- 4.63
chapinero_lat_max <- 4.70
chapinero_lon_min <- -74.07
chapinero_lon_max <- -74.03

train_df <- train_df %>%
  mutate(is_chapinero = ifelse(
    !is.na(lat) & !is.na(lon) &
      lat >= chapinero_lat_min & lat <= chapinero_lat_max &
      lon >= chapinero_lon_min & lon <= chapinero_lon_max,
    1, 0
  ))

test_df <- test %>%
  mutate(is_chapinero = ifelse(
    !is.na(lat) & !is.na(lon) &
      lat >= chapinero_lat_min & lat <= chapinero_lat_max &
      lon >= chapinero_lon_min & lon <= chapinero_lon_max,
    1, 0
  ))

numerical_features <- c(numerical_features, 'is_chapinero')

required_vars <- c(numerical_features, possible_categorical_features, "property_id")

missing_vars_train <- setdiff(required_vars, names(train_df))
missing_vars_test <- setdiff(required_vars, names(test_df))

print(missing_vars_train)
print(missing_vars_test)



# Unir train y test para asegurar consistencia en el preprocesamiento
# Nos aseguramos de incluir 'description' y 'title' temporalmente si queremos usarlas para otras features
# o si nchar(description) necesita que estén presentes antes del select.
all_data <- bind_rows(
  train_df %>% select(all_of(c(numerical_features, possible_categorical_features, "property_id"))),
  test_df %>% select(all_of(c(numerical_features, possible_categorical_features, "property_id")))
)

# Convertir columnas categóricas a tipo factor en all_data y verificar niveles
categorical_features <- c()
for (col in possible_categorical_features) {
  if(col %in% colnames(all_data)){
    all_data[[col]] <- as.factor(all_data[[col]])
    if (nlevels(all_data[[col]]) > 1) {
      categorical_features <- c(categorical_features, col)
    } else {
      warning(paste("La columna categórica '", col, "' tiene 1 o menos niveles y será ignorada para One-Hot Encoding.", sep=""))
    }
  } else {
    warning(paste("La columna categórica '", col, "' no se encontró en all_data y será ignorada.", sep=""))
  }
}

# 1. Preprocesamiento de características numéricas: imputación y escalado
preproc_num_model <- preProcess(
  all_data %>% select(all_of(numerical_features)),
  method = c("medianImpute", "zv", "center", "scale")
)
processed_num_data <- predict(preproc_num_model, all_data %>% select(all_of(numerical_features)))


# 2. Preprocesamiento de características categóricas: One-Hot Encoding
if (length(categorical_features) > 0) {
  formula_cat <- as.formula(paste("~", paste(categorical_features, collapse = " + ")))
  dummy_model <- dummyVars(formula_cat, data = all_data, fullRank = TRUE)
  processed_cat_data <- predict(dummy_model, newdata = all_data)
  processed_cat_data_df <- as.data.frame(as.matrix(processed_cat_data))
} else {
  processed_cat_data_df <- data.frame()
  warning("No hay características categóricas con 2 o más niveles para One-Hot Encoding.")
}

# Combinar las características numéricas y categóricas procesadas
if (ncol(processed_cat_data_df) > 0) {
  processed_all_data_final <- bind_cols(
    all_data %>% select(property_id),
    processed_num_data,
    processed_cat_data_df
  )
} else {
  processed_all_data_final <- bind_cols(
    all_data %>% select(property_id),
    processed_num_data
  )
}


# Separar de nuevo en conjuntos de entrenamiento y prueba

# Asegurarse de que X_train_processed y X_test_processed tengan los mismos nombres de columnas
# y en el mismo orden.
X_train_processed <- processed_all_data_final[1:nrow(train_df), ] %>%
  select(-property_id)

X_test_processed <- processed_all_data_final[(nrow(train_df) + 1):nrow(processed_all_data_final), ] %>%
  select(-property_id)

# Alineación de columnas: Asegura que ambos conjuntos tengan las mismas columnas en el mismo orden
# Esto es CRÍTICO para que predict funcione correctamente con el modelo entrenado.
train_cols <- colnames(X_train_processed)
test_cols <- colnames(X_test_processed)

# Columnas presentes en ambos
common_cols <- intersect(train_cols, test_cols)

X_train_processed <- X_train_processed %>% select(all_of(common_cols))
X_test_processed <- X_test_processed %>% select(all_of(common_cols))

# Añadir columnas faltantes a test_df (si las hay, rellenando con 0s)
missing_in_test <- setdiff(train_cols, test_cols)
if(length(missing_in_test) > 0){
  for(col_name in missing_in_test){
    X_test_processed[[col_name]] <- 0 # O un valor predeterminado apropiado, 0 para one-hot encodings
  }
}

# Añadir columnas faltantes a train_df (si las hay, rellenando con 0s)
missing_in_train <- setdiff(test_cols, train_cols)
if(length(missing_in_train) > 0){
  for(col_name in missing_in_train){
    X_train_processed[[col_name]] <- 0 # O un valor predeterminado apropiado
  }
}

# Asegurarse de que el orden de las columnas sea idéntico
X_test_processed <- X_test_processed %>% select(all_of(colnames(X_train_processed)))


# Convertir a matrices dispersas para XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train_processed), label = train_df$price_log)
dtest <- xgb.DMatrix(data = as.matrix(X_test_processed))

###########################################
# 4. ESPECIFICACIÓN DE HIPERPARÁMETROS    #
###########################################

# CAMBIO CLAVE: Optimización de Hiperparámetros con caret::train

# Definir la cuadrícula de hiperparámetros a buscar

xgb_grid <- expand.grid(
  nrounds = c(500, 1000, 1500),         # Número de árboles
  eta = c(0.01, 0.05, 0.1),             # Tasa de aprendizaje
  max_depth = c(4, 6, 8),               # Profundidad máxima del árbol
  gamma = c(0, 0.1, 0.2),               # Parámetro de poda (min_loss_reduction)
  colsample_bytree = c(0.6, 0.8),       # Subsampleo de columnas
  min_child_weight = c(1, 5),           # Peso mínimo en un nodo hijo
  subsample = c(0.7, 0.9)               # Subsampleo de filas
)

# Configurar el control del entrenamiento (validación cruzada)

fitControl <- trainControl(
  method = "cv",                   # Validación cruzada
  number = 5,                      # 5-fold CV
  verboseIter = TRUE,              # Mostrar progreso de cada iteración
  allowParallel = TRUE,            # Permitir paralelización si configuras un backend
  returnResamp = "final",          # Guardar resultados para el modelo final
  savePredictions = "final",       # Guardar predicciones para análisis
  search = "grid"                  # Usar grid search 
)

cat("\n--- Búsqueda de Hiperparámetros (Tuning) para XGBoost ---\n")

###########################################
# 5. ENTRENAMIENTO DEL MODELO            #
###########################################

set.seed(123) 
xgb_model_tuned <- train(
  x = X_train_processed,       # Características procesadas
  y = train_df$price_log,      # Variable objetivo log-transformada
  method = "xgbTree",          # Método para XGBoost
  trControl = fitControl,      # Control de entrenamiento (CV)
  tuneGrid = xgb_grid,         # Cuadrícula de hiperparámetros
  metric = "RMSE",             # Métrica a optimizar (caret buscará el menor RMSE)
  verbose = FALSE              # Desactivar la verbosidad interna de xgb.train durante el tuning
)

cat("\nBúsqueda de hiperparámetros completada.\n")
print(xgb_model_tuned) # Muestra los resultados del tuning
cat("\nMejores hiperparámetros encontrados:\n")
print(xgb_model_tuned$bestTune)

###########################################
# 7. PREDICCIONES                         #
###########################################


# Predicciones en el conjunto de entrenamiento con el mejor modelo tuneado
train_preds_log <- predict(xgb_model_tuned, X_train_processed)
train_preds <- expm1(train_preds_log)
rmse_train <- sqrt(mean((train_df$price - train_preds)^2))
r2_train <- caret::R2(train_preds, train_df$price)

cat(sprintf("\nRMSE en el conjunto de entrenamiento (tuneado): %.2f\n", rmse_train))
cat(sprintf("R-cuadrado en el conjunto de entrenamiento (tuneado): %.4f\n", r2_train))

# Obtener el RMSE de validación cruzada para los mejores parámetros:
cat(sprintf("\nRMSE de validación cruzada (mejor modelo): %.2f\n", min(xgb_model_tuned$results$RMSE)))


# Predicción en los Datos de Prueba
cat("\n--- Realizando Predicciones en los Datos de Prueba ---\n")

test_predictions_log <- predict(xgb_model_tuned, X_test_processed)
test_predictions <- expm1(test_predictions_log)

test_predictions[test_predictions < 0] = 0 # Asegurar no tener predicciones negativas

###########################################
# 8. EXPORTACIÓN DE RESULTADOS            #
###########################################

# Crear submission
submission <- data.frame(
  property_id = test_df$property_id,
  price = test_predictions
)

# Verificar submission
cat("Dimensiones del submission:", dim(submission), "\n")
cat("NAs en submission:", sum(is.na(submission$price)), "\n")

# Mostrar preview del submission
cat("Preview del submission:\n")
print(head(submission))

# Exportar submission
write_csv(submission, "stores/submissions/submission_xgb.csv")

cat("Submission exportado exitosamente: stores/submissions/submission_3.csv\n")


################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################