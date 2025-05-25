################################################################################
# TÍTULO: 07_elastic_net_model.R                                               #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Implementación de modelo Elastic Net para predicción de precios#
# FECHA: 22 de mayo de 2025                                                   #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Subir un nivel directorio para acceder a la estructura principal del proyecto
setwd("../")

# Cargar librerías
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,    # Manipulación de datos
  recipes,      # Preprocesamiento de datos
  parsnip,      # Especificación de modelos
  tune,         # Tuning de hiperparámetros
  rsample,      # Remuestreo y validación cruzada
  workflows,    # Flujos de trabajo
  yardstick,    # Métricas de evaluación
  dials,        # Grillas de parámetros
  workflowsets  # Conjuntos de flujos de trabajo
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

# Imputar valores faltantes en variables de texto con 0
variables_texto <- c("nivel_premium", "nivel_completitud", "nivel_venta_inmediata")

train <- train %>%
  mutate(
    across(all_of(variables_texto), ~ifelse(is.na(.), 0, .))
  )

test <- test %>%
  mutate(
    across(all_of(variables_texto), ~ifelse(is.na(.), 0, .))
  )

# Verificar que no hay valores faltantes en las variables de texto
cat("Missing values en train después de imputación:\n")
train %>% 
  select(all_of(variables_texto)) %>% 
  summarise_all(~sum(is.na(.))) %>% 
  print()

cat("Missing values en test después de imputación:\n")
test %>% 
  select(all_of(variables_texto)) %>% 
  summarise_all(~sum(is.na(.))) %>% 
  print()

# Remover property_id de las variables predictoras
train_features <- train %>% select(-property_id)
test_features <- test %>% select(-property_id, -price) # Remover price si existe en test

cat("Variables disponibles para el modelo:\n")
cat(names(train_features), "\n")

###########################################
# 2. ESPECIFICACIÓN DE RECETAS           #
###########################################

# Receta 1: Modelo básico con todas las variables
rec_1 <- recipe(price ~ ., data = train_features) %>%
  step_zv(all_predictors()) %>%                    # Remover variables con varianza cero
  step_normalize(all_numeric_predictors()) %>%     # Normalizar variables numéricas
  step_dummy(all_nominal_predictors()) %>%        # Crear dummies para categóricas
  step_interact(terms = ~ starts_with("distancia"):is_house) %>%  # Interacciones espaciales
  step_nzv(all_predictors())                      # Remover predictores near-zero variance

# Receta 2: Modelo con términos polinómicos y más interacciones
rec_2 <- recipe(price ~ ., data = train_features) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_poly(starts_with("distancia"), degree = 2) %>%  # Términos cuadráticos para distancias
  step_interact(terms = ~ starts_with("distancia"):is_house) %>%
  step_interact(terms = ~ starts_with("nivel"):is_house) %>%
  step_interact(terms = ~ bedrooms:antiguedad) %>%
  step_nzv(all_predictors())

###########################################
# 3. ESPECIFICACIÓN DEL MODELO           #
###########################################

# Especificación de Elastic Net
elastic_net_spec <- linear_reg(
  penalty = tune(),    # Parámetro de penalización (lambda)
  mixture = tune()     # Parámetro de mezcla (alpha)
) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Grilla de hiperparámetros
grid_values <- grid_regular(
  penalty(range = c(-2, 1)), 
  mixture(),
  levels = c(50, 5)  # 50 valores de penalty, 5 valores de mixture
)

cat("Número total de combinaciones de hiperparámetros:", nrow(grid_values), "\n")

###########################################
# 4. FLUJOS DE TRABAJO                   #
###########################################

# Workflow 1: Receta básica
workflow_1 <- workflow() %>%
  add_recipe(rec_1) %>%
  add_model(elastic_net_spec)

# Workflow 2: Receta con polinomios
workflow_2 <- workflow() %>%
  add_recipe(rec_2) %>%
  add_model(elastic_net_spec)

###########################################
# 5. VALIDACIÓN CRUZADA Y TUNING         #
###########################################

# Configurar validación cruzada 5-fold
set.seed(123)
cv_folds <- vfold_cv(train_features, v = 5, strata = price)

cat("Iniciando tuning de hiperparámetros para Workflow 1...\n")
# Tuning Workflow 1
set.seed(123)
tune_res1 <- tune_grid(
  workflow_1,
  resamples = cv_folds,
  grid = grid_values,
  metrics = metric_set(rmse, mae, rsq),
  control = control_grid(verbose = TRUE)
)

cat("Iniciando tuning de hiperparámetros para Workflow 2...\n")
# Tuning Workflow 2
set.seed(123)
tune_res2 <- tune_grid(
  workflow_2,
  resamples = cv_folds,
  grid = grid_values,
  metrics = metric_set(rmse, mae, rsq),
  control = control_grid(verbose = TRUE)
)

# Mostrar mejores resultados
cat("Mejores métricas Workflow 1:\n")
show_best(tune_res1, metric = "rmse") %>% print()

cat("Mejores métricas Workflow 2:\n")
show_best(tune_res2, metric = "rmse") %>% print()

###########################################
# 6. SELECCIÓN Y AJUSTE FINAL            #
###########################################

# Seleccionar mejores hiperparámetros
best_params_1 <- select_best(tune_res1, metric = "rmse")
best_params_2 <- select_best(tune_res2, metric = "rmse")

cat("Mejores parámetros Workflow 1:\n")
print(best_params_1)

cat("Mejores parámetros Workflow 2:\n")
print(best_params_2)

# Finalizar workflows con mejores parámetros
final_workflow_1 <- finalize_workflow(workflow_1, best_params_1)
final_workflow_2 <- finalize_workflow(workflow_2, best_params_2)

# Ajustar modelos finales en todo el conjunto de entrenamiento
cat("Ajustando modelo final 1...\n")
final_fit_1 <- fit(final_workflow_1, data = train_features)

cat("Ajustando modelo final 2...\n")
final_fit_2 <- fit(final_workflow_2, data = train_features)

###########################################
# 7. PREDICCIONES Y EXPORTACIÓN          #
###########################################

# Crear directorio submissions si no existe
if (!dir.exists("stores/submissions")) {
  dir.create("stores/submissions", recursive = TRUE)
}

# Realizar predicciones
cat("Generando predicciones...\n")
predictions_1 <- predict(final_fit_1, new_data = test_features)
predictions_2 <- predict(final_fit_2, new_data = test_features)

# Crear dataframes para submission
submission_5 <- data.frame(
  property_id = test$property_id,
  price = predictions_1$.pred
)

submission_6 <- data.frame(
  property_id = test$property_id,
  price = predictions_2$.pred
)

# Verificar que no hay valores faltantes en las predicciones
cat("NAs en predicciones modelo 1:", sum(is.na(predictions_1$.pred)), "\n")
cat("NAs en predicciones modelo 2:", sum(is.na(predictions_2$.pred)), "\n")

# Exportar submissions
write_csv(submission_5, "stores/submissions/submission_5.csv")
write_csv(submission_6, "stores/submissions/submission_6.csv")

cat("Submissions exportados exitosamente:\n")
cat("- stores/submissions/submission_5.csv (Modelo Elastic Net básico)\n")
cat("- stores/submissions/submission_6.csv (Modelo Elastic Net con polinomios)\n")

# Mostrar estadísticas de las predicciones
cat("\nEstadísticas de predicciones:\n")
cat("Modelo 1 - Min:", min(predictions_1$.pred), "Max:", max(predictions_1$.pred), 
    "Mean:", mean(predictions_1$.pred), "\n")
cat("Modelo 2 - Min:", min(predictions_2$.pred), "Max:", max(predictions_2$.pred), 
    "Mean:", mean(predictions_2$.pred), "\n")

# Crear directorio models si no existe
if (!dir.exists("stores/models")) {
  dir.create("stores/models", recursive = TRUE)
}

# Guardar información del modelo para referencia
model_info <- list(
  workflow_1_params = best_params_1,
  workflow_2_params = best_params_2,
  workflow_1_metrics = collect_metrics(tune_res1) %>% 
    filter(.config == best_params_1$.config),
  workflow_2_metrics = collect_metrics(tune_res2) %>% 
    filter(.config == best_params_2$.config)
)

saveRDS(model_info, "stores/models/elastic_net_model_info.rds")

cat("\nInformación del modelo guardada en: stores/models/elastic_net_model_info.rds\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################