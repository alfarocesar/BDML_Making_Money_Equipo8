################################################################################
# TÍTULO: 05_descriptive_analysis.R                                           #
# PROYECTO: Making Money with ML                                              #
# DESCRIPCIÓN: Análisis de estadísticas descriptivas y datos faltantes       #
# FECHA: 22 de mayo de 2025                                                   #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../")

# Cargar librerías
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,  # Manipulación de datos
  readr,      # Lectura de archivos
  knitr,      # Tablas formateadas
  corrplot,   # Gráficos de correlación
  VIM,        # Visualización de datos faltantes
  Hmisc,      # Estadísticas descriptivas avanzadas
  psych,      # Análisis psicológico y estadísticas
  moments,    # Asimetría y curtosis
  gridExtra   # Organización de gráficos
)

# Fijar semilla
set.seed(123)

###########################################
# 1. CARGA DE DATOS                      #
###########################################

cat("\n===================================================================\n")
cat("INICIANDO ANÁLISIS DE ESTADÍSTICAS DESCRIPTIVAS Y DATOS FALTANTES\n")
cat("===================================================================\n")

# Cargar datasets combinados
train_data <- read_csv("stores/processed/train_merged.csv")
test_data <- read_csv("stores/processed/test_merged.csv")

cat("\nDimensiones de los datasets:\n")
cat("Train:", dim(train_data), "\n")
cat("Test :", dim(test_data), "\n")

###########################################
# 2. FUNCIONES AUXILIARES                #
###########################################

# Función para clasificar tipos de variables según el diccionario
classify_variables <- function() {
  list(
    identifiers = c("property_id"),  # Identificadores únicos - NO incluir en modelos
    binary = c("is_house"),
    numeric_discrete = c("bedrooms"),
    numeric_continuous = c("price", "antiguedad", "distancia_parque", "distancia_universidad", 
                           "distancia_estacion_transporte", "distancia_zona_comercial", 
                           "nivel_completitud"),
    ordinal_discrete = c("nivel_premium", "nivel_venta_inmediata")
  )
}

# Función para estadísticas de variables categóricas nominales
stats_categorical_nominal <- function(data, var_name, dataset_name) {
  var_data <- data[[var_name]]
  
  # Número de categorías únicas
  n_unique <- n_distinct(var_data, na.rm = TRUE)
  
  # Frecuencias (solo para variables con pocas categorías)
  if(n_unique <= 20) {
    freq_table <- table(var_data, useNA = "ifany")
    prop_table <- prop.table(freq_table) * 100
    
    freq_df <- data.frame(
      Categoria = names(freq_table),
      Frecuencia = as.numeric(freq_table),
      Porcentaje = round(as.numeric(prop_table), 2)
    )
  } else {
    freq_df <- data.frame(
      Nota = paste("Variable con", n_unique, "categorías únicas - No se muestran frecuencias")
    )
  }
  
  result <- list(
    dataset = dataset_name,
    variable = var_name,
    tipo = "Categórica Nominal",
    categorias_unique = n_unique,
    observaciones = length(var_data),
    frecuencias = freq_df
  )
  
  return(result)
}

# Función para estadísticas de variables binarias
stats_binary <- function(data, var_name, dataset_name) {
  var_data <- data[[var_name]]
  
  freq_table <- table(var_data, useNA = "ifany")
  prop_table <- prop.table(freq_table) * 100
  
  freq_df <- data.frame(
    Valor = names(freq_table),
    Frecuencia = as.numeric(freq_table),
    Porcentaje = round(as.numeric(prop_table), 2)
  )
  
  # Moda
  moda <- names(freq_table)[which.max(freq_table)]
  
  result <- list(
    dataset = dataset_name,
    variable = var_name,
    tipo = "Binaria",
    observaciones = length(var_data),
    moda = moda,
    distribucion = freq_df
  )
  
  return(result)
}

# Función para estadísticas de variables numéricas continuas
stats_numeric_continuous <- function(data, var_name, dataset_name) {
  var_data <- data[[var_name]]
  var_data_clean <- var_data[!is.na(var_data)]
  
  if(length(var_data_clean) == 0) {
    return(list(
      dataset = dataset_name,
      variable = var_name,
      tipo = "Numérica Continua",
      nota = "No hay datos válidos para análisis"
    ))
  }
  
  # Estadísticas de tendencia central
  media <- mean(var_data_clean)
  mediana <- median(var_data_clean)
  
  # Estadísticas de dispersión
  desv_std <- sd(var_data_clean)
  varianza <- var(var_data_clean)
  rango <- max(var_data_clean) - min(var_data_clean)
  coef_var <- (desv_std / media) * 100
  
  # Cuartiles
  q1 <- quantile(var_data_clean, 0.25)
  q3 <- quantile(var_data_clean, 0.75)
  iqr <- q3 - q1
  
  # Estadísticas de forma
  asimetria <- skewness(var_data_clean)
  curtosis <- kurtosis(var_data_clean)
  
  result <- list(
    dataset = dataset_name,
    variable = var_name,
    tipo = "Numérica Continua",
    observaciones = length(var_data),
    obs_validas = length(var_data_clean),
    media = round(media, 4),
    mediana = round(mediana, 4),
    desviacion_std = round(desv_std, 4),
    varianza = round(varianza, 4),
    minimo = min(var_data_clean),
    maximo = max(var_data_clean),
    rango = round(rango, 4),
    q1 = round(q1, 4),
    q3 = round(q3, 4),
    rango_intercuartil = round(iqr, 4),
    coef_variacion = round(coef_var, 2),
    asimetria = round(asimetria, 4),
    curtosis = round(curtosis, 4)
  )
  
  return(result)
}

# Función para estadísticas de variables numéricas discretas
stats_numeric_discrete <- function(data, var_name, dataset_name) {
  var_data <- data[[var_name]]
  var_data_clean <- var_data[!is.na(var_data)]
  
  if(length(var_data_clean) == 0) {
    return(list(
      dataset = dataset_name,
      variable = var_name,
      tipo = "Numérica Discreta",
      nota = "No hay datos válidos para análisis"
    ))
  }
  
  # Estadísticas básicas
  media <- mean(var_data_clean)
  mediana <- median(var_data_clean)
  moda_calc <- as.numeric(names(sort(table(var_data_clean), decreasing = TRUE)[1]))
  
  # Dispersión
  desv_std <- sd(var_data_clean)
  varianza <- var(var_data_clean)
  
  # Cuartiles
  q1 <- quantile(var_data_clean, 0.25)
  q3 <- quantile(var_data_clean, 0.75)
  
  # Distribución de frecuencias
  freq_table <- table(var_data_clean)
  prop_table <- prop.table(freq_table) * 100
  
  freq_df <- data.frame(
    Valor = as.numeric(names(freq_table)),
    Frecuencia = as.numeric(freq_table),
    Porcentaje = round(as.numeric(prop_table), 2)
  )
  
  result <- list(
    dataset = dataset_name,
    variable = var_name,
    tipo = "Numérica Discreta",
    observaciones = length(var_data),
    obs_validas = length(var_data_clean),
    media = round(media, 4),
    mediana = round(mediana, 4),
    moda = moda_calc,
    desviacion_std = round(desv_std, 4),
    varianza = round(varianza, 4),
    minimo = min(var_data_clean),
    maximo = max(var_data_clean),
    q1 = round(q1, 4),
    q3 = round(q3, 4),
    distribucion = freq_df
  )
  
  return(result)
}

# Función para estadísticas de variables ordinales discretas
stats_ordinal_discrete <- function(data, var_name, dataset_name) {
  var_data <- data[[var_name]]
  var_data_clean <- var_data[!is.na(var_data)]
  
  if(length(var_data_clean) == 0) {
    return(list(
      dataset = dataset_name,
      variable = var_name,
      tipo = "Ordinal Discreta",
      nota = "No hay datos válidos para análisis"
    ))
  }
  
  # Para variables ordinales, la mediana es la medida de tendencia central más apropiada
  mediana <- median(var_data_clean)
  
  # Cuartiles
  q1 <- quantile(var_data_clean, 0.25)
  q3 <- quantile(var_data_clean, 0.75)
  iqr <- q3 - q1
  
  # Distribución de frecuencias
  freq_table <- table(var_data_clean)
  prop_table <- prop.table(freq_table) * 100
  
  freq_df <- data.frame(
    Nivel = as.numeric(names(freq_table)),
    Frecuencia = as.numeric(freq_table),
    Porcentaje = round(as.numeric(prop_table), 2)
  )
  
  result <- list(
    dataset = dataset_name,
    variable = var_name,
    tipo = "Ordinal Discreta",
    observaciones = length(var_data),
    obs_validas = length(var_data_clean),
    mediana = round(mediana, 4),
    q1 = round(q1, 4),
    q3 = round(q3, 4),
    rango_intercuartil = round(iqr, 4),
    minimo = min(var_data_clean),
    maximo = max(var_data_clean),
    distribucion = freq_df
  )
  
  return(result)
}

# Función para análisis de datos faltantes
analyze_missing_data <- function(data, dataset_name) {
  cat(paste0("\n--- ANÁLISIS DE DATOS FALTANTES: ", dataset_name, " ---\n"))
  
  # Resumen de datos faltantes por variable
  missing_summary <- data %>%
    summarise_all(~ sum(is.na(.))) %>%
    gather(variable, missing_count) %>%
    mutate(
      missing_pct = round(missing_count / nrow(data) * 100, 2),
      total_obs = nrow(data)
    ) %>%
    arrange(desc(missing_count))
  
  # Estadísticas generales
  total_missing <- sum(missing_summary$missing_count)
  total_cells <- nrow(data) * ncol(data)
  overall_missing_pct <- round(total_missing / total_cells * 100, 2)
  
  # Variables con datos faltantes
  vars_with_missing <- missing_summary %>%
    filter(missing_count > 0)
  
  # Resumen por consola
  cat("Total de observaciones:", nrow(data), "\n")
  cat("Total de variables:", ncol(data), "\n")
  cat("Total de celdas:", total_cells, "\n")
  cat("Total de valores faltantes:", total_missing, "\n")
  cat("Porcentaje global de datos faltantes:", overall_missing_pct, "%\n")
  cat("Variables con datos faltantes:", nrow(vars_with_missing), "\n\n")
  
  if(nrow(vars_with_missing) > 0) {
    cat("Detalle de variables con datos faltantes:\n")
    print(vars_with_missing)
  } else {
    cat("¡Excelente! No hay datos faltantes en este dataset.\n")
  }
  
  return(list(
    dataset = dataset_name,
    resumen_general = list(
      total_obs = nrow(data),
      total_vars = ncol(data),
      total_celdas = total_cells,
      total_faltantes = total_missing,
      pct_faltantes_global = overall_missing_pct
    ),
    detalle_por_variable = missing_summary,
    variables_con_faltantes = vars_with_missing
  ))
}

###########################################
# 3. EJECUCIÓN DEL ANÁLISIS              #
###########################################

cat("\n===================================================================\n")
cat("ANÁLISIS DE DATOS FALTANTES\n")
cat("===================================================================\n")

# Análisis de datos faltantes
missing_train <- analyze_missing_data(train_data, "TRAIN")
missing_test <- analyze_missing_data(test_data, "TEST")

cat("\n===================================================================\n")
cat("ESTADÍSTICAS DESCRIPTIVAS POR TIPO DE VARIABLE\n")
cat("===================================================================\n")

# Obtener clasificación de variables
var_types <- classify_variables()

# Almacenar todos los resultados
all_stats <- list()

# Procesar cada tipo de variable
for (dataset_name in c("train", "test")) {
  current_data <- if(dataset_name == "train") train_data else test_data
  
  cat(paste0("\n--- PROCESANDO DATASET: ", toupper(dataset_name), " ---\n"))
  
  # Variables identificadoras (NO para modelado)
  for (var in var_types$identifiers) {
    if(var %in% names(current_data)) {
      cat(paste0("Procesando ", var, " (identificador)...\n"))
      stats <- stats_categorical_nominal(current_data, var, dataset_name)
      stats$tipo <- "Identificador Único"
      all_stats[[paste0(dataset_name, "_", var)]] <- stats
    }
  }
  
  # Variables binarias
  for (var in var_types$binary) {
    if(var %in% names(current_data)) {
      cat(paste0("Procesando ", var, " (binaria)...\n"))
      stats <- stats_binary(current_data, var, dataset_name)
      all_stats[[paste0(dataset_name, "_", var)]] <- stats
    }
  }
  
  # Variables numéricas continuas
  for (var in var_types$numeric_continuous) {
    if(var %in% names(current_data)) {
      cat(paste0("Procesando ", var, " (numérica continua)...\n"))
      stats <- stats_numeric_continuous(current_data, var, dataset_name)
      all_stats[[paste0(dataset_name, "_", var)]] <- stats
    }
  }
  
  # Variables numéricas discretas
  for (var in var_types$numeric_discrete) {
    if(var %in% names(current_data)) {
      cat(paste0("Procesando ", var, " (numérica discreta)...\n"))
      stats <- stats_numeric_discrete(current_data, var, dataset_name)
      all_stats[[paste0(dataset_name, "_", var)]] <- stats
    }
  }
  
  # Variables ordinales discretas
  for (var in var_types$ordinal_discrete) {
    if(var %in% names(current_data)) {
      cat(paste0("Procesando ", var, " (ordinal discreta)...\n"))
      stats <- stats_ordinal_discrete(current_data, var, dataset_name)
      all_stats[[paste0(dataset_name, "_", var)]] <- stats
    }
  }
}

###########################################
# 4. MOSTRAR RESULTADOS PRINCIPALES      #
###########################################

cat("\n===================================================================\n")
cat("RESUMEN DE ESTADÍSTICAS DESCRIPTIVAS PRINCIPALES\n")
cat("===================================================================\n")

# Función para mostrar estadísticas principales de forma organizada
show_main_stats <- function(stats_list) {
  for (stat_name in names(stats_list)) {
    stat_data <- stats_list[[stat_name]]
    
    cat(paste0("\n", toupper(stat_data$dataset), " - ", toupper(stat_data$variable), 
               " (", stat_data$tipo, ")\n"))
    cat(paste0(rep("-", nchar(paste0(stat_data$dataset, " - ", stat_data$variable, 
                                     " (", stat_data$tipo, ")"))), collapse = ""), "\n")
    
    if(stat_data$tipo == "Binaria") {
      cat("Observaciones:", stat_data$observaciones, "\n")
      cat("Moda:", stat_data$moda, "\n")
      cat("Distribución:\n")
      print(stat_data$distribucion)
      
    } else if(stat_data$tipo == "Numérica Continua") {
      if(!"nota" %in% names(stat_data)) {
        cat("Observaciones:", stat_data$observaciones, "(válidas:", stat_data$obs_validas, ")\n")
        cat("Media:", stat_data$media, "| Mediana:", stat_data$mediana, "\n")
        cat("Desv. Std:", stat_data$desviacion_std, "| Coef. Variación:", stat_data$coef_variacion, "%\n")
        cat("Rango: [", stat_data$minimo, ",", stat_data$maximo, "] | IQR:", stat_data$rango_intercuartil, "\n")
        cat("Asimetría:", stat_data$asimetria, "| Curtosis:", stat_data$curtosis, "\n")
      } else {
        cat(stat_data$nota, "\n")
      }
      
    } else if(stat_data$tipo == "Numérica Discreta") {
      if(!"nota" %in% names(stat_data)) {
        cat("Observaciones:", stat_data$observaciones, "(válidas:", stat_data$obs_validas, ")\n")
        cat("Media:", stat_data$media, "| Mediana:", stat_data$mediana, "| Moda:", stat_data$moda, "\n")
        cat("Desv. Std:", stat_data$desviacion_std, "\n")
        cat("Rango: [", stat_data$minimo, ",", stat_data$maximo, "]\n")
        if(nrow(stat_data$distribucion) <= 10) {
          cat("Distribución:\n")
          print(stat_data$distribucion)
        }
      } else {
        cat(stat_data$nota, "\n")
      }
      
    } else if(stat_data$tipo == "Ordinal Discreta") {
      if(!"nota" %in% names(stat_data)) {
        cat("Observaciones:", stat_data$observaciones, "(válidas:", stat_data$obs_validas, ")\n")
        cat("Mediana:", stat_data$mediana, "\n")
        cat("Q1:", stat_data$q1, "| Q3:", stat_data$q3, "| IQR:", stat_data$rango_intercuartil, "\n")
        cat("Rango: [", stat_data$minimo, ",", stat_data$maximo, "]\n")
        cat("Distribución:\n")
        print(stat_data$distribucion)
      } else {
        cat(stat_data$nota, "\n")
      }
      
    } else if(stat_data$tipo == "Categórica Nominal") {
      cat("Observaciones:", stat_data$observaciones, "\n")
      cat("Categorías únicas:", stat_data$categorias_unique, "\n")
      if("frecuencias" %in% names(stat_data) && nrow(stat_data$frecuencias) > 0) {
        if(!"Nota" %in% names(stat_data$frecuencias)) {
          cat("Frecuencias:\n")
          print(stat_data$frecuencias)
        } else {
          print(stat_data$frecuencias)
        }
      }
    }
  }
}

show_main_stats(all_stats)

###########################################
# 5. GUARDAR RESULTADOS                  #
###########################################

cat("\n===================================================================\n")
cat("GUARDANDO RESULTADOS\n")
cat("===================================================================\n")

# Crear directorio de salida
output_dir <- "stores/processed"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Guardar análisis de datos faltantes
write_csv(missing_train$detalle_por_variable, 
          file.path(output_dir, "missing_data_train.csv"))
write_csv(missing_test$detalle_por_variable, 
          file.path(output_dir, "missing_data_test.csv"))

# Crear resumen de estadísticas principales para variables numéricas
create_numeric_summary <- function(stats_list) {
  numeric_stats <- list()
  
  for(stat_name in names(stats_list)) {
    stat_data <- stats_list[[stat_name]]
    if(stat_data$tipo %in% c("Numérica Continua", "Numérica Discreta") && 
       !"nota" %in% names(stat_data)) {
      
      summary_row <- data.frame(
        dataset = stat_data$dataset,
        variable = stat_data$variable,
        tipo = stat_data$tipo,
        n_total = stat_data$observaciones,
        n_validas = stat_data$obs_validas,
        media = stat_data$media,
        mediana = stat_data$mediana,
        desv_std = stat_data$desviacion_std,
        minimo = stat_data$minimo,
        maximo = stat_data$maximo,
        q1 = stat_data$q1,
        q3 = stat_data$q3
      )
      
      if(stat_data$tipo == "Numérica Continua") {
        summary_row$coef_variacion <- stat_data$coef_variacion
        summary_row$asimetria <- stat_data$asimetria
        summary_row$curtosis <- stat_data$curtosis
      } else {
        summary_row$moda <- stat_data$moda
      }
      
      numeric_stats[[stat_name]] <- summary_row
    }
  }
  
  if(length(numeric_stats) > 0) {
    return(bind_rows(numeric_stats))
  } else {
    return(data.frame())
  }
}

numeric_summary <- create_numeric_summary(all_stats)
if(nrow(numeric_summary) > 0) {
  write_csv(numeric_summary, file.path(output_dir, "numeric_variables_summary.csv"))
}

# Crear resumen de variables categóricas y ordinales
create_categorical_summary <- function(stats_list) {
  categorical_stats <- list()
  
  for(stat_name in names(stats_list)) {
    stat_data <- stats_list[[stat_name]]
    if(stat_data$tipo %in% c("Binaria", "Ordinal Discreta", "Identificador Único")) {
      
      summary_row <- data.frame(
        dataset = stat_data$dataset,
        variable = stat_data$variable,
        tipo = stat_data$tipo,
        n_total = stat_data$observaciones
      )
      
      if(stat_data$tipo == "Binaria") {
        summary_row$moda <- stat_data$moda
      } else if(stat_data$tipo == "Ordinal Discreta") {
        summary_row$mediana <- stat_data$mediana
        summary_row$q1 <- stat_data$q1
        summary_row$q3 <- stat_data$q3
        summary_row$minimo <- stat_data$minimo
        summary_row$maximo <- stat_data$maximo
      } else if(stat_data$tipo == "Identificador Único") {
        summary_row$categorias_unicas <- stat_data$categorias_unique
      }
      
      categorical_stats[[stat_name]] <- summary_row
    }
  }
  
  if(length(categorical_stats) > 0) {
    return(bind_rows(categorical_stats))
  } else {
    return(data.frame())
  }
}

categorical_summary <- create_categorical_summary(all_stats)
if(nrow(categorical_summary) > 0) {
  write_csv(categorical_summary, file.path(output_dir, "categorical_variables_summary.csv"))
}

# Crear archivo con distribuciones de frecuencias
create_frequency_distributions <- function(stats_list) {
  all_distributions <- list()
  
  for(stat_name in names(stats_list)) {
    stat_data <- stats_list[[stat_name]]
    
    if("distribucion" %in% names(stat_data)) {
      dist_data <- stat_data$distribucion
      dist_data$dataset <- stat_data$dataset
      dist_data$variable <- stat_data$variable
      dist_data$tipo <- stat_data$tipo
      
      # Reordenar columnas
      dist_data <- dist_data %>%
        select(dataset, variable, tipo, everything())
      
      all_distributions[[stat_name]] <- dist_data
    }
    
    if("frecuencias" %in% names(stat_data) && 
       is.data.frame(stat_data$frecuencias) && 
       !"Nota" %in% names(stat_data$frecuencias)) {
      
      freq_data <- stat_data$frecuencias
      freq_data$dataset <- stat_data$dataset
      freq_data$variable <- stat_data$variable
      freq_data$tipo <- stat_data$tipo
      
      # Reordenar columnas
      freq_data <- freq_data %>%
        select(dataset, variable, tipo, everything())
      
      all_distributions[[stat_name]] <- freq_data
    }
  }
  
  if(length(all_distributions) > 0) {
    return(bind_rows(all_distributions))
  } else {
    return(data.frame())
  }
}

frequency_distributions <- create_frequency_distributions(all_stats)
if(nrow(frequency_distributions) > 0) {
  write_csv(frequency_distributions, file.path(output_dir, "frequency_distributions.csv"))
}

cat("Archivos guardados exitosamente:\n")
cat("- stores/processed/missing_data_train.csv\n")
cat("- stores/processed/missing_data_test.csv\n")
cat("- stores/processed/numeric_variables_summary.csv\n")
cat("- stores/processed/categorical_variables_summary.csv\n")
cat("- stores/processed/frequency_distributions.csv\n")

###########################################
# 6. RESUMEN EJECUTIVO                   #
###########################################

cat("\n===================================================================\n")
cat("RESUMEN EJECUTIVO DEL ANÁLISIS\n")
cat("===================================================================\n")

cat("\n1. DIMENSIONES DE LOS DATASETS:\n")
cat("   - Train:", nrow(train_data), "observaciones con", ncol(train_data), "variables\n")
cat("   - Test :", nrow(test_data), "observaciones con", ncol(test_data), "variables\n")

cat("\n2. DATOS FALTANTES:\n")
cat("   - Train:", missing_train$resumen_general$pct_faltantes_global, "% de datos faltantes\n")
cat("   - Test :", missing_test$resumen_general$pct_faltantes_global, "% de datos faltantes\n")

cat("\n3. ESTRUCTURA DE VARIABLES:\n")
var_counts <- sapply(var_types, length)
for(type_name in names(var_counts)) {
  cat("   -", gsub("_", " ", str_to_title(type_name)), ":", var_counts[type_name], "\n")
}

cat("\n4. VARIABLE DEPENDIENTE (price):\n")
if("train_price" %in% names(all_stats)) {
  price_stats <- all_stats[["train_price"]]
  if(!"nota" %in% names(price_stats)) {
    cat("   - Media: $", format(price_stats$media, big.mark = ","), "\n")
    cat("   - Mediana: $", format(price_stats$mediana, big.mark = ","), "\n")
    cat("   - Coef. Variación:", price_stats$coef_variacion, "%\n")
  }
}

cat("\n6. NOTAS IMPORTANTES PARA MODELADO:\n")
cat("   - property_id es identificador único: NO incluir en modelos predictivos\n")
cat("   - antiguedad ahora se trata como variable continua para mejor performance\n")
cat("   - Variables espaciales requerirán normalización para redes neuronales\n")
if(missing_train$resumen_general$pct_faltantes_global == 0 && 
   missing_test$resumen_general$pct_faltantes_global == 0) {
  cat("   - No se requiere imputación de datos faltantes\n")
} else {
  cat("   - Considerar estrategias de imputación si es necesario\n")
}

cat("\nAnálisis completado exitosamente.\n")

################################################################################
# DICCIONARIO DE ARCHIVOS GENERADOS                                           #
################################################################################

# missing_data_train.csv
# ----------------------
# Contiene el análisis detallado de datos faltantes para el dataset de entrenamiento.
# Columnas:
# - variable: Nombre de la variable
# - missing_count: Número absoluto de valores faltantes
# - missing_pct: Porcentaje de valores faltantes
# - total_obs: Total de observaciones en el dataset

# missing_data_test.csv
# ---------------------
# Contiene el análisis detallado de datos faltantes para el dataset de prueba.
# Misma estructura que missing_data_train.csv

# numeric_variables_summary.csv
# -----------------------------
# Resumen de estadísticas descriptivas para todas las variables numéricas 
# (continuas y discretas) de ambos datasets.
# Columnas principales:
# - dataset: "train" o "test"
# - variable: Nombre de la variable
# - tipo: "Numérica Continua" o "Numérica Discreta"
# - n_total: Total de observaciones
# - n_validas: Observaciones sin valores faltantes
# - media, mediana, desv_std: Estadísticas de tendencia central y dispersión
# - minimo, maximo: Valores extremos
# - q1, q3: Primer y tercer cuartil
# - Columnas adicionales según el tipo de variable:
#   * Para continuas: coef_variacion, asimetria, curtosis
#   * Para discretas: moda

# categorical_variables_summary.csv
# ---------------------------------
# Resumen de estadísticas descriptivas para variables categóricas, binarias y ordinales.
# Columnas principales:
# - dataset: "train" o "test"
# - variable: Nombre de la variable
# - tipo: "Binaria", "Ordinal Discreta", o "Identificador Único"
# - n_total: Total de observaciones
# - Columnas adicionales según el tipo:
#   * Para binarias: moda
#   * Para ordinales: mediana, q1, q3, minimo, maximo
#   * Para identificadores: categorias_unicas

# frequency_distributions.csv
# ---------------------------
# Contiene las distribuciones de frecuencias para variables discretas y categóricas
# que tienen un número manejable de categorías/valores únicos.
# Columnas principales:
# - dataset: "train" o "test"
# - variable: Nombre de la variable
# - tipo: Tipo de variable
# - Columnas específicas según el tipo de variable:
#   * Para discretas: Valor, Frecuencia, Porcentaje
#   * Para categóricas: Categoria, Frecuencia, Porcentaje
#   * Para ordinales: Nivel, Frecuencia, Porcentaje

# NOTAS IMPORTANTES:
# - Todos los archivos están en formato CSV para fácil importación
# - Los porcentajes están redondeados a 2 decimales
# - Los valores faltantes se reportan tanto en conteo como en porcentaje
# - La variable 'antiguedad' se clasifica como continua según especificaciones
# - La variable 'property_id' se marca como identificador para exclusión del modelado

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################