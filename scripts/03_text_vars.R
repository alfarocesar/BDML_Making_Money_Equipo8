################################################################################
# TÍTULO: Variables derivadas de texto                                         #
# PROYECTO: Making Money with ML                                               #
# DESCRIPCIÓN: Crea variables basadas en la descripción de propiedades para    #
#              datasets de entrenamiento y prueba.                            #
# FECHA: 2025-07-21                                                            #
# VERSIÓN: 2.0 - Corregida con word boundaries y optimizaciones               #
################################################################################

# Configurar directorio de trabajo automáticamente
if (!require(rstudioapi)) install.packages("rstudioapi")
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd("../")

# Cargar librerías
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  stringi,
  stringr,
  readr,
  dplyr
)

# Fijar semilla
set.seed(123)

################################################################################
#                          Funciones para variables                            #
################################################################################

# Función para limpiar texto
limpiar_texto <- function(texto) {
  texto <- tolower(texto)
  texto <- stri_trans_general(texto, "Latin-ASCII")
  texto <- str_replace_all(texto, "[^[:alnum:] ]", " ")
  texto <- str_replace_all(texto, "\\s+", " ")
  return(str_trim(texto))
}

# Variable 1: Nivel Premium
calcular_premium <- function(desc_limpia) {
  terminos_premium <- c("lujo", "exclusiv", "premium", "vista", "panoram", 
                        "marmol", "granito", "terraza", "jacuzzi", "balcon",
                        "renovad", "modern", "nueva", "piscina", "gimnasio",
                        "seguridad", "vigilancia", "lobby")
  
  conteo <- sum(sapply(terminos_premium, function(termino) {
    str_detect(desc_limpia, paste0("\\b", termino, "\\b"))
  }))
  
  return(min(5, conteo))
}

# Variable 2: Completitud de espacios
calcular_completitud <- function(desc_limpia) {
  espacios <- c("cocina", "sala", "comedor", "habitacion", "dormitorio", 
                "bano", "estudio", "garaje", "parqueadero", "patio", 
                "jardin", "lavanderia")
  
  espacios_mencionados <- sum(sapply(espacios, function(espacio) {
    str_detect(desc_limpia, paste0("\\b", espacio, "\\b"))
  }))
  
  # CORREGIDO: Dividir por la cantidad real de espacios definidos
  return(min(1, espacios_mencionados / length(espacios)))
}

# Variable 3: Nivel de urgencia en la venta
calcular_venta_inmediata <- function(desc_limpia) {
  # SIMPLIFICADO: Eliminadas frases complejas por términos individuales
  terminos_urgencia <- c("negociable", "oportunidad", "remate", "oferta", 
                         "urge", "rebaja", "descuento", "ganga", "especial")
  
  conteo <- sum(sapply(terminos_urgencia, function(termino) {
    str_detect(desc_limpia, paste0("\\b", termino, "\\b"))
  }))
  
  return(min(3, conteo))
}

# Aplicación optimizada a los datos
procesar_datos <- function(datos) {
  datos %>%
    mutate(
      # OPTIMIZADO: Limpiar texto una sola vez
      desc_limpia = limpiar_texto(description),
      # Aplicar funciones al texto ya limpio
      nivel_premium = sapply(desc_limpia, calcular_premium),
      nivel_completitud = sapply(desc_limpia, calcular_completitud),
      nivel_venta_inmediata = sapply(desc_limpia, calcular_venta_inmediata)
    ) %>%
    select(property_id, nivel_premium, nivel_completitud, nivel_venta_inmediata)
}

################################################################################
#                      Importar y procesar los datasets                        #
################################################################################

train <- read_csv("stores/raw/train.csv")
test <- read_csv("stores/raw/test.csv")

train_vars_texto <- procesar_datos(train)
test_vars_texto <- procesar_datos(test)

write_csv(train_vars_texto, "stores/processed/train_vars_texto.csv")
write_csv(test_vars_texto,  "stores/processed/test_vars_texto.csv")

################################################################################
# DICCIONARIO DE VARIABLES CREADAS                                            #
################################################################################

# Variable: nivel_premium
# Tipo: Numérica ordinal discreta
# Rango: 0 a 5
# Descripción: 
#   Cuenta cuántos términos relacionados con lujo y exclusividad aparecen en 
#   la descripción del inmueble. Utiliza word boundaries para evitar falsos
#   positivos (ej. "vista" no detecta "revista").
# Términos incluidos: lujo, exclusiv, premium, vista, panoram, marmol, granito,
#   terraza, jacuzzi, balcon, renovad, modern, nueva, piscina, gimnasio,
#   seguridad, vigilancia, lobby
# Modelos y transformación:
#   - Usable directamente en modelos lineales, árboles, Random Forest, Boosting, 
#     y SuperLearner.
#   - Se recomienda aplicar normalización si se utiliza en redes neuronales.

# Variable: nivel_completitud
# Tipo: Numérica continua (proporción)
# Rango: 0 a 1
# Descripción:
#   Mide la proporción de espacios funcionales mencionados en relación al total
#   de 12 espacios definidos. Incluye sinónimos para capturar variedad 
#   lingüística (ej. "habitacion" y "dormitorio"). Refleja qué tan detallada 
#   es la descripción en términos de distribución del inmueble.
# Espacios incluidos: cocina, sala, comedor, habitacion, dormitorio, bano, 
#   estudio, garaje, parqueadero, patio, jardin, lavanderia
# Modelos y transformación:
#   - Usable directamente en todos los modelos del taller.
#   - Se recomienda normalizar si se utiliza en redes neuronales.

# Variable: nivel_venta_inmediata
# Tipo: Numérica ordinal discreta
# Rango: 0 a 3
# Descripción:
#   Cuenta la aparición de palabras asociadas a urgencia o incentivo de venta.
#   Refleja motivación del vendedor o presión por vender. Se eliminaron frases
#   complejas para mejorar robustez de detección.
# Términos incluidos: negociable, oportunidad, remate, oferta, urge, rebaja,
#   descuento, ganga, especial
# Modelos y transformación:
#   - Usable directamente en todos los algoritmos requeridos.
#   - Requiere normalización si se utiliza en redes neuronales.

################################################################################
# NOTAS DE USO                                                                 #
################################################################################

# Si se usa el paquete 'tidymodels', se sugiere aplicar 
# 'step_normalize(all_numeric_predictors())' en la receta para estandarizar 
# las variables en modelos sensibles a la escala como redes neuronales.

# Para validar el funcionamiento, revisar:
# - summary(train_vars_texto) para verificar rangos de variables
# - table(train_vars_texto$nivel_premium) para verificar distribución
# - sum(is.na(train_vars_texto)) para confirmar ausencia de NAs

cat("✅ Script ejecutado exitosamente.\n")
cat("📁 Archivos generados:\n")
cat("   - stores/processed/train_vars_texto.csv\n")
cat("   - stores/processed/test_vars_texto.csv\n")
cat("📊 Variables creadas: nivel_premium, nivel_completitud, nivel_venta_inmediata\n")

################################################################################
#                            FIN DEL SCRIPT                                   #
################################################################################
