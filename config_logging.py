# meu_projeto/config_logging.py

import logging
import sys

# --- 1. Definir o Formato da Mensagem ---
# Define como cada linha de log será formatada.
# asctime: Data e hora do log.
# name: Nome do logger (útil em projetos com múltiplos loggers).
# levelname: Nível do log (INFO, ERROR, etc.).
# message: A mensagem de log que você escreveu.
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- 2. Configurar o Handler para o arquivo de INFO ---
# FileHandler para escrever logs de nível INFO e acima no arquivo 'info.log'.
info_handler = logging.FileHandler('logs/info.log')
info_handler.setLevel(logging.INFO)  # Processa apenas logs INFO e mais críticos (WARNING, ERROR, CRITICAL).
info_handler.setFormatter(log_formatter)


# --- 3. Configurar o Handler para o arquivo de ERROR ---
# FileHandler para escrever logs de nível ERROR e acima no arquivo 'error.log'.
error_handler = logging.FileHandler('logs/error.log')
error_handler.setLevel(logging.ERROR) # Processa apenas logs ERROR e CRITICAL.
error_handler.setFormatter(log_formatter)



# --- 4. Configurar o Handler para o Console (para desenvolvimento) ---
# StreamHandler para exibir todos os logs no console.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG) # Mostra tudo a partir de DEBUG.
console_handler.setFormatter(log_formatter)


# --- 5. Obter e Configurar o Logger Principal ---
# Obtemos um logger com um nome específico para nosso app.
# É uma boa prática usar nomes para não interferir com loggers de bibliotecas.
logger = logging.getLogger('project')
logger.setLevel(logging.DEBUG) # Define o nível mais baixo; os handlers farão a filtragem final.

# Evita duplicar handlers se este módulo for importado várias vezes.
if not logger.handlers:
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

# Você pode exportar apenas o logger.
# from config_logging import logger