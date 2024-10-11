import os
import time

def create_directory_if_not_exists(directory, logger):
    """Cria o diretório se ele não existir."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Diretório criado: {directory}")

def extract_and_retry(func, logger, *args, max_retries=1, **kwargs):
    """Função auxiliar para tentar extrair novamente em caso de falha."""
    for attempt in range(max_retries + 1):
        try:
            func(*args, **kwargs)
            return True  # Sucesso
        except Exception as e:
            logger.error(f"Erro em {func.__name__} na tentativa {attempt + 1}: {str(e)}")
            if attempt < max_retries:
                logger.info(f"Tentando novamente {func.__name__} (tentativa {attempt + 1})...")
                time.sleep(2)  # Atraso antes de tentar novamente
            else:
                logger.error(f"Falhou após {max_retries + 1} tentativas.")
                return False