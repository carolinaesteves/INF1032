import logging
from extraction_and_treatment.data_extraction import extraction
from extraction_and_treatment.data_treatment import treatment
from utils.logging import setup_logging

def main():
    # Setup logging
    setup_logging(app_name='data_processing')

    logging.info("Comecando Aplicacao ...")

    try:
        # Start the extraction process
        extraction()

        # Start the treatment process
        treatment()

        logging.info("Aplicacao rodou com Sucesso.")
    except Exception as e:
        logging.error(f"Aplicacao falhou com Erro: {str(e)}")

if __name__ == "__main__":
    main()
