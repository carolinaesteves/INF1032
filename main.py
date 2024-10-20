import logging
from extraction_and_treatment.data_extraction import extraction
from extraction_and_treatment.data_treatment import treatment
from utils.logging import setup_logging
from validation.validateTreatedData import validate_all_treated_data
from models.models import run_models
from eda.eda import run_eda

def main():
    # Setup logging
    setup_logging(app_name='data_processing')

    logging.info("Comecando Aplicacao ...")

    try:
        # Start the extraction process
        #extraction()

        # Start the treatment process
        treatment()

        validate_all_treated_data()
        
        # Run EDA
        run_eda()  # Chame a função de EDA

        # model 1 - Preencher com primeiro modelo ou validação dos dados
        #run_models()

        logging.info("Aplicacao rodou com Sucesso.")
    except Exception as e:
        logging.error(f"Aplicacao falhou com Erro: {str(e)}")

if __name__ == "__main__":
    main()
