import logging
from pathlib import Path

def setup_logging(logfile: str = "logs/pipeline.log", level=logging.INFO):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode='a'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().info("Logging initialisiert.")
