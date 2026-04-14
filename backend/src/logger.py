import sys
from loguru import logger
import os

# Create a logs directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Remove the default logger
logger.remove()

# Add a clean console logger
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add a rotating file logger
logger.add(
    os.path.join(LOG_DIR, "rag_app.log"),
    rotation="10 MB",     # Rotate when file reaches 10MB
    retention="14 days",  # Keep logs for 2 weeks
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)
