# rename to config.py

# SQL Server Connection Details
SQL_SERVER = "localhost,1433" 
SQL_DATABASE = "AudioUndistortion"
SQL_USER = "your_username"  # Leave blank if using Windows Authentication
SQL_PASSWORD = "your_password"  # Leave blank if using Windows Authentication
USE_WINDOWS_AUTHENTICATION = True  # Set to False if using SQL authentication

# Directory for Audio File Storage
AUDIO_FILE_DIRECTORY = "./training/data/initial"

# Web Server Configuration
WEB_SERVER_PORT = 5000  # 5000 default port for Flask

# Log File Configuration
LOG_FILE_FOLDER_PATH = "./logs/"
LOG_LEVEL = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
