# View My Project in Google Colab
Google Colab - https://colab.research.google.com/drive/1elEfOHB1IhrtlxXKvk0_FZoTHu766xhe?usp=sharing

This GitHub project is still in progress, but you can view the core ideas and functionality by selecting the above link and going through the project.

# Audio Signal Processing for Undistorting Audio Clips

## Project Summary
This project aims to use neural networks to undistort audio clips, focusing primarily on human speech. It involves processing audio to reduce noise and distortion, thereby improving the clarity and intelligibility of the audio content. The application runs as a simple web server, allowing users to interact with it through a web browser.

## Setup Instructions
### Prerequisites
- Python 3.x
- Dependencies listed in `requirements.txt` (install using `pip install -r requirements.txt`)
- SQL Server (for storing audio file metadata and training progress)
- Web browser (for accessing the GUI)

### Installation
1. Clone the repository to your local machine.
2. Install the required Python packages: `pip install -r requirements.txt`
3. Configure the global variables in `config.py` according to your setup (details below).

### Configuration
Edit the `config.py` file to set up the following configurations:
- SQL Server connection details
- Database name for storing project data
- Directory path for audio file storage
- Port number for the web server

## Global Variables Explanation
Refer to `config.py` for setting global variables. These include:
- SQL Server connection inputs (server name, login credentials or Windows authentication)
- The database name where project data is stored
- The file directory path where audio recordings are stored
- The port number on which the web server will run

Please ensure all variables are correctly set before starting the server.

## Starting the Server
To start the web server, run `python server.py` from the command line. Once the server is running, open a web browser and navigate to `http://localhost:<PORT>` (replace `<PORT>` with the port number you configured).

For detailed information on how to use the application and examples of audio processing, visit the application's homepage.
