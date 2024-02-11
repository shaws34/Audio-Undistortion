import pyodbc
from config import SQL_SERVER_DATABASE_CONFIG

# Function to check and perform initial database setup
def setup_database():
    conn_str = f"""
        DRIVER=ODBC Driver 17 for SQL Server;
        SERVER={SQL_SERVER_DATABASE_CONFIG['SQL_SERVER']};
        DATABASE=master;
        UID={SQL_SERVER_DATABASE_CONFIG['SQL_USER']};
        PWD={SQL_SERVER_DATABASE_CONFIG['SQL_PASSWORD']};
    """ if SQL_SERVER_DATABASE_CONFIG['USE_WINDOWS_AUTHENTICATION'] else f"""
        DRIVER=ODBC Driver 17 for SQL Server;
        SERVER={SQL_SERVER_DATABASE_CONFIG['SQL_SERVER']};
        DATABASE=master;
        Trusted_Connection=yes;
    """
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE {SQL_SERVER_DATABASE_CONFIG['SQL_DATABASE']}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Database setup completed successfully.")
