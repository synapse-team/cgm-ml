from etl.etl import ETL
import sys


if __name__ == '__main__':
    config_path = sys.argv[1]
    etl_process = ETL()
    etl_process.initialize(config_path)
    etl_process.run()
