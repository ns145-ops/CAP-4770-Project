import pandas as pd
from getpass import getpass
from sqlalchemy import create_engine

password = getpass("MySQL password: ")

engine = create_engine("mysql+pymysql://root:" + password + "@localhost/cap4770_project")

df = pd.read_csv("data.csv", dtype={"CRS_DEP_TIME": str})

df.to_sql("flights", engine, if_exists="append", index=False, chunksize=1000)

