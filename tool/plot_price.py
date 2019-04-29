import matplotlib.pyplot as plt
import sqlite3
import pandas.io.sql as psql
import sys

con = sqlite3.connect(sys.argv[1])
cur = con.cursor()
board_mid_df = psql.read_sql('SELECT mid_price FROM boards;', con) 
board_mid = board_mid_df.values

mid_list = []
for i in range(len(board_mid)):
    mid_list.append(board_mid[i].flatten()[0])

print(len(mid_list))

plt.plot(mid_list)
plt.xlabel("step")
plt.ylabel("price")
plt.show()
